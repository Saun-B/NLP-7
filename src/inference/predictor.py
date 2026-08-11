from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from src.data.constants import (
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    MAX_WORDS_PER_EXAMPLE,
    PAD_ID,
    PROJECT_ROOT,
)
from src.inference.reconstruction import ReconstructionResult, reconstruct_with_details
from src.inference.tokenizer import InferenceTokenizer, TokenizedInput
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

__all__ = ["PunctuationRestorationPredictor", "RestorationResult"]

@dataclass
class RestorationResult:
    input_text: str
    restored_text: str
    words: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    num_words: int = 0
    num_sentences: int = 0
    num_commas: int = 0
    num_periods: int = 0
    num_questions: int = 0
    num_windows: int = 1
    latency_ms: float = 0.0
    experiment_id: str = ""
    model_name: str = ""
    model_type: str = ""
    device: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "restored_text": self.restored_text,
            "num_words": self.num_words,
            "num_sentences": self.num_sentences,
            "num_commas": self.num_commas,
            "num_periods": self.num_periods,
            "num_questions": self.num_questions,
            "num_windows": self.num_windows,
            "latency_ms": round(self.latency_ms, 2),
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
        }

class PunctuationRestorationPredictor:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_type: str,
        encoder: Any,
        metadata: Dict[str, Any],
        device: torch.device,
        selection: Optional[Dict[str, Any]] = None,
        max_words_per_chunk: int = MAX_WORDS_PER_EXAMPLE,
        batch_size: int = 16,
        use_amp: Optional[bool] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.encoder = encoder
        self.metadata = metadata
        self.device = device
        self.selection = selection or {}
        self.max_words_per_chunk = max_words_per_chunk
        self.batch_size = batch_size
        self.use_amp = (device.type == "cuda") if use_amp is None else bool(use_amp)
        self.text_tokenizer = InferenceTokenizer()

        self.model.eval()
        self.model.to(device)

        checkpoint_labels = metadata.get("label2id")
        if checkpoint_labels and dict(checkpoint_labels) != dict(LABEL2ID):
            raise ValueError(
                f"Checkpoint label mapping {checkpoint_labels} differs from the project "
                f"mapping {LABEL2ID}; predictions would be mislabelled."
            )

    @classmethod
    def from_selected_model(
        cls,
        model_selection_path: Optional[PathLike] = None,
        *,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> "PunctuationRestorationPredictor":
        """Load the **locked winner** named in ``model_selection.json``."""
        from src.evaluation.selection import (
            MODEL_SELECTION_PATH,
            load_locked_winner,
            resolve_winner_checkpoint,
        )

        path = Path(model_selection_path) if model_selection_path else MODEL_SELECTION_PATH
        selection = load_locked_winner(path)
        checkpoint_dir = resolve_winner_checkpoint(selection, path=path)
        logger.info(
            "Loading locked winner %s from %s", selection["winner"], checkpoint_dir
        )
        return cls.from_checkpoint(
            checkpoint_dir, device=device, selection=selection, **kwargs
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: PathLike,
        *,
        device: Optional[torch.device] = None,
        selection: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "PunctuationRestorationPredictor":
        from src.evaluation.loaders import load_checkpoint_encoder
        from src.models.factory import load_model_from_checkpoint

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_type, encoder, metadata = load_checkpoint_encoder(checkpoint_dir)
        model, _ = load_model_from_checkpoint(checkpoint_dir, device=device)

        return cls(
            model=model,
            model_type=model_type,
            encoder=encoder,
            metadata=metadata,
            device=device,
            selection=selection,
            **kwargs,
        )

    @property
    def experiment_id(self) -> str:
        return str(self.metadata.get("experiment_id", "unknown"))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    def describe(self) -> Dict[str, Any]:
        """Model card shown by the CLI and the UI sidebar."""
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_revision": self.metadata.get("model_revision"),
            "weight_mode": (self.metadata.get("training_config", {}) or {})
            .get("loss", {})
            .get("weight_mode"),
            "best_epoch": self.metadata.get("best_epoch"),
            "validation_punctuation_macro_f1": self.metadata.get("best_score"),
            "seed": self.metadata.get("seed"),
            "device": str(self.device),
            "fp16": self.use_amp,
            "selected_on": self.selection.get("selection_split"),
            "selection_metric": self.selection.get("selection_metric"),
            "winner_locked": self.selection.get("winner_locked"),
            "checkpoint_path": self.metadata.get("checkpoint_path")
            or self.selection.get("checkpoint_path"),
        }

    @torch.no_grad()
    def predict_labels(self, words: Sequence[str]) -> List[str]:
        """One label per word. Never truncates; never returns a short list."""
        if not words:
            return []
        model_words = [w.lower() for w in words]

        if self.model_type == "phobert":
            labels, _ = self._predict_phobert(model_words)
        else:
            labels, _ = self._predict_bilstm(model_words)

        if len(labels) != len(words):
            raise AssertionError(
                f"predicted {len(labels)} labels for {len(words)} words — "
                "the word/label alignment is broken."
            )
        return labels

    def _predict_phobert(self, words: Sequence[str]) -> tuple[List[str], int]:
        windows = self.encoder.encode_words(words)
        labels: List[Optional[str]] = [None] * len(words)

        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            max_len = max(len(w.input_ids) for w in batch)
            pad_id = self.encoder.pad_id

            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, w in enumerate(batch):
                n = len(w.input_ids)
                input_ids[i, :n] = torch.tensor(w.input_ids, dtype=torch.long)
                attention[i, :n] = 1

            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(input_ids=input_ids, attention_mask=attention)
            preds = logits.float().argmax(dim=-1).cpu()

            for i, w in enumerate(batch):
                for pos, word_idx in enumerate(w.word_index):
                    if word_idx >= 0:
                        labels[word_idx] = ID2LABEL[int(preds[i, pos])]

        missing = [i for i, l in enumerate(labels) if l is None]
        if missing:
            raise AssertionError(f"No prediction produced for word index/indices {missing[:10]}")
        return [l for l in labels if l is not None], len(windows)

    def _predict_bilstm(self, words: Sequence[str]) -> tuple[List[str], int]:
        chunks = [
            list(words[i : i + self.max_words_per_chunk])
            for i in range(0, len(words), self.max_words_per_chunk)
        ]
        labels: List[str] = []

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            lengths = [len(c) for c in batch]
            max_len = max(lengths)

            input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
            attention = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, chunk in enumerate(batch):
                ids = self.encoder.encode(chunk)
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[i, : len(ids)] = 1

            logits = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention.to(self.device),
                lengths=torch.tensor(lengths, dtype=torch.long),
            )
            preds = logits.float().argmax(dim=-1).cpu()

            for i, n in enumerate(lengths):
                labels.extend(ID2LABEL[int(preds[i, j])] for j in range(n))

        return labels, len(chunks)

    def restore(
        self,
        text: str,
        *,
        capitalize: bool = True,
        ensure_final_punctuation: bool = True,
        max_words: Optional[int] = None,
    ) -> RestorationResult:
        started = time.perf_counter()
        tokenized: TokenizedInput = self.text_tokenizer.tokenize(text or "")

        if tokenized.is_empty:
            return RestorationResult(
                input_text=text or "",
                restored_text="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                experiment_id=self.experiment_id,
                model_name=self.model_name,
                model_type=self.model_type,
                device=str(self.device),
            )

        if max_words is not None and len(tokenized) > max_words:
            raise ValueError(
                f"Input has {len(tokenized):,} words, above the configured limit of "
                f"{max_words:,}. The text is NOT truncated — split it and call again."
            )

        labels = self.predict_labels(tokenized.model_words)
        details: ReconstructionResult = reconstruct_with_details(
            tokenized.words,
            labels,
            capitalize=capitalize,
            ensure_final_punctuation=ensure_final_punctuation,
        )
        latency = (time.perf_counter() - started) * 1000.0

        num_windows = 1
        if self.model_type == "phobert":
            num_windows = max(1, len(self.encoder.encode_words(tokenized.model_words)))
        else:
            num_windows = max(
                1,
                (len(tokenized.model_words) + self.max_words_per_chunk - 1)
                // self.max_words_per_chunk,
            )

        return RestorationResult(
            input_text=text,
            restored_text=details.text,
            words=list(tokenized.words),
            labels=list(labels),
            num_words=details.num_words,
            num_sentences=details.num_sentences,
            num_commas=details.num_commas,
            num_periods=details.num_periods,
            num_questions=details.num_questions,
            num_windows=num_windows,
            latency_ms=latency,
            experiment_id=self.experiment_id,
            model_name=self.model_name,
            model_type=self.model_type,
            device=str(self.device),
        )

    def restore_batch(self, texts: Sequence[str], **kwargs: Any) -> List[RestorationResult]:
        return [self.restore(t, **kwargs) for t in texts]

    def warmup(self, text: str = "xin chào tôi đang thử mô hình khôi phục dấu câu") -> float:
        """Run one throwaway inference so timings exclude lazy CUDA init."""
        started = time.perf_counter()
        self.restore(text)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info("Warm-up inference took %.1f ms", elapsed)
        return elapsed
