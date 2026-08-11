from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
from src.data.constants import PAD_ID, PHOBERT_MAX_LENGTH, PROJECT_ROOT
from src.data.dataset import (
    BiLSTMDataset,
    PhoBERTDataset,
    PhoBERTEncoder,
    Vocabulary,
    collate_bilstm,
    collate_phobert,
)
from src.utils.environment import disable_progress_bars
from src.utils.io import read_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

__all__ = ["build_eval_dataloader", "load_checkpoint_encoder"]

def _resolve_max_length(checkpoint_dir: Path, meta: Dict[str, Any], fallback: int) -> int:
    cfg = meta.get("training_config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    value = model_cfg.get("max_length")
    if value:
        return int(value)

    experiment_id = meta.get("experiment_id")
    if experiment_id:
        exp_config = PROJECT_ROOT / "outputs" / "experiments" / experiment_id / "config.json"
        if exp_config.exists():
            try:
                return int(read_json(exp_config)["config"]["model"]["max_length"])
            except (KeyError, TypeError, ValueError):
                pass
    return fallback

def load_checkpoint_encoder(
    checkpoint_dir: PathLike, *, fallback_max_length: int = PHOBERT_MAX_LENGTH
) -> Tuple[str, Any, Dict[str, Any]]:
    directory = Path(checkpoint_dir)
    meta_path = directory / "checkpoint_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint_metadata.json in {directory}")
    meta = read_json(meta_path)
    model_type = meta.get("model_type")

    if model_type == "bilstm":
        vocab_path = directory / "vocabulary.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"{directory}: a BiLSTM checkpoint must ship vocabulary.json"
            )
        vocab = Vocabulary.load(vocab_path)
        logger.info("Loaded BiLSTM vocabulary (%d types) from %s", len(vocab), directory.name)
        return "bilstm", vocab, meta

    if model_type == "phobert":
        disable_progress_bars()
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(directory)
        except Exception as exc:
            from src.models.phobert import load_phobert_tokenizer

            logger.warning(
                "%s: tokenizer not loadable from the checkpoint (%s); falling back to the "
                "pinned hub revision.",
                directory.name,
                exc,
            )
            tokenizer = load_phobert_tokenizer()

        max_length = _resolve_max_length(directory, meta, fallback_max_length)
        encoder = PhoBERTEncoder(tokenizer, max_length=max_length)
        logger.info(
            "Loaded PhoBERT tokenizer from %s (max_length=%d)", directory.name, max_length
        )
        return "phobert", encoder, meta

    raise ValueError(f"Unknown model_type {model_type!r} in {meta_path}")


def build_eval_dataloader(
    checkpoint_dir: PathLike,
    examples: Sequence[Any],
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    encoder: Optional[Any] = None,
    model_type: Optional[str] = None,
) -> Tuple[DataLoader, str, Any]:
    if encoder is None or model_type is None:
        model_type, encoder, _ = load_checkpoint_encoder(checkpoint_dir)

    if model_type == "bilstm":
        dataset = BiLSTMDataset(examples, encoder)
        collate = lambda batch: collate_bilstm(batch, pad_id=PAD_ID)
    else:
        dataset = PhoBERTDataset(examples, encoder)
        pad_id = encoder.pad_id
        collate = lambda batch: collate_phobert(batch, pad_id=pad_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )
    return loader, model_type, encoder
