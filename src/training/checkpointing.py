"""Best-checkpoint management.

One rule: **the best checkpoint is the epoch with the highest validation
Punctuation Macro-F1** (mean F1 of COMMA/PERIOD/QUESTION). Not accuracy, not
loss — see :mod:`src.evaluation.metrics` for why.

Saved layout
------------
PhoBERT (E2-E4) - a normal Hugging Face folder, so downstream inference can do
``AutoModelForTokenClassification.from_pretrained("outputs/checkpoints/E2")``::

    outputs/checkpoints/E2/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer / bpe.codes / vocab.txt / …
    └── checkpoint_metadata.json

BiLSTM (E1) — no HF format exists, so a plain state dict plus the vocabulary
that produced its input ids::

    outputs/checkpoints/E1/
    ├── model.pt                 (state_dict + model config)
    ├── vocabulary.json
    └── checkpoint_metadata.json

``checkpoint_metadata.json`` always records: experiment id, model name and
revision, label mapping, full training config, best epoch, best validation
score, dataset hashes, seed, and the Python/PyTorch/Transformers/CUDA/GPU
versions — everything needed to explain or rerun the result.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.data.constants import ID2LABEL, LABEL2ID, MONITOR_METRIC
from src.utils.environment import collect_environment
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

__all__ = ["CheckpointManager", "CheckpointMetadata"]


@dataclass
class CheckpointMetadata:
    experiment_id: str
    model_type: str
    model_name: str
    model_revision: Optional[str]
    monitor_metric: str
    best_epoch: int
    best_score: float
    seed: int
    label2id: Dict[str, int] = field(default_factory=lambda: dict(LABEL2ID))
    id2label: Dict[str, str] = field(
        default_factory=lambda: {str(k): v for k, v in ID2LABEL.items()}
    )
    training_config: Dict[str, Any] = field(default_factory=dict)
    data_hashes: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    best_validation_metrics: Dict[str, Any] = field(default_factory=dict)
    saved_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        env = self.environment or {}
        return {
            "experiment_id": self.experiment_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "monitor_metric": self.monitor_metric,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_validation_metrics": self.best_validation_metrics,
            "seed": self.seed,
            "training_config": self.training_config,
            "data_hashes": self.data_hashes,
            "saved_at_utc": self.saved_at_utc
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": env.get("python_version"),
            "torch_version": env.get("torch_version"),
            "transformers_version": env.get("transformers_version"),
            "cuda_version": env.get("cuda_runtime_version"),
            "cuda_available": env.get("cuda_available"),
            "gpu_name": env.get("gpu_name"),
            "environment": env,
        }


class CheckpointManager:
    """Keeps only the best checkpoint, by validation Punctuation Macro-F1."""

    def __init__(
        self,
        experiment_id: str,
        checkpoint_dir: PathLike,
        *,
        model_type: str,
        model_name: str,
        model_revision: Optional[str] = None,
        monitor: str = MONITOR_METRIC,
        mode: str = "max",
        seed: int = 42,
        training_config: Optional[Dict[str, Any]] = None,
        data_hashes: Optional[Dict[str, Any]] = None,
        tokenizer: Any = None,
        vocabulary: Any = None,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.experiment_id = experiment_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.model_name = model_name
        self.model_revision = model_revision
        self.monitor = monitor
        self.mode = mode
        self.seed = seed
        self.training_config = training_config or {}
        self.data_hashes = data_hashes or {}
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary

        self.best_score: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.best_metrics: Dict[str, Any] = {}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


    def is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def maybe_save(
        self,
        model: torch.nn.Module,
        *,
        epoch: int,
        metrics: Dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Save the model iff ``metrics[self.monitor]`` improved. Returns saved?"""
        if self.monitor not in metrics:
            raise KeyError(
                f"Monitored metric {self.monitor!r} missing from validation metrics "
                f"(available: {sorted(k for k in metrics if isinstance(metrics[k], (int, float)))})"
            )
        score = float(metrics[self.monitor])
        if not (force or self.is_improvement(score)):
            logger.info(
                "Epoch %d: %s=%.6f did not beat best=%.6f (epoch %d) — not saving",
                epoch,
                self.monitor,
                score,
                self.best_score,
                self.best_epoch,
            )
            return False

        previous = self.best_score
        self.best_score = score
        self.best_epoch = epoch
        self.best_metrics = metrics

        self._write_checkpoint(model)
        logger.info(
            "Epoch %d: new best %s=%.6f (was %.6f) — checkpoint written to %s",
            epoch,
            self.monitor,
            score,
            previous,
            self.checkpoint_dir,
        )
        return True


    def _write_checkpoint(self, model: torch.nn.Module) -> None:
        self._clear_dir()
        if self.model_type == "phobert":
            saver = getattr(model, "save_pretrained", None)
            if saver is None:
                raise AttributeError("PhoBERT model has no save_pretrained()")
            saver(self.checkpoint_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.checkpoint_dir)
        else:
            payload = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_config": (
                    model.config.to_dict() if hasattr(model, "config") else {}
                ),
                "label2id": dict(LABEL2ID),
            }
            torch.save(payload, self.checkpoint_dir / "model.pt")
            if self.vocabulary is not None:
                self.vocabulary.save(self.checkpoint_dir / "vocabulary.json")

        self.write_metadata()

    def _clear_dir(self) -> None:
        """Remove the previous best so only one checkpoint ever lives here."""
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return
        for item in self.checkpoint_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    def write_metadata(self) -> Path:
        meta = CheckpointMetadata(
            experiment_id=self.experiment_id,
            model_type=self.model_type,
            model_name=self.model_name,
            model_revision=self.model_revision,
            monitor_metric=self.monitor,
            best_epoch=self.best_epoch,
            best_score=self.best_score,
            seed=self.seed,
            training_config=self.training_config,
            data_hashes=self.data_hashes,
            environment=collect_environment(),
            best_validation_metrics={
                k: v
                for k, v in self.best_metrics.items()
                if k not in ("confusion_matrix",)
            },
        )
        return write_json(self.checkpoint_dir / "checkpoint_metadata.json", meta.to_dict())


    @staticmethod
    def read_metadata(checkpoint_dir: PathLike) -> Dict[str, Any]:
        path = Path(checkpoint_dir) / "checkpoint_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint metadata at {path}")
        return read_json(path)

    @staticmethod
    def has_checkpoint(checkpoint_dir: PathLike) -> bool:
        d = Path(checkpoint_dir)
        if not d.exists():
            return False
        return (d / "checkpoint_metadata.json").exists() and (
            (d / "model.pt").exists()
            or (d / "model.safetensors").exists()
            or (d / "pytorch_model.bin").exists()
        )
