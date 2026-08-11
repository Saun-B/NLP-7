"""Experiment artifact writer.

Every experiment folder ``outputs/experiments/<E>/`` ends up with the same set
of files, whichever model produced them::

    config.json                        resolved config + model/loss description
    training_history.csv               one row per epoch
    best_validation_metrics.json       metrics of the best epoch
    per_class_validation_metrics.csv   precision/recall/F1/support per class
    validation_confusion_matrix.csv    gold × predicted counts
    environment.json                   python/torch/transformers/CUDA/GPU
    data_hashes.json                   SHA-256 of the exact data used
    experiment_summary.json            status + headline numbers
    validation_sample_predictions.csv  a few gold-vs-predicted examples
    training_curves.png                loss / PUNCT-F1 curves

``experiment_summary.json`` starts life as ``{"status": "NOT_YET_RUN"}`` and is
only overwritten with real numbers by
:meth:`ExperimentArtifactWriter.write_summary`, which the notebook calls **after**
training actually finished. Nothing here ever invents a metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.data.constants import EXPERIMENT_DIR, LABELS, MONITOR_METRIC
from src.evaluation.metrics import confusion_matrix_header, confusion_matrix_rows
from src.utils.environment import collect_environment
from src.utils.io import project_relative_path, read_json, write_csv, write_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

STATUS_NOT_RUN = "NOT_YET_RUN"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

__all__ = ["ExperimentArtifactWriter", "STATUS_NOT_RUN", "STATUS_COMPLETED"]


@dataclass
class EpochRecord:
    """One row of ``training_history.csv``."""

    epoch: int
    train_loss: float
    learning_rate: float
    epoch_seconds: float
    validation: Dict[str, Any] = field(default_factory=dict)
    is_best: bool = False

    def to_row(self) -> Dict[str, Any]:
        from src.evaluation.metrics import flatten_metrics

        row: Dict[str, Any] = {
            "epoch": self.epoch,
            "train_loss": round(float(self.train_loss), 6),
            "learning_rate": float(self.learning_rate),
            "epoch_seconds": round(float(self.epoch_seconds), 2),
            "is_best": int(bool(self.is_best)),
        }
        for key, value in flatten_metrics(self.validation, prefix="val_").items():
            row[key] = round(float(value), 6)
        row["val_num_evaluated_tokens"] = int(
            self.validation.get("num_evaluated_tokens", 0)
        )
        return row


class ExperimentArtifactWriter:
    """Owns everything written under ``outputs/experiments/<experiment_id>``."""

    def __init__(
        self,
        experiment_id: str,
        *,
        base_dir: PathLike = EXPERIMENT_DIR,
        monitor_metric: str = MONITOR_METRIC,
    ):
        self.experiment_id = experiment_id
        self.dir = Path(base_dir) / experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = monitor_metric
        self.history: List[EpochRecord] = []


    def path(self, filename: str) -> Path:
        return self.dir / filename


    def init_not_run(self) -> None:
        """Write the honest placeholder summary (used when scaffolding)."""
        if self.path("experiment_summary.json").exists():
            existing = read_json(self.path("experiment_summary.json"))
            if existing.get("status") == STATUS_COMPLETED:
                return
        write_json(
            self.path("experiment_summary.json"),
            {
                "status": STATUS_NOT_RUN,
                "experiment_id": self.experiment_id,
                "note": (
                    "This experiment has not been trained yet. Open the matching "
                    "notebook in notebooks/ and run it end to end."
                ),
            },
        )

    def write_config(self, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
        blob = {"experiment_id": self.experiment_id, "config": config}
        if extra:
            blob.update(extra)
        blob["written_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return write_json(self.path("config.json"), blob)

    def write_environment(self) -> Path:
        return write_json(self.path("environment.json"), collect_environment())

    def write_data_hashes(self, hashes: Dict[str, Any]) -> Path:
        return write_json(self.path("data_hashes.json"), hashes)

    def mark_running(self) -> Path:
        return write_json(
            self.path("experiment_summary.json"),
            {
                "status": STATUS_RUNNING,
                "experiment_id": self.experiment_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )


    def add_epoch(self, record: EpochRecord) -> None:
        self.history.append(record)
        self.write_history()

    def write_history(self) -> Optional[Path]:
        if not self.history:
            return None
        rows = [r.to_row() for r in self.history]
        header = list(rows[0].keys())
        return write_csv(
            self.path("training_history.csv"),
            header,
            [[row.get(col, "") for col in header] for row in rows],
        )


    def write_best_validation(self, metrics: Dict[str, Any], *, best_epoch: int) -> Path:
        blob = {
            "experiment_id": self.experiment_id,
            "best_epoch": best_epoch,
            "monitor_metric": self.monitor_metric,
            "evaluated_on": "validation.jsonl (official JointCapPunc dev split)",
            "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        }
        return write_json(self.path("best_validation_metrics.json"), blob)

    def write_per_class(self, metrics: Dict[str, Any]) -> Path:
        header = ["label", "precision", "recall", "f1", "support", "tp", "fp", "fn"]
        rows = []
        for label in LABELS:
            m = metrics["per_class"][label]
            rows.append(
                [
                    label,
                    round(m["precision"], 6),
                    round(m["recall"], 6),
                    round(m["f1"], 6),
                    int(m["support"]),
                    int(m["tp"]),
                    int(m["fp"]),
                    int(m["fn"]),
                ]
            )
        rows.append(
            [
                "PUNCTUATION_MACRO_F1",
                "",
                "",
                round(metrics["punctuation_macro_f1"], 6),
                "",
                "",
                "",
                "",
            ]
        )
        rows.append(["MACRO_F1_ALL_CLASSES", "", "", round(metrics["macro_f1"], 6), "", "", "", ""])
        rows.append(["ACCURACY", "", "", round(metrics["accuracy"], 6), "", "", "", ""])
        return write_csv(self.path("per_class_validation_metrics.csv"), header, rows)

    def write_confusion_matrix(self, metrics: Dict[str, Any]) -> Path:
        cm = metrics["confusion_matrix"]
        return write_csv(
            self.path("validation_confusion_matrix.csv"),
            confusion_matrix_header(LABELS),
            confusion_matrix_rows(cm, LABELS),
        )

    def write_sample_predictions(self, rows: Sequence[Dict[str, Any]]) -> Optional[Path]:
        if not rows:
            return None
        header = list(rows[0].keys())
        return write_csv(
            self.path("validation_sample_predictions.csv"),
            header,
            [[r.get(c, "") for c in header] for r in rows],
        )

    def write_summary(
        self,
        *,
        model: str,
        weight_mode: str,
        best_epoch: int,
        best_score: float,
        checkpoint_path: PathLike,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write the real ``experiment_summary.json`` — only after training."""
        blob: Dict[str, Any] = {
            "status": STATUS_COMPLETED,
            "experiment_id": self.experiment_id,
            "model": model,
            "weight_mode": weight_mode,
            "best_epoch": int(best_epoch),
            "best_validation_punctuation_macro_f1": float(best_score),
            "checkpoint_path": project_relative_path(checkpoint_path),
            "monitor_metric": self.monitor_metric,
            "num_epochs_completed": len(self.history),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if extra:
            blob.update(extra)
        logger.info("Experiment %s COMPLETED: %s", self.experiment_id, blob)
        return write_json(self.path("experiment_summary.json"), blob)

    def write_failure(self, message: str) -> Path:
        return write_json(
            self.path("experiment_summary.json"),
            {
                "status": STATUS_FAILED,
                "experiment_id": self.experiment_id,
                "error": message,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )


    def plot_training_curves(self, *, show: bool = True) -> Optional[Path]:
        """Plot train loss / val loss / validation Punctuation Macro-F1."""
        if not self.history:
            return None
        try:
            import matplotlib

            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning("matplotlib unavailable — skipping training curves")
            return None

        epochs = [r.epoch for r in self.history]
        train_loss = [r.train_loss for r in self.history]
        val_loss = [r.validation.get("loss", float("nan")) for r in self.history]
        punct_f1 = [r.validation.get("punctuation_macro_f1", float("nan")) for r in self.history]
        macro_f1 = [r.validation.get("macro_f1", float("nan")) for r in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, train_loss, marker="o", label="train loss")
        axes[0].plot(epochs, val_loss, marker="s", label="validation loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title(f"{self.experiment_id} — loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, punct_f1, marker="o", color="tab:green", label="PUNCT macro-F1")
        axes[1].plot(epochs, macro_f1, marker="s", color="tab:gray", alpha=0.7, label="macro-F1 (4 cls)")
        best = max(range(len(punct_f1)), key=lambda i: (punct_f1[i] if punct_f1[i] == punct_f1[i] else -1))
        axes[1].scatter([epochs[best]], [punct_f1[best]], s=160, facecolors="none",
                        edgecolors="red", linewidths=2, label="best epoch", zorder=5)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("F1")
        axes[1].set_title(f"{self.experiment_id} — validation F1")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        out = self.path("training_curves.png")
        fig.savefig(out, dpi=140)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out
