from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.constants import MONITOR_METRIC
from src.evaluation.evaluator import evaluate
from src.evaluation.metrics import format_metrics_table
from src.training.artifacts import EpochRecord, ExperimentArtifactWriter
from src.training.checkpointing import CheckpointManager
from src.training.scheduler import current_lr
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["TrainerBase", "TrainingResult"]

@dataclass
class TrainingResult:
    experiment_id: str
    best_epoch: int
    best_score: float
    best_metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    checkpoint_dir: str
    experiment_dir: str
    total_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "checkpoint_dir": self.checkpoint_dir,
            "experiment_dir": self.experiment_dir,
            "total_seconds": round(self.total_seconds, 2),
            "num_epochs": len(self.history),
        }

class TrainerBase:
    def __init__(
        self,
        *,
        experiment_id: str,
        model: torch.nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scheduler_step_mode: str = "none",
        device: Optional[torch.device] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        artifacts: Optional[ExperimentArtifactWriter] = None,
        use_amp: bool = False,
    ):
        training_cfg = config.get("training", {})

        self.experiment_id = experiment_id
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_mode = scheduler_step_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints = checkpoint_manager
        self.artifacts = artifacts or ExperimentArtifactWriter(experiment_id)

        self.epochs = int(training_cfg.get("epochs", 1))
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        self.monitor_metric = str(training_cfg.get("monitor_metric", MONITOR_METRIC))
        self.log_every = int(training_cfg.get("log_every_n_steps", 200))
        self.early_stopping_patience = training_cfg.get("early_stopping_patience")
        self.gradient_accumulation_steps = max(
            1, int(training_cfg.get("gradient_accumulation_steps", 1))
        )

        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.model.to(self.device)
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        raise NotImplementedError

    def fit(self) -> TrainingResult:
        logger.info(
            "=== %s: training for %d epoch(s) on %s (AMP=%s) ===",
            self.experiment_id,
            self.epochs,
            self.device,
            self.use_amp,
        )
        self.artifacts.mark_running()
        started = time.time()
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            epoch_started = time.time()
            lr_before = current_lr(self.optimizer)

            train_loss = self.train_epoch(epoch)

            val_metrics = evaluate(
                self.model,
                self.validation_loader,
                device=self.device,
                loss_fn=self.loss_fn,
                use_amp=self.use_amp,
                desc=f"{self.experiment_id} epoch {epoch} validation",
            )

            improved = False
            if self.checkpoints is not None:
                improved = self.checkpoints.maybe_save(
                    self.model, epoch=epoch, metrics=val_metrics
                )

            elapsed = time.time() - epoch_started
            self.artifacts.add_epoch(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    learning_rate=lr_before,
                    epoch_seconds=elapsed,
                    validation=val_metrics,
                    is_best=improved,
                )
            )

            print(
                f"\n[{self.experiment_id}] epoch {epoch}/{self.epochs} "
                f"({elapsed / 60:.1f} min) — train_loss={train_loss:.4f} "
                f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
                f"val_PUNCT_F1={val_metrics['punctuation_macro_f1']:.4f}"
                f"{'  <-- new best' if improved else ''}"
            )
            print(format_metrics_table(val_metrics))


            if self.scheduler is not None and self.scheduler_step_mode == "epoch":
                self.scheduler.step(val_metrics[self.monitor_metric])

            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    self.early_stopping_patience
                    and epochs_without_improvement >= int(self.early_stopping_patience)
                ):
                    logger.info(
                        "Early stopping: no improvement for %d epoch(s)",
                        epochs_without_improvement,
                    )
                    break

        total = time.time() - started
        best_epoch = self.checkpoints.best_epoch if self.checkpoints else -1
        best_score = self.checkpoints.best_score if self.checkpoints else float("nan")
        best_metrics = self.checkpoints.best_metrics if self.checkpoints else {}

        return TrainingResult(
            experiment_id=self.experiment_id,
            best_epoch=best_epoch,
            best_score=best_score,
            best_metrics=best_metrics,
            history=[r.to_row() for r in self.artifacts.history],
            checkpoint_dir=(
                Path(self.checkpoints.checkpoint_dir).as_posix() if self.checkpoints else ""
            ),
            experiment_dir=self.artifacts.dir.as_posix(),
            total_seconds=total,
        )

    def _clip_gradients(self) -> None:
        if self.max_grad_norm and self.max_grad_norm > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _optimizer_step(self) -> None:
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None and self.scheduler_step_mode == "step":
            self.scheduler.step()
        self.global_step += 1

    def _progress(self, iterable, desc: str):
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, desc=desc, leave=False)
        except Exception:
            return iterable
