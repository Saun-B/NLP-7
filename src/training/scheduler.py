from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["build_scheduler", "get_linear_schedule_with_warmup", "STEP_MODES"]

STEP_MODES = ("step", "epoch")

def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        remaining = num_training_steps - current_step
        return max(0.0, float(remaining) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def build_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    *,
    num_training_steps: Optional[int] = None,
) -> Tuple[Optional[object], str]:
    sched_cfg = config.get("scheduler", {})
    name = str(sched_cfg.get("name", "none")).lower()

    if name in ("none", "constant"):
        logger.info("Scheduler: constant learning rate")
        return None, "none"

    if name == "linear_warmup":
        if num_training_steps is None:
            raise ValueError("linear_warmup scheduler requires num_training_steps")
        warmup_ratio = float(sched_cfg.get("warmup_ratio", 0.1))
        num_warmup = int(sched_cfg.get("warmup_steps", math.floor(warmup_ratio * num_training_steps)))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training_steps
        )
        logger.info(
            "Scheduler: linear warmup+decay (%d warmup / %d total optimizer steps, ratio=%.2f)",
            num_warmup,
            num_training_steps,
            warmup_ratio,
        )
        return scheduler, "step"

    if name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(sched_cfg.get("mode", "max")),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 1)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
        logger.info(
            "Scheduler: ReduceLROnPlateau(mode=%s, factor=%s, patience=%s) on validation %s",
            sched_cfg.get("mode", "max"),
            sched_cfg.get("factor", 0.5),
            sched_cfg.get("patience", 1),
            config.get("training", {}).get("monitor_metric", "punctuation_macro_f1"),
        )
        return scheduler, "epoch"

    raise ValueError(f"Unknown scheduler {name!r}")

def current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
