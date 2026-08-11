"""Optimizer construction.

Both families use ``torch.optim.AdamW`` but with different conventions:

* **E1 (BiLSTM)** — one parameter group, ``lr=1e-3``, ``weight_decay=1e-4``.
* **E2–E4 (PhoBERT)** — ``lr=3e-5``, ``weight_decay=1e-2``, with the standard
  transformer split: biases and LayerNorm parameters get **no** weight decay.
  Decaying them is known to hurt fine-tuning, and every reference PhoBERT
  recipe excludes them, so the three PhoBERT runs stay comparable to published
  numbers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["build_optimizer", "build_param_groups", "NO_DECAY_KEYWORDS"]


NO_DECAY_KEYWORDS = ("bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight")


def build_param_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    no_decay_keywords: Iterable[str] = NO_DECAY_KEYWORDS,
    split_no_decay: bool = True,
) -> List[Dict[str, Any]]:
    """Split parameters into decay / no-decay groups."""
    if not split_no_decay:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": weight_decay,
            }
        ]

    keywords = tuple(no_decay_keywords)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in keywords):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build AdamW from the ``optimizer`` section of an experiment config."""
    opt_cfg = config.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer {name!r}; this project uses AdamW.")

    lr = float(opt_cfg.get("learning_rate", 3e-5))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
    eps = float(opt_cfg.get("eps", 1e-8))
    split_no_decay = bool(opt_cfg.get("no_decay_on_bias_and_layernorm", True))

    groups = build_param_groups(
        model, weight_decay=weight_decay, split_no_decay=split_no_decay
    )
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)

    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_no_decay = sum(p.numel() for p in groups[1]["params"]) if len(groups) > 1 else 0
    logger.info(
        "AdamW(lr=%g, weight_decay=%g, betas=%s, eps=%g) — %s decayed / %s not decayed params",
        lr,
        weight_decay,
        betas,
        eps,
        f"{n_decay:,}",
        f"{n_no_decay:,}",
    )
    return optimizer
