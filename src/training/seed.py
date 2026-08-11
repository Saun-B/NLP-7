"""Seeding for reproducible runs (SEED = 42 everywhere).

Covers ``random``, ``numpy``, ``torch`` (CPU + all CUDA devices), the
``PYTHONHASHSEED`` env var, and DataLoader worker seeding.

Caveat worth stating in the report: even with ``deterministic=True``, cuDNN
kernels and non-deterministic reductions can make two GPU runs differ in the
last decimals. ``deterministic`` mode is therefore *available* but off by
default, because it slows PhoBERT training noticeably.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from src.data.constants import SEED
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["set_seed", "seed_worker", "make_generator"]


def set_seed(seed: int = SEED, *, deterministic: bool = False) -> int:
    """Seed every RNG this project touches. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` — keeps shuffling reproducible."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    """Generator to hand to ``DataLoader(generator=...)`` for stable shuffles."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
