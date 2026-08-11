"""Loss factory — the only thing that differs between E2, E3 and E4.

============  ==================================================  ============
experiment    weight mode                                          class weight
============  ==================================================  ============
E1            ``none``                                             1, 1, 1, 1
E2            ``none``                                             1, 1, 1, 1
E3            ``inverse``   ``total / (4 * count_c)``               ~0.27 … ~45
E4            ``sqrt_inverse``  ``sqrt(inverse_c)``                 ~0.52 … ~6.7
============  ==================================================  ============

All weights come from :file:`outputs/data/class_weights.json`, which is
computed from the **train split only** (see
:func:`src.data.statistics.compute_class_weights`). Reading weights from
validation or test would leak evaluation information into training, so
:func:`load_class_weights` refuses any other source split.

Every loss uses ``ignore_index=-100``, which is what makes padding,
non-final subwords and special tokens invisible to the optimiser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.data.constants import IGNORE_INDEX, LABELS, OUTPUT_DATA_DIR
from src.utils.io import read_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

WEIGHT_MODES = ("none", "inverse", "sqrt_inverse")

__all__ = [
    "WEIGHT_MODES",
    "load_class_weights",
    "get_class_weight_vector",
    "build_loss",
]


def load_class_weights(
    path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Load ``outputs/data/class_weights.json`` and assert its provenance."""
    path = Path(path) if path is not None else OUTPUT_DATA_DIR / "class_weights.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/run_data_pipeline.py` first."
        )
    blob = read_json(path)
    if blob.get("source_split") != "train":
        raise ValueError(
            f"{path}: class weights must be derived from the train split, "
            f"got source_split={blob.get('source_split')!r}. Refusing to train on "
            "weights that could leak validation/test statistics."
        )
    if blob.get("label_order") != LABELS:
        raise ValueError(
            f"{path}: label_order {blob.get('label_order')} does not match "
            f"the project label order {LABELS}"
        )
    return blob


def get_class_weight_vector(
    weight_mode: str, *, class_weights: Optional[Dict[str, Any]] = None, path: Optional[PathLike] = None
) -> Optional[List[float]]:
    """Return the weight vector for ``weight_mode``, ordered by label id.

    ``none`` returns ``None`` — an unweighted ``CrossEntropyLoss``, not a
    vector of ones, so the E2 loss is exactly PyTorch's default.
    """
    mode = weight_mode.lower()
    if mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown weight_mode {weight_mode!r}; expected {WEIGHT_MODES}")
    if mode == "none":
        return None

    blob = class_weights or load_class_weights(path)
    key = "inverse_vector" if mode == "inverse" else "sqrt_inverse_vector"
    vector = blob.get(key)
    if vector is None:
        mapping = blob["inverse" if mode == "inverse" else "sqrt_inverse"]
        vector = [float(mapping[lab]) for lab in LABELS]
    if len(vector) != len(LABELS):
        raise ValueError(f"weight vector has {len(vector)} entries, expected {len(LABELS)}")
    return [float(x) for x in vector]


def build_loss(
    weight_mode: str = "none",
    *,
    class_weights: Optional[Dict[str, Any]] = None,
    weights_path: Optional[PathLike] = None,
    device: Optional[torch.device] = None,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """Build the ``CrossEntropyLoss`` for one experiment."""
    vector = get_class_weight_vector(
        weight_mode, class_weights=class_weights, path=weights_path
    )
    weight_tensor: Optional[torch.Tensor] = None
    if vector is not None:
        weight_tensor = torch.tensor(vector, dtype=torch.float32)
        if device is not None:
            weight_tensor = weight_tensor.to(device)

    loss = nn.CrossEntropyLoss(
        weight=weight_tensor,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    logger.info(
        "Loss: CrossEntropyLoss(weight_mode=%s, weights=%s, ignore_index=%d)",
        weight_mode,
        None if vector is None else [round(v, 4) for v in vector],
        ignore_index,
    )
    return loss


def describe_loss(weight_mode: str, vector: Optional[Sequence[float]]) -> Dict[str, Any]:
    """Serialisable description written into each experiment's ``config.json``."""
    return {
        "type": "CrossEntropyLoss",
        "weight_mode": weight_mode,
        "ignore_index": IGNORE_INDEX,
        "class_weights": (
            None if vector is None else {lab: float(w) for lab, w in zip(LABELS, vector)}
        ),
        "weights_source": "outputs/data/class_weights.json (train split only)",
    }
