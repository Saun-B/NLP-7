"""Metrics for word-level punctuation restoration.

Headline metric — **Punctuation Macro-F1**
------------------------------------------
``O`` covers ~91% of tokens. A model that predicts ``O`` everywhere already
scores 91% accuracy and a 4-class macro-F1 near 0.24 while being completely
useless. So the number that selects checkpoints is the unweighted mean of the
F1 of the three punctuation classes only::

    punctuation_macro_f1 = (F1_COMMA + F1_PERIOD + F1_QUESTION) / 3

Accuracy, 4-class macro-F1 and per-class precision/recall/F1 are still
reported — they are informative — but they never decide anything.

Only positions where ``gold != IGNORE_INDEX`` are scored, so padding,
non-final subwords and special tokens are excluded automatically.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.data.constants import (
    ID2LABEL,
    IGNORE_INDEX,
    LABELS,
    NUM_LABELS,
    PUNCTUATION_LABELS,
)

__all__ = [
    "confusion_matrix",
    "per_class_metrics",
    "compute_metrics",
    "metrics_from_confusion_matrix",
    "punctuation_macro_f1",
    "format_metrics_table",
    "flatten_metrics",
    "confusion_matrix_rows",
    "confusion_matrix_header",
]


def _as_array(values: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64) if not isinstance(values, np.ndarray) else values
    return arr.reshape(-1)


def filter_ignored(
    y_true: Iterable[int], y_pred: Iterable[int], *, ignore_index: int = IGNORE_INDEX
) -> tuple[np.ndarray, np.ndarray]:
    """Drop positions whose gold label is ``ignore_index``."""
    t = _as_array(y_true)
    p = _as_array(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true{t.shape} vs y_pred{p.shape}")
    mask = t != ignore_index
    return t[mask], p[mask]


def confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    num_labels: int = NUM_LABELS,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """``[gold, predicted]`` counts, shape ``(num_labels, num_labels)``."""
    t, p = filter_ignored(y_true, y_pred, ignore_index=ignore_index)
    cm = np.zeros((num_labels, num_labels), dtype=np.int64)
    if t.size == 0:
        return cm
    if t.min() < 0 or t.max() >= num_labels:
        raise ValueError(f"gold label out of range [0,{num_labels - 1}]")
    if p.min() < 0 or p.max() >= num_labels:
        raise ValueError(f"predicted label out of range [0,{num_labels - 1}]")
    np.add.at(cm, (t, p), 1)
    return cm


def per_class_metrics(cm: np.ndarray, *, labels: Sequence[str] = LABELS) -> Dict[str, Dict[str, float]]:
    """Precision / recall / F1 / support per class, derived from a confusion matrix."""
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(labels):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = float(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def punctuation_macro_f1(
    per_class: Dict[str, Dict[str, float]],
    *,
    punctuation_labels: Sequence[str] = PUNCTUATION_LABELS,
) -> float:
    """Mean F1 over COMMA / PERIOD / QUESTION (``O`` excluded)."""
    scores = [per_class[name]["f1"] for name in punctuation_labels]
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Sequence[str] = LABELS,
    punctuation_labels: Sequence[str] = PUNCTUATION_LABELS,
    ignore_index: int = IGNORE_INDEX,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Full metric bundle for one evaluation pass."""
    cm = confusion_matrix(
        y_true, y_pred, num_labels=len(labels), ignore_index=ignore_index
    )
    return metrics_from_confusion_matrix(
        cm, labels=labels, punctuation_labels=punctuation_labels, loss=loss
    )


def metrics_from_confusion_matrix(
    cm: np.ndarray,
    *,
    labels: Sequence[str] = LABELS,
    punctuation_labels: Sequence[str] = PUNCTUATION_LABELS,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Metric bundle from an already-accumulated confusion matrix.

    The evaluator accumulates counts batch by batch instead of keeping every
    prediction in memory (validation is ~2M tokens), then calls this.
    """
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    correct = int(np.trace(cm))

    per_class = per_class_metrics(cm, labels=labels)
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in labels]))
    punct_f1 = punctuation_macro_f1(per_class, punctuation_labels=punctuation_labels)



    tp = sum(per_class[n]["tp"] for n in punctuation_labels)
    fp = sum(per_class[n]["fp"] for n in punctuation_labels)
    fn = sum(per_class[n]["fn"] for n in punctuation_labels)
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )

    weighted_f1 = (
        float(
            sum(per_class[n]["f1"] * per_class[n]["support"] for n in labels) / total
        )
        if total
        else 0.0
    )

    metrics: Dict[str, Any] = {
        "num_evaluated_tokens": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "punctuation_macro_f1": punct_f1,
        "punctuation_micro_f1": micro_f1,
        "punctuation_micro_precision": micro_p,
        "punctuation_micro_recall": micro_r,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def flatten_metrics(metrics: Dict[str, Any], *, prefix: str = "") -> Dict[str, float]:
    """Flatten a metric bundle into scalar columns for ``training_history.csv``."""
    flat: Dict[str, float] = {}
    for key in (
        "loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "punctuation_macro_f1",
        "punctuation_micro_f1",
        "punctuation_micro_precision",
        "punctuation_micro_recall",
    ):
        if key in metrics:
            flat[f"{prefix}{key}"] = float(metrics[key])
    for name, values in metrics.get("per_class", {}).items():
        flat[f"{prefix}f1_{name}"] = float(values["f1"])
        flat[f"{prefix}precision_{name}"] = float(values["precision"])
        flat[f"{prefix}recall_{name}"] = float(values["recall"])
    return flat


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Readable per-class table + headline numbers (printed in notebooks)."""
    lines = [
        f"{'label':<10}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>12}",
        "-" * 53,
    ]
    for name in metrics.get("labels", LABELS):
        m = metrics["per_class"][name]
        lines.append(
            f"{name:<10}{m['precision']:>11.4f}{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}{int(m['support']):>12,}"
        )
    lines.append("-" * 53)
    lines.append(f"{'accuracy':<10}{metrics['accuracy']:>11.4f}")
    lines.append(f"{'macro F1':<10}{metrics['macro_f1']:>11.4f}   (all 4 classes)")
    lines.append(
        f"{'PUNCT-F1':<10}{metrics['punctuation_macro_f1']:>11.4f}   "
        f"<-- model selection metric (COMMA/PERIOD/QUESTION)"
    )
    return "\n".join(lines)


def confusion_matrix_rows(cm: Sequence[Sequence[int]], labels: Sequence[str] = LABELS) -> List[List[Any]]:
    """Confusion matrix as CSV rows: ``gold_<L>`` rows × ``pred_<L>`` columns."""
    rows: List[List[Any]] = []
    for i, gold in enumerate(labels):
        rows.append([f"gold_{gold}"] + [int(cm[i][j]) for j in range(len(labels))])
    return rows


def confusion_matrix_header(labels: Sequence[str] = LABELS) -> List[str]:
    return ["gold\\pred"] + [f"pred_{l}" for l in labels]


def id_sequences_to_labels(ids: Iterable[int]) -> List[str]:
    return [ID2LABEL[int(i)] for i in ids]
