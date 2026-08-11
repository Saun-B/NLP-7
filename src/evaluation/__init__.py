"""Evaluation: metrics, the shared eval loop, model selection, baselines.

``src.evaluation.selection`` is imported lazily by callers rather than
re-exported here, so that importing metrics never pulls in the model-selection
machinery (and vice versa).
"""

from src.evaluation.baselines import build_baseline, evaluate_baseline
from src.evaluation.error_analysis import analyze_errors, error_rows_for_csv
from src.evaluation.evaluator import evaluate, predict_word_labels, sample_predictions
from src.evaluation.loaders import build_eval_dataloader, load_checkpoint_encoder
from src.evaluation.metrics import (
    compute_metrics,
    confusion_matrix,
    confusion_matrix_header,
    confusion_matrix_rows,
    flatten_metrics,
    format_metrics_table,
    metrics_from_confusion_matrix,
    per_class_metrics,
    punctuation_macro_f1,
)

__all__ = [
    "evaluate",
    "predict_word_labels",
    "sample_predictions",
    "build_eval_dataloader",
    "load_checkpoint_encoder",
    "build_baseline",
    "evaluate_baseline",
    "analyze_errors",
    "error_rows_for_csv",
    "compute_metrics",
    "metrics_from_confusion_matrix",
    "confusion_matrix",
    "confusion_matrix_rows",
    "confusion_matrix_header",
    "per_class_metrics",
    "punctuation_macro_f1",
    "flatten_metrics",
    "format_metrics_table",
]
