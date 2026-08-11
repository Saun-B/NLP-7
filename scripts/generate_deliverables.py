"""Generate final project deliverables from verified artifacts.

This script does not invent evaluation numbers. It consolidates the artifacts
already produced by notebooks 05-07 into the exact file names requested by the
project handoff checklist.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap

from src.utils.io import read_json, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _checkpoint_size_mb(exp_id: str) -> float | None:
    directory = OUTPUTS_DIR / "checkpoints" / exp_id
    if not directory.exists():
        return None
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


def generate_metrics() -> Path:
    final_test = read_json(EVAL_DIR / "final_test_results.json")
    final_report = read_json(EVAL_DIR / "final_report.json")

    seconds = float(final_test.get("evaluation_seconds", 0.0))
    examples = int(final_test.get("num_examples", 0))
    words = int(final_test.get("num_evaluated_words", 0))
    metrics = final_test["metrics"]

    out = {
        "source_artifact": "outputs/evaluation/final_test_results.json",
        "split": "test",
        "winner": final_test["winner"],
        "winner_model": final_test["winner_model"],
        "selection_was_validation_only": final_test["selection_was_validation_only"],
        "test_used_for_selection": final_test["test_used_for_selection"],
        "evaluation_runs_on_test": final_test["evaluation_runs_on_test"],
        "accuracy_reference_only": metrics["accuracy"],
        "macro_f1_all_4_classes": metrics["macro_f1"],
        "punctuation_macro_f1": metrics["punctuation_macro_f1"],
        "per_punctuation_label": {
            label: metrics["per_class"][label]
            for label in ("COMMA", "PERIOD", "QUESTION")
        },
        "confusion_matrix": final_test["confusion_matrix"],
        "timing": {
            "evaluation_seconds": seconds,
            "average_inference_ms_per_example": seconds * 1000.0 / examples if examples else None,
            "examples_per_second": examples / seconds if seconds else None,
            "words_per_second": words / seconds if seconds else None,
            "note": (
                "Timing is for the locked winner E2 on the full official test split. "
                "Post-hoc E1/E3/E4 timing was not stored in the current artifacts."
            ),
        },
        "asr_evaluation": {
            "status": "not_available",
            "reason": (
                "No ASR transcript/audio/reference ASR dataset is present in this repository. "
                "The project is text-only; ASR is documented as future work."
            ),
        },
        "limitations": final_report.get("limitations", []),
    }
    return write_json(OUTPUTS_DIR / "metrics.json", out)


def generate_model_comparison() -> Path:
    rows = _read_csv_dicts(EVAL_DIR / "posthoc_test_model_comparison.csv")
    wanted = [r for r in rows if r["type"] == "trained model"]
    wanted.sort(key=lambda r: int(r["test_rank"]))

    header = [
        "model",
        "description",
        "weight_mode",
        "f1_comma",
        "f1_period",
        "f1_question",
        "punctuation_macro_f1",
        "accuracy_reference_only",
        "checkpoint_size_mb",
        "timing_note",
    ]
    out_rows = []
    for r in wanted:
        exp_id = r["system"]
        desc = {
            "E1": "BiLSTM",
            "E2": "PhoBERT",
            "E3": "PhoBERT + inverse weight",
            "E4": "PhoBERT + sqrt-inverse weight",
        }.get(exp_id, exp_id)
        out_rows.append(
            [
                exp_id,
                desc,
                r["weight_mode"],
                r["test_f1_comma"],
                r["test_f1_period"],
                r["test_f1_question"],
                r["test_punct_f1"],
                r["test_accuracy"],
                _checkpoint_size_mb(exp_id),
                "Only full-test timing for locked winner E2 is stored; use src/training/evaluate.py to benchmark each checkpoint.",
            ]
        )
    return write_csv(OUTPUTS_DIR / "model_comparison.csv", header, out_rows)


def generate_confusion_matrix_png() -> Path:
    final_test = read_json(EVAL_DIR / "final_test_results.json")
    labels = final_test["confusion_matrix"]["labels"]
    matrix = final_test["confusion_matrix"]["matrix"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion matrix - official test winner E2")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)

    max_value = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > max_value * 0.45 else "black"
            ax.text(j, i, f"{value:,}", ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    out = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _error_category(gold: str, predicted: str, meaning: str) -> str:
    if {gold, predicted} == {"PERIOD", "QUESTION"}:
        return "Nhầm dấu chấm và dấu hỏi"
    if gold == "COMMA" and predicted == "O":
        return "Bỏ sót dấu phẩy"
    if "COMMA" in (gold, predicted):
        return "Dấu phẩy / câu liệt kê"
    if gold == "PERIOD" and predicted == "O":
        return "Câu quá dài hoặc run-on"
    if predicted == "PERIOD" and gold == "O":
        return "Cắt câu quá sớm"
    if "QUESTION" in (gold, predicted):
        return "Câu hội thoại / câu hỏi"
    return meaning


def generate_error_analysis_csv(max_cases: int = 30) -> Path:
    analysis = read_json(EVAL_DIR / "final_error_analysis.json")
    header = [
        "case_id",
        "example_id",
        "word_index",
        "input_context",
        "gold_label",
        "predicted_label",
        "error_type",
        "cause",
        "suggested_fix",
    ]
    rows = []
    case_id = 1
    for pair in analysis["all_pairs"]:
        if not pair["is_error"]:
            continue
        gold = pair["gold_label"]
        predicted = pair["predicted_label"]
        category = _error_category(gold, predicted, pair["meaning"])
        for ex in pair["representative_examples"]:
            rows.append(
                [
                    case_id,
                    ex["example_id"],
                    ex["word_index"],
                    ex["context"],
                    gold,
                    predicted,
                    category,
                    pair["meaning"],
                    "Add more diverse punctuation contexts, inspect ambiguous comma labels, and consider domain-specific validation examples.",
                ]
            )
            case_id += 1
            if len(rows) >= max_cases:
                return write_csv(OUTPUTS_DIR / "error_analysis.csv", header, rows)
    return write_csv(OUTPUTS_DIR / "error_analysis.csv", header, rows)


def generate_predictions_sample() -> Path:
    analysis = read_json(EVAL_DIR / "final_error_analysis.json")
    header = [
        "example_id",
        "word_index",
        "context",
        "gold_label",
        "predicted_label",
        "is_error",
    ]
    rows = []
    for pair in analysis["all_pairs"]:
        if not pair["is_error"]:
            continue
        for ex in pair["representative_examples"]:
            rows.append(
                [
                    ex["example_id"],
                    ex["word_index"],
                    ex["context"],
                    ex["gold_label"],
                    ex["predicted_label"],
                    True,
                ]
            )
            if len(rows) >= 100:
                return write_csv(OUTPUTS_DIR / "predictions.csv", header, rows)
    return write_csv(OUTPUTS_DIR / "predictions.csv", header, rows)


def generate_asr_note() -> Path:
    path = OUTPUTS_DIR / "asr_evaluation_status.md"
    path.write_text(
        "# ASR evaluation status\n\n"
        "This repository does not currently include ASR transcripts, audio files, "
        "or a reference "
        "ASR-labeled test set. Therefore no Macro-F1 for ASR text is reported.\n\n"
        "Current implemented scope: text punctuation restoration only.\n\n"
        "Required inputs to complete ASR evaluation later:\n\n"
        "1. Clean reference transcript with punctuation labels.\n"
        "2. ASR transcript without punctuation.\n"
        "3. Audio file or ASR metadata.\n"
        "4. Word alignment between clean transcript and ASR transcript if WER is non-zero.\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    generated = [
        generate_metrics(),
        generate_model_comparison(),
        generate_confusion_matrix_png(),
        generate_error_analysis_csv(),
        generate_predictions_sample(),
        generate_asr_note(),
    ]
    for path in generated:
        print(path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
