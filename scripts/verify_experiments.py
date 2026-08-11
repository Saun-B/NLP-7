from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap

import numpy as np

from src.data.constants import (
    CHECKPOINT_DIR,
    EXPERIMENT_DIR,
    EXPERIMENT_IDS,
    LABEL2ID,
    LABELS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    PUNCTUATION_LABELS,
)
from src.evaluation.metrics import metrics_from_confusion_matrix
from src.training.artifacts import STATUS_COMPLETED
from src.training.checkpointing import CheckpointManager
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("verify_experiments")

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

REQUIRED_ARTIFACTS = [
    "experiment_summary.json",
    "best_validation_metrics.json",
    "training_history.csv",
    "config.json",
    "environment.json",
    "data_hashes.json",
    "per_class_validation_metrics.csv",
    "validation_confusion_matrix.csv",
]

TOLERANCE = 1e-6

class Verifier:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def check(self, condition: bool, msg: str) -> bool:
        if not condition:
            self.error(msg)
        return condition

def _read_history(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def verify_experiment(v: Verifier, experiment_id: str) -> Dict[str, Any]:
    exp_dir = EXPERIMENT_DIR / experiment_id
    ckpt_dir = CHECKPOINT_DIR / experiment_id
    facts: Dict[str, Any] = {"experiment_id": experiment_id, "ok": False}

    missing = [a for a in REQUIRED_ARTIFACTS if not (exp_dir / a).exists()]
    if missing:
        v.error(f"{experiment_id}: missing artifact(s) {missing}")
        facts["missing_artifacts"] = missing
        return facts
    facts["missing_artifacts"] = []

    summary = read_json(exp_dir / "experiment_summary.json")
    best = read_json(exp_dir / "best_validation_metrics.json")
    hashes = read_json(exp_dir / "data_hashes.json")
    config = read_json(exp_dir / "config.json")

    v.check(
        summary.get("status") == STATUS_COMPLETED,
        f"{experiment_id}: status is {summary.get('status')!r}, expected {STATUS_COMPLETED}",
    )

    has_ckpt = CheckpointManager.has_checkpoint(ckpt_dir)
    v.check(has_ckpt, f"{experiment_id}: no usable checkpoint in {ckpt_dir}")
    meta: Dict[str, Any] = {}
    if has_ckpt:
        meta = CheckpointManager.read_metadata(ckpt_dir)
        weight_files = [
            p.name
            for p in ckpt_dir.iterdir()
            if p.suffix in (".pt", ".safetensors", ".bin")
        ]
        facts["checkpoint_files"] = sorted(weight_files)
        facts["checkpoint_bytes"] = sum(
            p.stat().st_size for p in ckpt_dir.iterdir() if p.is_file()
        )

    history = _read_history(exp_dir / "training_history.csv")
    v.check(len(history) > 0, f"{experiment_id}: training_history.csv is empty")
    facts["num_epochs"] = len(history)

    if history:
        scores = [float(r["val_punctuation_macro_f1"]) for r in history]
        epochs = [int(r["epoch"]) for r in history]
        best_row = int(np.argmax(scores))
        history_best_epoch = epochs[best_row]
        history_best_score = scores[best_row]
        facts["history_best_epoch"] = history_best_epoch
        facts["history_best_score"] = history_best_score

        v.check(
            summary.get("best_epoch") == history_best_epoch,
            f"{experiment_id}: summary best_epoch={summary.get('best_epoch')} but the "
            f"best row in training_history.csv is epoch {history_best_epoch}",
        )
        v.check(
            abs(float(summary["best_validation_punctuation_macro_f1"]) - history_best_score)
            < TOLERANCE,
            f"{experiment_id}: summary score {summary['best_validation_punctuation_macro_f1']} "
            f"disagrees with history {history_best_score}",
        )
        marked_best = [int(r["epoch"]) for r in history if str(r.get("is_best")) == "1"]
        if marked_best and marked_best[-1] != history_best_epoch:
            v.warn(
                f"{experiment_id}: last is_best row is epoch {marked_best[-1]} but the "
                f"highest score is at epoch {history_best_epoch}"
            )

    metrics = best["metrics"]
    per_class = metrics["per_class"]
    recomputed = float(
        np.mean([per_class[c]["f1"] for c in PUNCTUATION_LABELS])
    )
    v.check(
        abs(recomputed - float(metrics["punctuation_macro_f1"])) < TOLERANCE,
        f"{experiment_id}: punctuation_macro_f1={metrics['punctuation_macro_f1']} but the "
        f"mean of F1(COMMA/PERIOD/QUESTION) is {recomputed}",
    )
    v.check(
        abs(float(summary["best_validation_punctuation_macro_f1"]) - recomputed) < TOLERANCE,
        f"{experiment_id}: summary score disagrees with best_validation_metrics.json",
    )

    cm_path = exp_dir / "validation_confusion_matrix.csv"
    with open(cm_path, "r", encoding="utf-8-sig", newline="") as f:
        cm_rows = list(csv.reader(f))
    cm = np.array([[int(x) for x in row[1:]] for row in cm_rows[1:]], dtype=np.int64)
    facts["confusion_matrix_total"] = int(cm.sum())

    rebuilt = metrics_from_confusion_matrix(cm)
    for name in ("accuracy", "macro_f1", "punctuation_macro_f1"):
        v.check(
            abs(rebuilt[name] - float(metrics[name])) < 1e-5,
            f"{experiment_id}: {name} recomputed from the confusion matrix "
            f"({rebuilt[name]:.8f}) disagrees with the stored value ({metrics[name]:.8f})",
        )
    v.check(
        int(cm.sum()) == int(metrics["num_evaluated_tokens"]),
        f"{experiment_id}: confusion matrix sums to {int(cm.sum())} but "
        f"num_evaluated_tokens={metrics['num_evaluated_tokens']}",
    )

    if summary.get("test_split_used") is True:
        v.error(f"{experiment_id}: summary claims the test split was used during training")
    v.check(
        best.get("evaluated_on", "").startswith("validation.jsonl"),
        f"{experiment_id}: best_validation_metrics.json was not evaluated on validation.jsonl",
    )

    facts.update(
        {
            "status": summary.get("status"),
            "model": summary.get("model"),
            "model_type": meta.get("model_type"),
            "model_revision": meta.get("model_revision"),
            "weight_mode": summary.get("weight_mode"),
            "best_epoch": summary.get("best_epoch"),
            "punctuation_macro_f1": float(metrics["punctuation_macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "own_loss": metrics.get("loss"),
            "num_evaluated_tokens": int(metrics["num_evaluated_tokens"]),
            "seed": meta.get("seed"),
            "label2id": meta.get("label2id"),
            "train_hash": hashes["train"]["content_sha256"],
            "validation_hash": hashes["validation"]["content_sha256"],
            "test_hash": hashes["test"]["content_sha256"],
            "checkpoint_dir": ckpt_dir.relative_to(PROJECT_ROOT).as_posix(),
            "experiment_dir": exp_dir.relative_to(PROJECT_ROOT).as_posix(),
            "total_training_seconds": summary.get("total_training_seconds"),
            "config_epochs": config.get("config", {}).get("training", {}).get("epochs"),
            "has_checkpoint": has_ckpt,
        }
    )
    facts["ok"] = True
    return facts

def verify_across(v: Verifier, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [f for f in facts if f.get("ok")]
    out: Dict[str, Any] = {}
    if len(usable) < 2:
        v.error("Fewer than two usable experiments — cannot compare.")
        return out

    for key, label in (
        ("train_hash", "train data hash"),
        ("validation_hash", "validation data hash"),
        ("test_hash", "test data hash"),
        ("seed", "seed"),
        ("num_evaluated_tokens", "validation token count"),
    ):
        values = {f["experiment_id"]: f.get(key) for f in usable}
        unique = set(values.values())
        out[key] = {"consistent": len(unique) == 1, "values": values}
        if len(unique) != 1:
            v.error(
                f"Experiments disagree on {label}: {values}. "
                "They were NOT run under the same protocol — do not compare them as one "
                "experiment series until this is resolved."
            )

    label_maps = {
        f["experiment_id"]: tuple(sorted((f.get("label2id") or {}).items())) for f in usable
    }
    consistent_labels = len(set(label_maps.values())) == 1
    out["label2id"] = {"consistent": consistent_labels, "values": {k: dict(v_) for k, v_ in label_maps.items()}}
    if not consistent_labels:
        v.error(f"Experiments disagree on the label mapping: {label_maps}")
    else:
        expected = tuple(sorted(LABEL2ID.items()))
        if next(iter(label_maps.values())) != expected:
            v.error(f"Checkpoint label mapping differs from the project mapping {LABEL2ID}")


    revisions = {
        f["experiment_id"]: f.get("model_revision")
        for f in usable
        if f.get("model_type") == "phobert"
    }
    unique_rev = set(revisions.values())
    out["phobert_revision"] = {"consistent": len(unique_rev) <= 1, "values": revisions}
    if len(unique_rev) > 1:
        v.error(f"PhoBERT experiments used different revisions: {revisions}")

    out["test_used_for_training"] = {
        "any": False,
        "note": "every experiment recorded test_split_used = false",
    }
    return out

def main(argv: Optional[List[str]] = None) -> int:
    section("PHASE 2 · STEP 1 — VERIFY THE FOUR TRAINING RUNS")
    v = Verifier()

    facts = [verify_experiment(v, e) for e in EXPERIMENT_IDS]
    across = verify_across(v, facts)

    print(
        f"\n  {'exp':<5}{'status':<11}{'model':<26}{'weight':<14}"
        f"{'ep':>4}{'val PUNCT-F1':>14}{'ckpt':>7}"
    )
    print("  " + "-" * 82)
    for f in facts:
        if not f.get("ok"):
            print(f"  {f['experiment_id']:<5}{'BROKEN':<11}{str(f.get('missing_artifacts'))[:60]}")
            continue
        print(
            f"  {f['experiment_id']:<5}{f['status']:<11}{str(f['model'])[:25]:<26}"
            f"{str(f['weight_mode']):<14}{f['best_epoch']:>4}"
            f"{f['punctuation_macro_f1']:>14.6f}{'yes' if f['has_checkpoint'] else 'NO':>7}"
        )

    print("\n  Cross-experiment consistency")
    for key in ("train_hash", "validation_hash", "test_hash", "seed", "num_evaluated_tokens", "label2id"):
        rec = across.get(key)
        if rec:
            mark = "OK  " if rec["consistent"] else "FAIL"
            sample = next(iter(rec["values"].values()))
            shown = str(sample)[:24] + ("…" if len(str(sample)) > 24 else "")
            print(f"    [{mark}] {key:<22} {shown}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": not v.errors,
        "num_errors": len(v.errors),
        "num_warnings": len(v.warnings),
        "errors": v.errors,
        "warnings": v.warnings,
        "experiments": facts,
        "cross_experiment": across,
        "note": (
            "The 'own_loss' field is each experiment's validation loss under ITS OWN "
            "loss function (E3 inverse-weighted, E4 sqrt-inverse-weighted). Those values "
            "are not comparable across experiments; a comparable unweighted validation "
            "loss is computed separately by scripts/compute_unweighted_validation_loss.py."
        ),
    }
    out = write_json(EVALUATION_DIR / "training_verification.json", report)

    print(f"\n  Report: {out.relative_to(PROJECT_ROOT).as_posix()}")
    if v.warnings:
        print(f"  Warnings ({len(v.warnings)}):")
        for w in v.warnings:
            print(f"    - {w}")
    if v.errors:
        print(f"\n  VERIFICATION FAILED with {len(v.errors)} error(s):")
        for e in v.errors:
            print(f"    - {e}")
        return 1

    print("\n  VERIFICATION PASSED — all four runs are complete, self-consistent,")
    print("  and were trained on identical data with an identical label mapping.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
