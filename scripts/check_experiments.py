from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import _bootstrap

from src.data.constants import CHECKPOINT_DIR, EXPERIMENT_DIR, EXPERIMENT_IDS, PROJECT_ROOT
from src.training.artifacts import STATUS_COMPLETED, STATUS_NOT_RUN, ExperimentArtifactWriter
from src.training.checkpointing import CheckpointManager
from src.utils.io import read_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("check_experiments")

REQUIRED_FILES = [
    "config.json",
    "training_history.csv",
    "best_validation_metrics.json",
    "per_class_validation_metrics.csv",
    "validation_confusion_matrix.csv",
    "environment.json",
    "data_hashes.json",
    "experiment_summary.json",
]

def inspect(experiment_id: str) -> Dict[str, Any]:
    exp_dir = EXPERIMENT_DIR / experiment_id
    ckpt_dir = CHECKPOINT_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = ExperimentArtifactWriter(experiment_id)
    writer.init_not_run()

    summary_path = exp_dir / "experiment_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"status": STATUS_NOT_RUN}

    missing = [f for f in REQUIRED_FILES if not (exp_dir / f).exists()]
    has_ckpt = CheckpointManager.has_checkpoint(ckpt_dir)

    return {
        "experiment_id": experiment_id,
        "status": summary.get("status", STATUS_NOT_RUN),
        "has_checkpoint": has_ckpt,
        "missing_artifacts": missing,
        "best_epoch": summary.get("best_epoch"),
        "best_validation_punctuation_macro_f1": summary.get(
            "best_validation_punctuation_macro_f1"
        ),
        "ready": summary.get("status") == STATUS_COMPLETED and has_ckpt and not missing,
    }

def main(argv: Optional[List[str]] = None) -> int:
    section("EXPERIMENT STATUS — E1 / E2 / E3 / E4")
    rows = [inspect(e) for e in EXPERIMENT_IDS]

    print(f"  {'exp':<5}{'status':<16}{'ckpt':<7}{'best epoch':>11}{'val PUNCT-F1':>15}   missing")
    print("  " + "-" * 78)
    for r in rows:
        score = r["best_validation_punctuation_macro_f1"]
        score_txt = f"{score:.6f}" if isinstance(score, (int, float)) else "-"
        epoch_txt = str(r["best_epoch"]) if r["best_epoch"] is not None else "-"
        missing = ",".join(r["missing_artifacts"]) if r["missing_artifacts"] else "-"
        print(
            f"  {r['experiment_id']:<5}{r['status']:<16}"
            f"{'yes' if r['has_checkpoint'] else 'no':<7}{epoch_txt:>11}{score_txt:>15}   {missing[:40]}"
        )

    done = [r["experiment_id"] for r in rows if r["ready"]]
    todo = [r["experiment_id"] for r in rows if not r["ready"]]

    print(f"\n  Trained and complete : {done or 'none yet'}")
    print(f"  Still to run         : {todo or 'none — all four are done'}")

    if todo:
        print("\n  Run the remaining notebooks manually, in order:")
        names = {
            "E1": "notebooks/01_E1_BiLSTM.ipynb",
            "E2": "notebooks/02_E2_PhoBERT_NoWeight.ipynb",
            "E3": "notebooks/03_E3_PhoBERT_InverseWeight.ipynb",
            "E4": "notebooks/04_E4_PhoBERT_SqrtInverse.ipynb",
        }
        for e in todo:
            print(f"    - {names[e]}")
    else:
        print(
            "\n  All four experiments are complete. Model selection, the official test "
            "\n  evaluation, inference and the UI belong to Phase 2 — none of them are "
            "\n  performed by this script."
        )

    print(
        "\n  Note: this script reports per-experiment readiness only. It does not "
        "\n  compare experiments and never reads data/processed/test.jsonl."
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
