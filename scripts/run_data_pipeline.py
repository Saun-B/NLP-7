from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import _bootstrap

import compute_statistics
import download_data
import prepare_data
import validate_data

from src.data.constants import (
    ALL_LABELED_FILE,
    CONFIG_DIR,
    OUTPUT_DATA_DIR,
    PROCESSED_FILES,
    PROJECT_ROOT,
)
from src.utils.io import write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("run_data_pipeline")

REQUIRED_OUTPUTS = [
    "data/processed/train.jsonl",
    "data/processed/validation.jsonl",
    "data/processed/test.jsonl",
    "data/processed/all_labeled.jsonl",
    "outputs/data/data_source_manifest.json",
    "outputs/data/preparation_report.json",
    "outputs/data/validation_report.json",
    "outputs/data/data_statistics.json",
    "outputs/data/data_statistics.csv",
    "outputs/data/class_weights.json",
    "outputs/data/data_hashes.json",
    "outputs/data/human_review_samples.csv",
]

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the whole data pipeline end to end.")
    parser.add_argument("--config", default=str(CONFIG_DIR / "data.yaml"))
    parser.add_argument("--force-download", action="store_true", help="re-clone the dataset")
    parser.add_argument("--skip-download", action="store_true", help="reuse the existing checkout")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="development only: parse just N raw lines per split",
    )
    parser.add_argument(
        "--strict-overlap",
        action="store_true",
        help="fail if any example text is shared across official splits",
    )
    args = parser.parse_args(argv)

    started = time.time()
    section("VIETNAMESE PUNCTUATION RESTORATION — DATA PIPELINE")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Started (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")


    if args.skip_download:
        logger.info("--skip-download: reusing the existing checkout")
    else:
        rc = download_data.main(["--force"] if args.force_download else [])
        if rc != 0:
            logger.error("Download stage failed.")
            return rc

    prep_args = ["--config", args.config]
    if args.max_lines:
        prep_args += ["--max-lines", str(args.max_lines)]
    rc = prepare_data.main(prep_args)
    if rc != 0:
        logger.error("Preparation stage failed.")
        return rc

    val_args = ["--config", args.config]
    if args.strict_overlap:
        val_args.append("--strict-overlap")
    rc = validate_data.main(val_args)
    if rc != 0:
        logger.error("Validation stage failed — refusing to continue.")
        return rc

    rc = compute_statistics.main(["--config", args.config])
    if rc != 0:
        logger.error("Statistics stage failed.")
        return rc

    section("STAGE 5/6 — ARTIFACT CHECK")
    missing = [p for p in REQUIRED_OUTPUTS if not (PROJECT_ROOT / p).exists()]
    for path in REQUIRED_OUTPUTS:
        full = PROJECT_ROOT / path
        mark = "OK  " if full.exists() else "MISS"
        size = f"{full.stat().st_size / 1024:,.0f} KB" if full.exists() else "-"
        print(f"  [{mark}] {path:<48} {size:>12}")
    if missing:
        logger.error("Missing expected artifacts: %s", missing)
        return 1

    section("STAGE 6/6 — PIPELINE RECEIPT")
    elapsed = time.time() - started
    receipt = {
        "status": "SUCCESS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": round(elapsed, 1),
        "truncated_for_development": bool(args.max_lines),
        "artifacts": REQUIRED_OUTPUTS,
        "processed_files": {
            name: Path(path).relative_to(PROJECT_ROOT).as_posix()
            for name, path in {**PROCESSED_FILES, "all_labeled": ALL_LABELED_FILE}.items()
        },
        "next_step": (
            "Data is ready. Train the four experiments manually, in order: "
            "notebooks/01_E1_BiLSTM.ipynb, notebooks/02_E2_PhoBERT_NoWeight.ipynb, "
            "notebooks/03_E3_PhoBERT_InverseWeight.ipynb, "
            "notebooks/04_E4_PhoBERT_SqrtInverse.ipynb"
        ),
    }
    write_json(OUTPUT_DATA_DIR / "pipeline_receipt.json", receipt)

    print(f"\nDATA PIPELINE COMPLETED in {elapsed / 60:.1f} min")
    print("Next: run the four training notebooks manually (E1 -> E2 -> E3 -> E4).")
    print("      No test evaluation happens during training — test.jsonl stays untouched.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
