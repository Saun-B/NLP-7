"""Stage 3 — validate ``data/processed/*.jsonl``.

    python scripts/validate_data.py [--strict-overlap]

Re-reads the written files and checks every invariant listed in
:mod:`src.data.validation`. Writes ``outputs/data/validation_report.json`` and
exits non-zero if any **ERROR** was found. Warnings (notably cross-split text
overlap) are reported but do not fail the run unless ``--strict-overlap`` is
passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import _bootstrap

from src.data.constants import (
    ALL_LABELED_FILE,
    CONFIG_DIR,
    MAX_WORDS_PER_EXAMPLE,
    OUTPUT_DATA_DIR,
    PROJECT_ROOT,
)
from src.data.validation import validate_processed_data
from src.utils.io import read_yaml, write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("validate_data")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the processed dataset.")
    parser.add_argument("--config", default=str(CONFIG_DIR / "data.yaml"))
    parser.add_argument(
        "--strict-overlap",
        action="store_true",
        help="treat cross-split text overlap as a failure instead of a warning",
    )
    args = parser.parse_args(argv)

    cfg = read_yaml(args.config)
    val_cfg = cfg.get("validation", {})

    section("STAGE 3/6 — VALIDATE PROCESSED DATA")
    report = validate_processed_data(
        all_labeled_file=ALL_LABELED_FILE,
        max_words=int(val_cfg.get("max_words", MAX_WORDS_PER_EXAMPLE)),
        max_schema_errors=int(val_cfg.get("max_schema_errors_reported", 20)),
    )

    blob = report.to_dict()
    overlap_warnings = [i for i in report.warnings if i.code == "TEXT_OVERLAP"]
    strict = args.strict_overlap or bool(val_cfg.get("fail_on_cross_split_text_overlap", False))
    blob["strict_overlap_mode"] = strict

    out = write_json(OUTPUT_DATA_DIR / "validation_report.json", blob)
    print("\n" + report.summary_text())
    print(f"\nFull report: {out.relative_to(PROJECT_ROOT).as_posix()}")

    if not report.passed:
        logger.error("Validation FAILED with %d error(s).", len(report.errors))
        return 1
    if strict and overlap_warnings:
        logger.error("--strict-overlap: cross-split text overlap present, failing.")
        return 1

    logger.info("Validation PASSED (%d warning(s)).", len(report.warnings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
