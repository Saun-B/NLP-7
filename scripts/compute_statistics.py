"""Stage 4 — statistics, class weights, hashes, human-review samples.

    python scripts/compute_statistics.py

Writes into ``outputs/data/``:

===============================  ============================================
``data_statistics.json``          per-split counts, lengths, label ratios
``data_statistics.csv``           same, as a spreadsheet-friendly table
``class_weights.json``            inverse + sqrt-inverse weights (TRAIN only)
``data_hashes.json``              SHA-256 of every processed file
``human_review_samples.csv``      100 random examples for manual checking
===============================  ============================================
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import _bootstrap

from src.data.constants import CONFIG_DIR, LABELS, OUTPUT_DATA_DIR, PROJECT_ROOT, SEED, SPLITS
from src.data.statistics import write_statistics_artifacts
from src.utils.io import read_yaml
from src.utils.logging_utils import get_logger, section

logger = get_logger("compute_statistics")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute dataset statistics and class weights.")
    parser.add_argument("--config", default=str(CONFIG_DIR / "data.yaml"))
    parser.add_argument("--review-samples", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = read_yaml(args.config)
    stats_cfg = cfg.get("statistics", {})
    n_review = args.review_samples or int(stats_cfg.get("human_review_samples", 100))
    seed = int(stats_cfg.get("seed", SEED))

    section("STAGE 4/6 — STATISTICS, CLASS WEIGHTS, HASHES, HUMAN REVIEW")
    result = write_statistics_artifacts(
        OUTPUT_DATA_DIR, n_review_samples=n_review, seed=seed
    )

    stats = result["statistics"]
    weights = result["class_weights"]

    print("\nPer-split statistics")
    print(f"  {'split':<12}{'examples':>10}{'tokens':>14}{'mean len':>10}{'max len':>9}")
    for split in SPLITS:
        s = stats["splits"][split]
        print(
            f"  {split:<12}{s['num_examples']:>10,}{s['num_tokens']:>14,}"
            f"{s['mean_length']:>10.1f}{s['max_length']:>9}"
        )

    print("\nLabel distribution (share of tokens)")
    header = "  " + f"{'split':<12}" + "".join(f"{lab:>12}" for lab in LABELS)
    print(header)
    for split in SPLITS:
        s = stats["splits"][split]
        line = f"  {split:<12}"
        for lab in LABELS:
            line += f"{s['label_ratios'][lab] * 100:>11.3f}%"
        print(line)

    print("\nClass weights (computed from TRAIN only)")
    print(f"  {'label':<12}{'count':>12}{'inverse':>12}{'sqrt_inverse':>14}")
    for lab in LABELS:
        print(
            f"  {lab:<12}{weights['counts'][lab]:>12,}"
            f"{weights['inverse'][lab]:>12.4f}{weights['sqrt_inverse'][lab]:>14.4f}"
        )

    print("\nDataset hashes (content_sha256)")
    for name, rec in result["hashes"].items():
        if rec.get("exists"):
            print(f"  {name:<12} {rec['content_sha256']}")

    print(
        f"\nHuman review file: "
        f"{(OUTPUT_DATA_DIR / 'human_review_samples.csv').relative_to(PROJECT_ROOT).as_posix()} "
        f"({result['num_review_samples']} rows — 'is_correct' and 'review_note' "
        f"are intentionally left blank for a human to fill in)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
