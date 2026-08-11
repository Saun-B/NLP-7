from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap

from src.data.chunking import assert_chunking_invariants, chunk_sequence
from src.data.constants import (
    ALL_LABELED_FILE,
    CONFIG_DIR,
    MAX_WORDS_PER_EXAMPLE,
    OUTPUT_DATA_DIR,
    PROCESSED_DIR,
    PROCESSED_FILES,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SENTENCE_END_LABELS,
    SPLITS,
)
from src.data.deduplication import deduplicate_rows
from src.data.jointcappunc_parser import parse_split
from src.data.schema import make_example_id, make_source_id, validate_row
from src.utils.io import iter_jsonl, read_yaml, write_json, write_jsonl
from src.utils.logging_utils import get_logger, section

logger = get_logger("prepare_data")

def build_split(
    split: str,
    *,
    raw_data_dir: Path,
    max_words: int,
    boundary_labels: List[str],
    dedup_enabled: bool,
    dedup_key_mode: str,
    strict: bool,
    max_lines: Optional[int],
) -> Dict[str, Any]:
    """Run the full per-split pipeline and write its JSONL file."""
    section(f"  split: {split}")
    started = time.time()

    documents, parse_report = parse_split(
        split, raw_data_dir=raw_data_dir, strict=strict, max_lines=max_lines
    )


    rows: List[Dict[str, Any]] = []
    chunk_totals = {
        "input_tokens": 0,
        "sentences": 0,
        "segments": 0,
        "chunks": 0,
        "hard_cut_chunks": 0,
        "oversized_sentences": 0,
        "chunks_ending_at_boundary": 0,
    }
    example_counter = 0
    segment_counter = 0

    for doc in documents:
        chunks, stats = chunk_sequence(
            doc.tokens,
            doc.labels,
            max_words=max_words,
            end_labels=boundary_labels,
        )
        assert_chunking_invariants(doc.tokens, doc.labels, chunks, max_words=max_words)

        for key in chunk_totals:
            chunk_totals[key] += getattr(
                stats,
                key if key != "chunks_ending_at_boundary" else "chunks_ending_at_boundary",
            )

        last_segment = -1
        for chunk in chunks:
            if chunk.segment_index != last_segment:
                segment_counter += 1
                last_segment = chunk.segment_index
            example_counter += 1
            rows.append(
                {
                    "id": make_example_id(split, example_counter),
                    "source_split": split,
                    "source_id": make_source_id(split, segment_counter),
                    "chunk_index": chunk.chunk_index,
                    "tokens": chunk.tokens,
                    "labels": chunk.labels,
                }
            )

    logger.info(
        "%s: %d document(s) -> %d example(s) (%d hard-cut, %.2f%% end at a sentence boundary)",
        split,
        len(documents),
        len(rows),
        chunk_totals["hard_cut_chunks"],
        100.0 * chunk_totals["chunks_ending_at_boundary"] / max(1, chunk_totals["chunks"]),
    )

    if dedup_enabled:
        rows, dedup_report = deduplicate_rows(rows, split=split, key_mode=dedup_key_mode)
        logger.info(
            "%s: removed %d exact duplicate example(s) (%d kept)",
            split,
            dedup_report.removed_examples,
            dedup_report.kept_examples,
        )
        dedup_dict = dedup_report.to_dict()
    else:
        dedup_dict = {"enabled": False}

    for new_index, row in enumerate(rows, start=1):
        row["id"] = make_example_id(split, new_index)


    for i, row in enumerate(rows):
        validate_row(row, max_words=max_words, expected_split=split, where=f"{split}[{i}]")

    out_path = PROCESSED_FILES[split]
    n_written = write_jsonl(out_path, rows)
    elapsed = time.time() - started
    logger.info(
        "%s: wrote %d example(s) to %s in %.1fs",
        split,
        n_written,
        out_path.relative_to(PROJECT_ROOT).as_posix(),
        elapsed,
    )

    return {
        "split": split,
        "output_file": out_path.relative_to(PROJECT_ROOT).as_posix(),
        "num_examples": n_written,
        "num_tokens": sum(len(r["tokens"]) for r in rows),
        "parse": parse_report.to_dict(),
        "chunking": chunk_totals,
        "deduplication": dedup_dict,
        "seconds": round(elapsed, 2),
    }

def write_all_labeled() -> Dict[str, Any]:
    """Concatenate the three splits into the audit-only file."""
    section("  all_labeled.jsonl (audit / statistics only — NOT for training)")

    def rows():
        for split in SPLITS:
            for row in iter_jsonl(PROCESSED_FILES[split]):
                yield row

    n = write_jsonl(ALL_LABELED_FILE, rows())
    logger.info("all_labeled.jsonl: %d example(s)", n)
    return {
        "output_file": ALL_LABELED_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "num_examples": n,
        "purpose": "audit and statistics only; never used for training",
    }

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/processed/*.jsonl from raw JointCapPunc.")
    parser.add_argument("--config", default=str(CONFIG_DIR / "data.yaml"))
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="parse only the first N raw lines per split (development aid — "
        "never use for the real dataset)",
    )
    parser.add_argument("--raw-data-dir", default=str(RAW_DATA_DIR))
    args = parser.parse_args(argv)

    cfg = read_yaml(args.config)
    chunk_cfg = cfg.get("chunking", {})
    dedup_cfg = cfg.get("deduplication", {})
    parse_cfg = cfg.get("parsing", {})

    max_words = int(chunk_cfg.get("max_words_per_example", MAX_WORDS_PER_EXAMPLE))
    boundary_labels = list(chunk_cfg.get("boundary_labels", SENTENCE_END_LABELS))

    section("STAGE 2/6 — PARSE, NORMALIZE, CHUNK, DEDUPLICATE, WRITE JSONL")
    if args.max_lines:
        logger.warning(
            "--max-lines=%d is set: this produces a TRUNCATED dataset for development only.",
            args.max_lines,
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": Path(args.config).name,
        "max_words_per_example": max_words,
        "boundary_labels": boundary_labels,
        "deduplication_enabled": bool(dedup_cfg.get("enabled", True)),
        "deduplication_key_mode": str(dedup_cfg.get("key_mode", "text_and_labels")),
        "truncated_for_development": bool(args.max_lines),
        "splits": {},
    }

    for split in SPLITS:
        report["splits"][split] = build_split(
            split,
            raw_data_dir=Path(args.raw_data_dir),
            max_words=max_words,
            boundary_labels=boundary_labels,
            dedup_enabled=bool(dedup_cfg.get("enabled", True)),
            dedup_key_mode=str(dedup_cfg.get("key_mode", "text_and_labels")),
            strict=bool(parse_cfg.get("strict", True)),
            max_lines=args.max_lines,
        )

    report["all_labeled"] = write_all_labeled()
    write_json(OUTPUT_DATA_DIR / "preparation_report.json", report)

    print("\nProcessed dataset:")
    for split in SPLITS:
        s = report["splits"][split]
        print(f"  {split:<11} {s['num_examples']:>8,} examples  {s['num_tokens']:>12,} tokens")
    print(f"  {'all_labeled':<11} {report['all_labeled']['num_examples']:>8,} examples (audit only)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
