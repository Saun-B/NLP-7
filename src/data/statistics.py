from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from src.data.constants import (
    ALL_LABELED_FILE,
    LABELS,
    LABEL_TO_SYMBOL,
    NUM_LABELS,
    OUTPUT_DATA_DIR,
    PROCESSED_FILES,
    SEED,
    SPLITS,
)
from src.utils.hashing import hash_jsonl_dataset, missing_hash_record
from src.utils.io import iter_jsonl, write_csv, write_json

PathLike = Union[str, Path]

__all__ = [
    "SplitStatistics",
    "compute_split_statistics",
    "compute_all_statistics",
    "compute_class_weights",
    "statistics_to_csv_rows",
    "build_human_review_samples",
    "compute_dataset_hashes",
    "render_text_with_punctuation",
]

@dataclass
class SplitStatistics:
    split: str
    num_examples: int = 0
    num_tokens: int = 0
    min_length: int = 0
    max_length: int = 0
    mean_length: float = 0.0
    median_length: float = 0.0
    p95_length: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    label_ratios: Dict[str, float] = field(default_factory=dict)
    num_sentences: int = 0
    vocabulary_size: int = 0
    num_source_segments: int = 0
    num_hard_cut_examples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "num_examples": self.num_examples,
            "num_tokens": self.num_tokens,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "mean_length": round(self.mean_length, 4),
            "median_length": self.median_length,
            "p95_length": self.p95_length,
            "label_counts": {k: self.label_counts.get(k, 0) for k in LABELS},
            "label_ratios": {k: round(self.label_ratios.get(k, 0.0), 8) for k in LABELS},
            "num_sentences": self.num_sentences,
            "vocabulary_size": self.vocabulary_size,
            "num_source_segments": self.num_source_segments,
            "num_hard_cut_examples": self.num_hard_cut_examples,
        }

def compute_split_statistics(path: PathLike, split: str) -> SplitStatistics:
    """Stream a processed JSONL file and compute descriptive statistics."""
    stats = SplitStatistics(split=split)
    lengths: List[int] = []
    label_counts: Counter[str] = Counter()
    vocab: set[str] = set()
    segments: set[str] = set()

    for row in iter_jsonl(path):
        tokens = row["tokens"]
        labels = row["labels"]
        stats.num_examples += 1
        lengths.append(len(tokens))
        label_counts.update(labels)
        vocab.update(tokens)
        segments.add(row.get("source_id", ""))
        if int(row.get("chunk_index", 0)) > 0:
            stats.num_hard_cut_examples += 1

    stats.num_tokens = sum(lengths)
    if lengths:
        lengths.sort()
        stats.min_length = lengths[0]
        stats.max_length = lengths[-1]
        stats.mean_length = stats.num_tokens / len(lengths)
        mid = len(lengths) // 2
        stats.median_length = (
            float(lengths[mid])
            if len(lengths) % 2 == 1
            else (lengths[mid - 1] + lengths[mid]) / 2.0
        )
        stats.p95_length = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]

    stats.label_counts = {k: int(label_counts.get(k, 0)) for k in LABELS}
    total = stats.num_tokens or 1
    stats.label_ratios = {k: stats.label_counts[k] / total for k in LABELS}
    stats.num_sentences = stats.label_counts["PERIOD"] + stats.label_counts["QUESTION"]
    stats.vocabulary_size = len(vocab)
    stats.num_source_segments = len(segments)
    return stats

def compute_all_statistics(
    files: Optional[Dict[str, PathLike]] = None,
    *,
    all_labeled_file: Optional[PathLike] = ALL_LABELED_FILE,
) -> Dict[str, Any]:
    """Statistics for every split (+ the audit-only ``all_labeled`` file)."""
    files = files or dict(PROCESSED_FILES)
    per_split: Dict[str, Any] = {}
    for split in SPLITS:
        path = Path(files[split])
        if not path.exists():
            raise FileNotFoundError(f"Processed split missing: {path}")
        per_split[split] = compute_split_statistics(path, split).to_dict()

    totals = {
        "num_examples": sum(per_split[s]["num_examples"] for s in SPLITS),
        "num_tokens": sum(per_split[s]["num_tokens"] for s in SPLITS),
        "label_counts": {
            lab: sum(per_split[s]["label_counts"][lab] for s in SPLITS) for lab in LABELS
        },
    }
    grand_total = totals["num_tokens"] or 1
    totals["label_ratios"] = {
        lab: round(totals["label_counts"][lab] / grand_total, 8) for lab in LABELS
    }

    out: Dict[str, Any] = {
        "labels": LABELS,
        "splits": per_split,
        "total": totals,
        "split_example_share": {
            s: round(per_split[s]["num_examples"] / (totals["num_examples"] or 1), 6)
            for s in SPLITS
        },
    }

    if all_labeled_file is not None and Path(all_labeled_file).exists():
        audit = compute_split_statistics(all_labeled_file, "all_labeled")
        audit_dict = audit.to_dict()
        audit_dict["note"] = "audit/statistics only — never used for training"
        out["all_labeled"] = audit_dict

    return out

def compute_class_weights(
    train_label_counts: Dict[str, int], *, num_classes: int = NUM_LABELS
) -> Dict[str, Any]:
    """Inverse-frequency and sqrt-inverse class weights from train counts."""
    counts = {lab: int(train_label_counts.get(lab, 0)) for lab in LABELS}
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot compute class weights: train label counts are all zero.")

    inverse: Dict[str, float] = {}
    for lab in LABELS:
        c = counts[lab]
        if c == 0:
            raise ValueError(
                f"Class {lab!r} has zero occurrences in train — cannot build weights."
            )
        inverse[lab] = total / (num_classes * c)

    sqrt_inverse = {lab: math.sqrt(w) for lab, w in inverse.items()}

    return {
        "source_split": "train",
        "note": (
            "Weights are computed from the TRAIN split only. Using validation or "
            "test counts here would leak evaluation information into the loss."
        ),
        "num_classes": num_classes,
        "total_train_tokens": total,
        "label_order": LABELS,
        "counts": counts,
        "frequencies": {lab: counts[lab] / total for lab in LABELS},
        "formulas": {
            "inverse": "w_c = total_train_tokens / (num_classes * count_c)",
            "sqrt_inverse": "w_c = sqrt(inverse_weight_c)",
        },
        "inverse": inverse,
        "sqrt_inverse": sqrt_inverse,
        "none": {lab: 1.0 for lab in LABELS},

        "inverse_vector": [inverse[lab] for lab in LABELS],
        "sqrt_inverse_vector": [sqrt_inverse[lab] for lab in LABELS],
        "none_vector": [1.0] * num_classes,
    }

def statistics_to_csv_rows(stats: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    sections = [(s, stats["splits"][s]) for s in SPLITS]
    if "all_labeled" in stats:
        sections.append(("all_labeled", stats["all_labeled"]))
    for name, s in sections:
        row: List[Any] = [
            name,
            s["num_examples"],
            s["num_tokens"],
            s["min_length"],
            s["max_length"],
            s["mean_length"],
            s["median_length"],
            s["p95_length"],
            s["vocabulary_size"],
            s["num_sentences"],
        ]
        for lab in LABELS:
            row.append(s["label_counts"][lab])
        for lab in LABELS:
            row.append(round(s["label_ratios"][lab], 6))
        rows.append(row)
    return rows

STATISTICS_CSV_HEADER: List[str] = (
    [
        "split",
        "num_examples",
        "num_tokens",
        "min_length",
        "max_length",
        "mean_length",
        "median_length",
        "p95_length",
        "vocabulary_size",
        "num_sentences",
    ]
    + [f"count_{lab}" for lab in LABELS]
    + [f"ratio_{lab}" for lab in LABELS]
)

def render_text_with_punctuation(
    tokens: Sequence[str], labels: Sequence[str], *, capitalize: bool = True
) -> str:
    pieces: List[str] = []
    start_of_sentence = True
    for tok, lab in zip(tokens, labels):
        word = tok
        if capitalize and start_of_sentence and word:
            word = word[0].upper() + word[1:]
        pieces.append(word + LABEL_TO_SYMBOL.get(lab, ""))
        start_of_sentence = lab in ("PERIOD", "QUESTION")
    return " ".join(pieces)

def build_human_review_samples(
    files: Optional[Dict[str, PathLike]] = None,
    *,
    n_samples: int = 100,
    seed: int = SEED,
) -> List[List[Any]]:

    files = files or dict(PROCESSED_FILES)
    pool: List[Dict[str, Any]] = []
    for split in SPLITS:
        for row in iter_jsonl(files[split]):
            pool.append(row)

    rng = random.Random(seed)
    if len(pool) <= n_samples:
        chosen = pool
    else:
        chosen = rng.sample(pool, n_samples)
    chosen.sort(key=lambda r: r["id"])

    rows: List[List[Any]] = []
    for row in chosen:
        rows.append(
            [
                row["id"],
                row["source_split"],
                render_text_with_punctuation(row["tokens"], row["labels"]),
                " ".join(row["tokens"]),
                " ".join(row["labels"]),
                "",
                "",
            ]
        )
    return rows

HUMAN_REVIEW_HEADER: List[str] = [
    "id",
    "source_split",
    "text",
    "tokens",
    "labels",
    "is_correct",
    "review_note",
]

def compute_dataset_hashes(
    files: Optional[Dict[str, PathLike]] = None,
    *,
    all_labeled_file: Optional[PathLike] = ALL_LABELED_FILE,
) -> Dict[str, Any]:
    """SHA-256 records for every processed file (pinned into each experiment)."""
    files = files or dict(PROCESSED_FILES)
    out: Dict[str, Any] = {}
    for split in SPLITS:
        p = Path(files[split])
        out[split] = hash_jsonl_dataset(p) if p.exists() else missing_hash_record(p)
    if all_labeled_file is not None:
        p = Path(all_labeled_file)
        out["all_labeled"] = hash_jsonl_dataset(p) if p.exists() else missing_hash_record(p)
    return out

def write_statistics_artifacts(
    output_dir: PathLike = OUTPUT_DATA_DIR,
    files: Optional[Dict[str, PathLike]] = None,
    *,
    n_review_samples: int = 100,
    seed: int = SEED,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    files = files or dict(PROCESSED_FILES)

    stats = compute_all_statistics(files)
    write_json(output_dir / "data_statistics.json", stats)
    write_csv(
        output_dir / "data_statistics.csv",
        STATISTICS_CSV_HEADER,
        statistics_to_csv_rows(stats),
    )

    weights = compute_class_weights(stats["splits"]["train"]["label_counts"])
    write_json(output_dir / "class_weights.json", weights)

    hashes = compute_dataset_hashes(files)
    write_json(output_dir / "data_hashes.json", hashes)

    review_rows = build_human_review_samples(files, n_samples=n_review_samples, seed=seed)
    write_csv(output_dir / "human_review_samples.csv", HUMAN_REVIEW_HEADER, review_rows)

    return {
        "statistics": stats,
        "class_weights": weights,
        "hashes": hashes,
        "num_review_samples": len(review_rows),
    }
