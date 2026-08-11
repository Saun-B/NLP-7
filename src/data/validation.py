"""Post-hoc validation of the processed dataset.

The pipeline writes ``data/processed/*.jsonl``; this module then re-reads those
files from disk and tries to break them. Checking the *written artifact* rather
than the in-memory objects is deliberate — it catches serialisation bugs too.

Severity model
--------------
``ERROR``
    A hard invariant is broken (missing file, bad schema, duplicate id,
    length mismatch, id overlap between splits, a split missing a punctuation
    class …). Any error makes the report ``passed = false`` and the pipeline
    exits non-zero.

``WARNING``
    Something a human should look at but that does not invalidate the dataset.
    The most important one is **cross-split text overlap**: JointCapPunc is a
    large crawled corpus, so short generic utterances ("vâng ạ.", "cảm ơn bác
    sĩ.") legitimately appear in more than one official split. We report the
    exact count and show examples, but we do **not** delete or move rows —
    the official split has to stay intact, and silently "fixing" leakage would
    change the benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.data.constants import (
    ALL_LABELED_FILE,
    LABELS,
    MAX_WORDS_PER_EXAMPLE,
    PROCESSED_FILES,
    PUNCTUATION_LABELS,
    SPLITS,
)
from src.data.normalization import canonical_text
from src.data.schema import SchemaError, validate_row
from src.utils.hashing import sha256_text
from src.utils.io import iter_jsonl, project_relative_path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

ERROR = "ERROR"
WARNING = "WARNING"



OVERLAP_WARN_RATIO = 0.02


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationReport:
    def __init__(self) -> None:
        self.issues: List[Issue] = []
        self.checks: List[Dict[str, Any]] = []
        self.split_summary: Dict[str, Any] = {}


    def error(self, code: str, message: str, **details: Any) -> None:
        self.issues.append(Issue(ERROR, code, message, details))
        logger.error("[%s] %s", code, message)

    def warn(self, code: str, message: str, **details: Any) -> None:
        self.issues.append(Issue(WARNING, code, message, details))
        logger.warning("[%s] %s", code, message)

    def check(self, name: str, passed: bool, **details: Any) -> bool:
        self.checks.append({"check": name, "passed": bool(passed), **details})
        return passed


    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "checks": self.checks,
            "splits": self.split_summary,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary_text(self) -> str:
        lines = [
            f"Validation: {'PASS' if self.passed else 'FAIL'} "
            f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
        ]
        for issue in self.issues[:40]:
            lines.append(f"  [{issue.severity}] {issue.code}: {issue.message}")
        if len(self.issues) > 40:
            lines.append(f"  … {len(self.issues) - 40} more")
        return "\n".join(lines)





def _validate_split_file(
    report: ValidationReport,
    split: str,
    path: PathLike,
    *,
    max_words: int,
    max_schema_errors: int,
) -> Dict[str, Any]:
    """Schema + invariant checks for one split. Returns per-split facts."""
    path = Path(path)
    facts: Dict[str, Any] = {
        "path": project_relative_path(path),
        "exists": path.exists(),
        "num_examples": 0,
        "num_tokens": 0,
        "label_counts": {lab: 0 for lab in LABELS},
        "max_length": 0,
        "min_length": 0,
        "ids": 0,
        "text_hashes": set(),
        "id_set": set(),
        "source_id_set": set(),
    }

    if not path.exists():
        report.error("FILE_MISSING", f"{split}: file not found at {path}", split=split)
        return facts

    schema_errors: List[str] = []
    ids: set[str] = set()
    duplicate_ids: List[str] = []
    text_hashes: set[str] = set()
    source_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    lengths_min: Optional[int] = None
    lengths_max = 0
    n = 0
    n_tokens = 0
    over_length = 0

    for line_no, row in enumerate(iter_jsonl(path), start=1):
        n += 1
        try:
            validate_row(
                row,
                max_words=max_words,
                expected_split=split,
                where=f"{path.name}:{line_no}",
            )
        except SchemaError as exc:
            if len(schema_errors) < max_schema_errors:
                schema_errors.append(str(exc))
            if len(row.get("tokens", [])) > max_words:
                over_length += 1
            continue

        rid = row["id"]
        if rid in ids:
            if len(duplicate_ids) < 25:
                duplicate_ids.append(rid)
        ids.add(rid)
        source_ids.add(row["source_id"])

        tokens = row["tokens"]
        labels = row["labels"]
        length = len(tokens)
        n_tokens += length
        lengths_max = max(lengths_max, length)
        lengths_min = length if lengths_min is None else min(lengths_min, length)
        label_counts.update(labels)
        text_hashes.add(sha256_text(canonical_text(tokens)))

    facts.update(
        {
            "num_examples": n,
            "num_tokens": n_tokens,
            "num_unique_ids": len(ids),
            "num_duplicate_ids": n - len(ids),
            "num_source_segments": len(source_ids),
            "label_counts": {lab: int(label_counts.get(lab, 0)) for lab in LABELS},
            "min_length": lengths_min or 0,
            "max_length": lengths_max,
            "num_schema_errors": len(schema_errors),
            "text_hashes": text_hashes,
            "id_set": ids,
            "source_id_set": source_ids,
        }
    )


    report.check(f"{split}.file_exists", True, path=project_relative_path(path))

    if n == 0:
        report.error("EMPTY_SPLIT", f"{split}: dataset is empty", split=split)
    report.check(f"{split}.non_empty", n > 0, num_examples=n)

    if schema_errors:
        report.error(
            "SCHEMA_INVALID",
            f"{split}: {len(schema_errors)} row(s) violate the schema "
            f"(showing up to {max_schema_errors})",
            split=split,
            examples=schema_errors,
        )
    report.check(f"{split}.schema_valid", not schema_errors, num_bad_rows=len(schema_errors))

    if duplicate_ids:
        report.error(
            "DUPLICATE_ID",
            f"{split}: {n - len(ids)} duplicate id(s)",
            split=split,
            examples=duplicate_ids,
        )
    report.check(f"{split}.unique_ids", not duplicate_ids, num_duplicates=n - len(ids))

    if lengths_max > max_words:
        report.error(
            "LENGTH_EXCEEDED",
            f"{split}: longest example has {lengths_max} tokens (cap {max_words})",
            split=split,
        )
    report.check(f"{split}.max_length_ok", lengths_max <= max_words, max_length=lengths_max)

    missing_classes = [lab for lab in PUNCTUATION_LABELS if label_counts.get(lab, 0) == 0]
    if missing_classes:
        report.error(
            "MISSING_PUNCTUATION_CLASS",
            f"{split}: punctuation class(es) absent: {missing_classes}",
            split=split,
        )
    report.check(
        f"{split}.all_punctuation_classes_present",
        not missing_classes,
        missing=missing_classes,
    )

    return facts





def _validate_cross_split(report: ValidationReport, facts: Dict[str, Dict[str, Any]]) -> None:
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]

    for a, b in pairs:
        ids_a = facts[a]["id_set"]
        ids_b = facts[b]["id_set"]
        shared_ids = ids_a & ids_b
        if shared_ids:
            report.error(
                "ID_OVERLAP",
                f"{a}/{b}: {len(shared_ids)} id(s) appear in both splits",
                split_a=a,
                split_b=b,
                examples=sorted(shared_ids)[:20],
            )
        report.check(f"{a}_vs_{b}.no_id_overlap", not shared_ids, num_shared=len(shared_ids))

        seg_a = facts[a]["source_id_set"]
        seg_b = facts[b]["source_id_set"]
        shared_seg = seg_a & seg_b
        if shared_seg:
            report.error(
                "SOURCE_ID_OVERLAP",
                f"{a}/{b}: {len(shared_seg)} source segment id(s) shared",
                split_a=a,
                split_b=b,
                examples=sorted(shared_seg)[:20],
            )
        report.check(
            f"{a}_vs_{b}.no_source_id_overlap", not shared_seg, num_shared=len(shared_seg)
        )

        texts_a = facts[a]["text_hashes"]
        texts_b = facts[b]["text_hashes"]
        shared_text = texts_a & texts_b
        smaller = min(len(texts_a), len(texts_b)) or 1
        ratio = len(shared_text) / smaller
        if shared_text:
            report.warn(
                "TEXT_OVERLAP",
                f"{a}/{b}: {len(shared_text)} identical normalized example text(s) "
                f"({ratio:.4%} of the smaller split). The official JointCapPunc split "
                f"is preserved as-is; overlapping rows are reported, not removed.",
                split_a=a,
                split_b=b,
                num_shared_texts=len(shared_text),
                overlap_ratio=round(ratio, 6),
                exceeds_threshold=bool(ratio > OVERLAP_WARN_RATIO),
            )
        report.check(
            f"{a}_vs_{b}.text_overlap_measured",
            True,
            num_shared_texts=len(shared_text),
            overlap_ratio=round(ratio, 6),
            severity="warning-only",
        )





def _validate_all_labeled(
    report: ValidationReport,
    all_labeled_path: PathLike,
    facts: Dict[str, Dict[str, Any]],
    *,
    max_words: int,
) -> None:
    path = Path(all_labeled_path)
    if not path.exists():
        report.error("FILE_MISSING", f"all_labeled: file not found at {path}")
        return

    n = 0
    per_split: Counter[str] = Counter()
    bad = 0
    for line_no, row in enumerate(iter_jsonl(path), start=1):
        n += 1
        try:
            validate_row(row, max_words=max_words, where=f"{path.name}:{line_no}")
        except SchemaError:
            bad += 1
            continue
        per_split[row["source_split"]] += 1

    expected_total = sum(facts[s]["num_examples"] for s in SPLITS)
    matches = all(per_split.get(s, 0) == facts[s]["num_examples"] for s in SPLITS)
    if bad:
        report.error("SCHEMA_INVALID", f"all_labeled: {bad} row(s) violate the schema")
    if n != expected_total or not matches:
        report.error(
            "ALL_LABELED_MISMATCH",
            f"all_labeled has {n} rows / {dict(per_split)} but the splits hold "
            f"{expected_total} rows / "
            f"{ {s: facts[s]['num_examples'] for s in SPLITS} }",
        )
    report.check(
        "all_labeled.consistent_with_splits",
        n == expected_total and matches and not bad,
        num_rows=n,
        per_split=dict(per_split),
    )





def validate_processed_data(
    files: Optional[Dict[str, PathLike]] = None,
    *,
    all_labeled_file: Optional[PathLike] = ALL_LABELED_FILE,
    max_words: int = MAX_WORDS_PER_EXAMPLE,
    max_schema_errors: int = 20,
) -> ValidationReport:
    """Run every validation check and return the report."""
    files = files or dict(PROCESSED_FILES)
    report = ValidationReport()

    facts: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        facts[split] = _validate_split_file(
            report,
            split,
            files[split],
            max_words=max_words,
            max_schema_errors=max_schema_errors,
        )

    if all(facts[s]["exists"] for s in SPLITS):
        _validate_cross_split(report, facts)
        if all_labeled_file is not None:
            _validate_all_labeled(report, all_labeled_file, facts, max_words=max_words)


    report.split_summary = {
        s: {k: v for k, v in facts[s].items() if not isinstance(v, set)} for s in SPLITS
    }
    return report
