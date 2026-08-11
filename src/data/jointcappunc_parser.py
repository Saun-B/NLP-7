from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from src.data.constants import (
    RAW_CAPITALIZATION_LABELS,
    RAW_DATA_DIR,
    RAW_LABELS,
    SOURCE_SPLIT_FILES,
)
from src.data.normalization import map_raw_label, normalize_token
from src.utils.io import project_relative_path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

EXPECTED_COLUMNS = 3

class ParseError(ValueError):
    """Raised on a malformed raw line, with file and line number attached."""

    def __init__(self, path: PathLike, line_no: int, message: str, raw_line: str = ""):
        self.path = project_relative_path(path)
        self.line_no = line_no
        self.raw_line = raw_line
        super().__init__(f"{self.path}:{line_no}: {message} | raw={raw_line!r}")

@dataclass
class ParseReport:

    path: str = ""
    split: str = ""
    total_lines: int = 0
    blank_lines: int = 0
    parsed_tokens: int = 0
    dropped_empty_tokens: int = 0
    documents: int = 0
    raw_label_counts: Dict[str, int] = field(default_factory=dict)
    mapped_label_counts: Dict[str, int] = field(default_factory=dict)
    capitalization_label_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "split": self.split,
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "parsed_tokens": self.parsed_tokens,
            "dropped_empty_tokens": self.dropped_empty_tokens,
            "documents": self.documents,
            "raw_label_counts": dict(sorted(self.raw_label_counts.items())),
            "mapped_label_counts": dict(sorted(self.mapped_label_counts.items())),
            "capitalization_label_counts": dict(
                sorted(self.capitalization_label_counts.items())
            ),
            "num_errors": len(self.errors),
            "errors": self.errors[:50],
        }

@dataclass
class RawDocument:
    tokens: List[str]
    labels: List[str]
    start_line: int
    end_line: int

    def __len__(self) -> int:
        return len(self.tokens)

def iter_raw_lines(path: PathLike) -> Iterator[Tuple[int, str]]:
    """Yield ``(1-based line number, line without its newline)``."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            yield line_no, line.rstrip("\r\n")

def parse_line(path: PathLike, line_no: int, line: str) -> Tuple[str, str, str, str]:
    """Parse one raw line into ``(token, cap_label, raw_punc_label, mapped_label)``."""
    parts = line.split("\t")
    if len(parts) != EXPECTED_COLUMNS:
        raise ParseError(
            path,
            line_no,
            f"expected {EXPECTED_COLUMNS} tab-separated columns, got {len(parts)}",
            line,
        )

    raw_word, cap_label, punc_label = parts
    cap_label = cap_label.strip()
    punc_label = punc_label.strip().upper()

    if cap_label not in RAW_CAPITALIZATION_LABELS:
        raise ParseError(
            path,
            line_no,
            f"unknown capitalization label {cap_label!r} "
            f"(expected one of {RAW_CAPITALIZATION_LABELS})",
            line,
        )
    if punc_label not in RAW_LABELS:
        raise ParseError(
            path,
            line_no,
            f"unknown punctuation label {punc_label!r} (expected one of {RAW_LABELS})",
            line,
        )

    token = normalize_token(raw_word, lowercase=True)
    mapped = map_raw_label(punc_label)
    return token, cap_label, punc_label, mapped

def parse_file(
    path: PathLike,
    split: str,
    *,
    strict: bool = True,
    max_lines: Optional[int] = None,
    intern_tokens: bool = True,
) -> Tuple[List[RawDocument], ParseReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}. Run `python scripts/download_data.py` first."
        )

    report = ParseReport(path=project_relative_path(path), split=split)
    documents: List[RawDocument] = []

    tokens: List[str] = []
    labels: List[str] = []
    doc_start_line = 1

    def flush(end_line: int) -> None:
        nonlocal tokens, labels, doc_start_line
        if tokens:
            documents.append(
                RawDocument(
                    tokens=tokens, labels=labels, start_line=doc_start_line, end_line=end_line
                )
            )
        tokens, labels = [], []
        doc_start_line = end_line + 1

    intern = sys.intern if intern_tokens else (lambda s: s)
    last_line_no = 0

    raw_counts = report.raw_label_counts
    mapped_counts = report.mapped_label_counts
    cap_counts = report.capitalization_label_counts

    for line_no, line in iter_raw_lines(path):
        last_line_no = line_no
        report.total_lines += 1
        if max_lines is not None and line_no > max_lines:
            report.total_lines -= 1
            last_line_no = line_no - 1
            break

        if not line.strip():

            report.blank_lines += 1
            flush(line_no)
            continue

        try:
            token, cap_label, raw_punc, mapped = parse_line(path, line_no, line)
        except ParseError as exc:
            if strict:
                raise
            report.errors.append(str(exc))
            continue

        raw_counts[raw_punc] = raw_counts.get(raw_punc, 0) + 1
        mapped_counts[mapped] = mapped_counts.get(mapped, 0) + 1
        cap_counts[cap_label] = cap_counts.get(cap_label, 0) + 1

        if not token:

            report.dropped_empty_tokens += 1
            if strict:
                raise ParseError(path, line_no, "token is empty after normalization", line)
            report.errors.append(f"{path}:{line_no}: empty token after normalization")
            continue

        tokens.append(intern(token))
        labels.append(intern(mapped))
        report.parsed_tokens += 1

    flush(last_line_no)
    report.documents = len(documents)

    logger.info(
        "Parsed %s: %d lines -> %d tokens in %d document(s); labels=%s",
        path.name,
        report.total_lines,
        report.parsed_tokens,
        report.documents,
        report.mapped_label_counts,
    )
    if report.errors:
        logger.warning("%s: %d malformed line(s) recorded", path.name, len(report.errors))
    return documents, report

def parse_split(
    split: str,
    *,
    raw_data_dir: PathLike = RAW_DATA_DIR,
    strict: bool = True,
    max_lines: Optional[int] = None,
) -> Tuple[List[RawDocument], ParseReport]:
    """Parse an official split by name, resolving the upstream filename."""
    if split not in SOURCE_SPLIT_FILES:
        raise KeyError(f"Unknown split {split!r}; expected one of {sorted(SOURCE_SPLIT_FILES)}")
    path = Path(raw_data_dir) / SOURCE_SPLIT_FILES[split]
    return parse_file(path, split, strict=strict, max_lines=max_lines)
