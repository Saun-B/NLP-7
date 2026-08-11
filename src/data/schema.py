from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.data.constants import (
    LABEL2ID,
    MAX_WORDS_PER_EXAMPLE,
    SPLITS,
)

SCHEMA_FIELDS: List[str] = [
    "id",
    "source_split",
    "source_id",
    "chunk_index",
    "tokens",
    "labels",
]

SCHEMA_VERSION: str = "1.0"


class SchemaError(ValueError):
    """Raised when a row violates the processed-example schema."""


@dataclass
class Example:

    id: str
    source_split: str
    source_id: str
    chunk_index: int
    tokens: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_split": self.source_split,
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "tokens": list(self.tokens),
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Example":
        validate_row(row)
        return cls(
            id=row["id"],
            source_split=row["source_split"],
            source_id=row["source_id"],
            chunk_index=int(row["chunk_index"]),
            tokens=list(row["tokens"]),
            labels=list(row["labels"]),
        )

    @property
    def text(self) -> str:
        """The unpunctuated model input."""
        return " ".join(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

def make_example_id(split: str, index: int) -> str:
    return f"{split}_{index:06d}"

def make_source_id(split: str, index: int) -> str:
    return f"{split}_seg_{index:06d}"

def validate_row(
    row: Any,
    *,
    max_words: int = MAX_WORDS_PER_EXAMPLE,
    expected_split: str | None = None,
    where: str = "<row>",
) -> None:
    if not isinstance(row, dict):
        raise SchemaError(f"{where}: expected a JSON object, got {type(row).__name__}")

    missing = [f for f in SCHEMA_FIELDS if f not in row]
    if missing:
        raise SchemaError(f"{where}: missing field(s): {missing}")

    unexpected = [k for k in row if k not in SCHEMA_FIELDS]
    if unexpected:
        raise SchemaError(f"{where}: unexpected field(s): {sorted(unexpected)}")

    if not isinstance(row["id"], str) or not row["id"]:
        raise SchemaError(f"{where}: 'id' must be a non-empty string")

    split = row["source_split"]
    if split not in SPLITS:
        raise SchemaError(f"{where}: 'source_split' must be one of {SPLITS}, got {split!r}")
    if expected_split is not None and split != expected_split:
        raise SchemaError(
            f"{where}: 'source_split' is {split!r} but this file holds {expected_split!r}"
        )

    if not isinstance(row["source_id"], str) or not row["source_id"]:
        raise SchemaError(f"{where}: 'source_id' must be a non-empty string")

    chunk_index = row["chunk_index"]
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
        raise SchemaError(f"{where}: 'chunk_index' must be a non-negative int")

    tokens = row["tokens"]
    labels = row["labels"]
    if not isinstance(tokens, list) or not isinstance(labels, list):
        raise SchemaError(f"{where}: 'tokens' and 'labels' must both be lists")
    if len(tokens) == 0:
        raise SchemaError(f"{where}: 'tokens' must not be empty")
    if len(tokens) != len(labels):
        raise SchemaError(
            f"{where}: length mismatch — len(tokens)={len(tokens)} != len(labels)={len(labels)}"
        )
    if len(tokens) > max_words:
        raise SchemaError(
            f"{where}: {len(tokens)} tokens exceeds the {max_words}-word cap"
        )

    for i, tok in enumerate(tokens):
        if not isinstance(tok, str):
            raise SchemaError(f"{where}: tokens[{i}] is {type(tok).__name__}, expected str")
        if not tok.strip():
            raise SchemaError(f"{where}: tokens[{i}] is empty or whitespace-only")

    for i, lab in enumerate(labels):
        if not isinstance(lab, str):
            raise SchemaError(f"{where}: labels[{i}] is {type(lab).__name__}, expected str")
        if lab not in LABEL2ID:
            raise SchemaError(
                f"{where}: labels[{i}] = {lab!r} is not a valid label {sorted(LABEL2ID)}"
            )

def validate_rows(
    rows: Sequence[Dict[str, Any]], *, expected_split: str | None = None
) -> None:
    for i, row in enumerate(rows):
        validate_row(row, expected_split=expected_split, where=f"row[{i}]")
