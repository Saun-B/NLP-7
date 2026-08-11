"""Exact-duplicate removal, applied **within** each official split.

Rules
-----
* A duplicate is detected on a *normalised* key, so casing/whitespace noise
  cannot hide one.
* Two key modes:

  ``text_and_labels`` (default, used by the pipeline)
      Key = normalised text **plus** the label sequence. Removes only rows that
      are byte-for-byte the same supervision signal.
  ``text``
      Key = normalised text only. Reported as a statistic so we can see how
      many rows share text but disagree on labels (a data-quality signal).

* Deduplication **never moves an example between splits** — that would destroy
  the official split. Cross-split overlap is measured separately in
  :mod:`src.data.validation` and reported as a warning, not silently "fixed".
* The first occurrence in file order is kept, so the result is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.data.normalization import canonical_text
from src.utils.hashing import sha256_text

__all__ = [
    "DedupReport",
    "example_text_key",
    "example_full_key",
    "deduplicate_rows",
]


@dataclass
class DedupReport:
    split: str = ""
    input_examples: int = 0
    kept_examples: int = 0
    removed_examples: int = 0
    removed_tokens: int = 0
    key_mode: str = "text_and_labels"

    text_collisions_with_different_labels: int = 0
    removed_example_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        ratio = self.removed_examples / self.input_examples if self.input_examples else 0.0
        return {
            "split": self.split,
            "key_mode": self.key_mode,
            "input_examples": self.input_examples,
            "kept_examples": self.kept_examples,
            "removed_examples": self.removed_examples,
            "removed_ratio": round(ratio, 6),
            "removed_tokens": self.removed_tokens,
            "text_collisions_with_different_labels": (
                self.text_collisions_with_different_labels
            ),
            "removed_example_ids_sample": self.removed_example_ids[:25],
        }


def example_text_key(tokens: Sequence[str]) -> str:
    """Normalised-text dedup key (hashed to keep memory flat on 5M tokens)."""
    return sha256_text(canonical_text(tokens))


def example_full_key(tokens: Sequence[str], labels: Sequence[str]) -> str:
    """Normalised text + label sequence dedup key."""
    return sha256_text(canonical_text(tokens) + "␟" + " ".join(labels))


def deduplicate_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    split: str = "",
    key_mode: str = "text_and_labels",
) -> Tuple[List[Dict[str, Any]], DedupReport]:
    """Remove exact duplicates from one split, keeping the first occurrence."""
    if key_mode not in {"text_and_labels", "text"}:
        raise ValueError(f"Unknown key_mode {key_mode!r}")

    report = DedupReport(split=split, key_mode=key_mode)
    seen_full: set[str] = set()
    seen_text: Dict[str, str] = {}
    kept: List[Dict[str, Any]] = []

    for row in rows:
        report.input_examples += 1
        tokens = row["tokens"]
        labels = row["labels"]
        t_key = example_text_key(tokens)
        f_key = sha256_text(t_key + "␟" + " ".join(labels))

        label_sig = sha256_text(" ".join(labels))
        if t_key in seen_text and seen_text[t_key] != label_sig:
            report.text_collisions_with_different_labels += 1
        else:
            seen_text.setdefault(t_key, label_sig)

        dup_key = f_key if key_mode == "text_and_labels" else t_key
        if dup_key in seen_full:
            report.removed_examples += 1
            report.removed_tokens += len(tokens)
            if len(report.removed_example_ids) < 1000:
                report.removed_example_ids.append(row.get("id", "<no-id>"))
            continue

        seen_full.add(dup_key)
        kept.append(row)

    report.kept_examples = len(kept)
    return kept, report
