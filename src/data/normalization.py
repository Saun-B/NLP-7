"""Text normalization and raw-label mapping.

Design rules (non-negotiable for this project)
----------------------------------------------
1. **Unicode NFC.** Vietnamese can be written with precomposed characters
   (``ế`` = U+1EBF) or with combining marks (``ê`` + U+0301). Those are two
   different strings for Python and two different vocabulary entries for a
   model. Everything is normalised to NFC exactly once, here.

2. **Diacritics are preserved.** We *never* strip tone marks:
   ``"tiếng Việt"`` must stay ``"tiếng việt"``, never ``"tieng viet"``.
   :func:`strip_diacritics` exists only so tests can prove the pipeline does
   not call it — it is not used anywhere in the pipeline.

3. **Case.** The task input is unpunctuated, uncased text, and the JointCapPunc
   token column is already lowercase (capitalization lives in a separate column
   that this project does not use). Lowercasing is therefore applied for model
   input and is a no-op on this corpus, but it makes the pipeline robust if a
   different source is ever plugged in.

4. **Label mapping.** ``QMARK -> QUESTION``. No other renaming; unknown raw
   labels raise instead of being silently bucketed into ``O``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from src.data.constants import LABEL2ID, RAW_LABEL_MAP

__all__ = [
    "nfc",
    "normalize_token",
    "normalize_tokens",
    "normalize_text",
    "map_raw_label",
    "map_raw_labels",
    "canonical_text",
    "strip_diacritics",
]


_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def nfc(text: str) -> str:
    """Unicode NFC normalisation (composes diacritics, keeps them)."""
    return unicodedata.normalize("NFC", text)


def normalize_token(token: str, *, lowercase: bool = True) -> str:
    """Normalise a single lexical word.

    NFC -> strip surrounding whitespace -> collapse inner whitespace ->
    optional lowercase. Tone marks are always kept.
    """
    out = nfc(token).strip()
    out = _WHITESPACE_RE.sub(" ", out)
    if lowercase:
        out = out.lower()
    return out


def normalize_tokens(tokens: Iterable[str], *, lowercase: bool = True) -> List[str]:
    """Normalise a sequence of tokens, dropping ones that become empty.

    Dropping is *reported* by the caller (the parser counts them) — it never
    happens silently at pipeline level.
    """
    out: List[str] = []
    for tok in tokens:
        norm = normalize_token(tok, lowercase=lowercase)
        if norm:
            out.append(norm)
    return out


def normalize_text(text: str, *, lowercase: bool = True) -> str:
    """Normalise a whole sentence/paragraph (NFC + whitespace + case)."""
    out = nfc(text).strip()
    out = _WHITESPACE_RE.sub(" ", out)
    if lowercase:
        out = out.lower()
    return out


def map_raw_label(raw_label: str) -> str:
    """Map an upstream punctuation label to the project label space.

    ``QMARK -> QUESTION``; ``O/COMMA/PERIOD`` pass through unchanged.

    Raises
    ------
    ValueError
        If the label is not one of the four upstream labels. Failing loudly is
        deliberate: a silently-bucketed label would corrupt every metric
        downstream.
    """
    key = raw_label.strip().upper()
    try:
        mapped = RAW_LABEL_MAP[key]
    except KeyError:
        raise ValueError(
            f"Unknown raw punctuation label {raw_label!r}. "
            f"Expected one of {sorted(RAW_LABEL_MAP)}."
        ) from None
    if mapped not in LABEL2ID:
        raise ValueError(f"Mapped label {mapped!r} is not in LABEL2ID.")
    return mapped


def map_raw_labels(raw_labels: Iterable[str]) -> List[str]:
    return [map_raw_label(x) for x in raw_labels]


def canonical_text(tokens: Iterable[str]) -> str:
    """Canonical string form of a token sequence, used as a dedup / overlap key.

    Two examples are "the same text" iff their canonical forms are equal.
    """
    return normalize_text(" ".join(tokens), lowercase=True)


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese tone marks. **Never used by the pipeline.**

    Kept only as an explicit, named counter-example so that
    ``tests/test_normalization.py`` can assert the pipeline output still
    contains diacritics (i.e. ``normalize_token(x) != strip_diacritics(x)`` for
    accented input).
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")
