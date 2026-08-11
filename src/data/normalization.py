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
    return unicodedata.normalize("NFC", text)

def normalize_token(token: str, *, lowercase: bool = True) -> str:
    out = nfc(token).strip()
    out = _WHITESPACE_RE.sub(" ", out)
    if lowercase:
        out = out.lower()
    return out

def normalize_tokens(tokens: Iterable[str], *, lowercase: bool = True) -> List[str]:

    out: List[str] = []
    for tok in tokens:
        norm = normalize_token(tok, lowercase=lowercase)
        if norm:
            out.append(norm)
    return out

def normalize_text(text: str, *, lowercase: bool = True) -> str:
    out = nfc(text).strip()
    out = _WHITESPACE_RE.sub(" ", out)
    if lowercase:
        out = out.lower()
    return out

def map_raw_label(raw_label: str) -> str:

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
    return normalize_text(" ".join(tokens), lowercase=True)

def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")
