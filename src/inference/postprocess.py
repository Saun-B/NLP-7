"""Compatibility post-processing helpers for final deliverables.

The project already keeps the real reconstruction logic in
``src.inference.reconstruction``. This module adds a small text cleanup layer for raw strings that already
contain punctuation separated by spaces.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.inference.reconstruction import reconstruct_text, reconstruct_with_details

__all__ = ["postprocess_text", "restore_from_words"]


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.?])")
_SPACE_AFTER_PUNCT = re.compile(r"([,.?])(?=\S)")
_DUPLICATE_PUNCT = re.compile(r"([,.?])(?:\s*\1)+")


def _capitalize_sentences(text: str) -> str:
    chars = list(text)
    at_sentence_start = True
    for i, ch in enumerate(chars):
        if ch.isspace():
            continue
        if at_sentence_start and ch.isalpha():
            chars[i] = ch.upper()
            at_sentence_start = False
            continue
        at_sentence_start = ch in ".?"
    return "".join(chars)


def postprocess_text(text: str, *, capitalize: bool = True) -> str:
    """Clean a raw punctuated string.

    Examples
    --------
    ``hom nay troi dep . ban di dau ?`` becomes
    ``Hom nay troi dep. Ban di dau?``.
    """
    cleaned = " ".join((text or "").split())
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _SPACE_AFTER_PUNCT.sub(r"\1 ", cleaned)
    cleaned = _DUPLICATE_PUNCT.sub(r"\1", cleaned)
    cleaned = " ".join(cleaned.split())
    return _capitalize_sentences(cleaned) if capitalize else cleaned


def restore_from_words(
    words: Sequence[str],
    labels: Sequence[str],
    *,
    capitalize: bool = True,
    ensure_final_punctuation: bool = True,
) -> str:
    """Insert punctuation labels after words and run final cleanup."""
    raw = reconstruct_text(
        words,
        labels,
        capitalize=capitalize,
        ensure_final_punctuation=ensure_final_punctuation,
    )
    return postprocess_text(raw, capitalize=False)
