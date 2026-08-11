"""Turn free-form user text into the lexical words the model expects.

The models were trained on JointCapPunc, whose tokens are already clean
lowercase Vietnamese syllables with punctuation stripped out. Real user input
is not like that: it has capitals, stray punctuation, URLs, emails, decimal
numbers and inconsistent spacing. This module bridges the two.

What it protects
----------------
Splitting naively on non-letters would shred things that must stay whole:

* ``https://vnexpress.net/tin-tuc`` → one token, not eight
* ``bacsi@benhvien.vn`` → one token
* ``38.5`` / ``1,5`` / ``2.500.000`` → one numeric token, not "38" "." "5"
* ``covid-19``, ``x-quang`` → hyphenated words stay joined
* ``t/h``, ``bs.`` → kept readable

Everything else that is punctuation gets removed, because the model's job is to
put punctuation *back*; leaving the user's existing marks in would both confuse
the model and make the output ambiguous.

The result is always a plain list of lexical words. The predictor assigns
exactly one label per word, and the reconstructor re-inserts punctuation using
those labels — so the invariant ``len(words) == len(labels)`` is preserved
end to end.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

from src.data.normalization import normalize_token

__all__ = [
    "TokenizedInput",
    "InferenceTokenizer",
    "tokenize_for_inference",
]


_URL = r"(?:https?://|www\.)[^\s]+"
_EMAIL = r"[^\s@]+@[^\s@]+\.[^\s@,.;:!?)]+"

_NUMBER = r"\d+(?:[.,:/]\d+)*(?:%|°c|°|k|tr|tỷ)?"



_WORD = r"[^\W\d_]+\d*(?:[-_'’][^\W_]+)*"

_ALNUM = r"[^\W_]+(?:[-_][^\W_]+)*"

_TOKEN_RE = re.compile(
    "|".join(f"(?:{p})" for p in (_URL, _EMAIL, _NUMBER, _WORD, _ALNUM)),
    flags=re.UNICODE | re.IGNORECASE,
)


_STRIPPED_PUNCTUATION = ".,?!;:\"'“”‘’()[]{}…—–-"


@dataclass
class TokenizedInput:
    """Result of preparing user text for the model."""

    words: List[str]

    model_words: List[str] = field(default_factory=list)
    original_text: str = ""
    num_removed_punctuation: int = 0

    def __len__(self) -> int:
        return len(self.words)

    @property
    def is_empty(self) -> bool:
        return not self.words


class InferenceTokenizer:
    """Text → lexical words, preserving URLs / emails / numbers."""

    def __init__(self, *, lowercase: bool = True, keep_original_case: bool = True):
        self.lowercase = lowercase
        self.keep_original_case = keep_original_case

    def tokenize(self, text: str) -> TokenizedInput:
        if text is None:
            raise TypeError("text must be a string, got None")
        original = text
        normalized = unicodedata.normalize("NFC", text)

        raw_tokens = _TOKEN_RE.findall(normalized)

        words: List[str] = []
        model_words: List[str] = []
        for tok in raw_tokens:
            surface = tok.strip()
            if not surface:
                continue


            surface = surface.strip(_STRIPPED_PUNCTUATION)
            if not surface:
                continue
            model_form = normalize_token(surface, lowercase=self.lowercase)
            if not model_form:
                continue
            words.append(surface if self.keep_original_case else model_form)
            model_words.append(model_form)

        removed = sum(1 for ch in normalized if ch in _STRIPPED_PUNCTUATION)

        return TokenizedInput(
            words=words,
            model_words=model_words,
            original_text=original,
            num_removed_punctuation=removed,
        )

    def __call__(self, text: str) -> TokenizedInput:
        return self.tokenize(text)


_DEFAULT = InferenceTokenizer()


def tokenize_for_inference(text: str) -> TokenizedInput:
    """Tokenize with the default settings."""
    return _DEFAULT.tokenize(text)
