from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
from src.data.constants import LABELS, LABEL_TO_SYMBOL, SENTENCE_END_LABELS

__all__ = [
    "ReconstructionResult",
    "reconstruct_text",
    "reconstruct_with_details",
    "split_into_sentences",
]

@dataclass
class ReconstructionResult:
    text: str
    num_words: int
    num_commas: int
    num_periods: int
    num_questions: int
    num_sentences: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "num_words": self.num_words,
            "num_commas": self.num_commas,
            "num_periods": self.num_periods,
            "num_questions": self.num_questions,
            "num_sentences": self.num_sentences,
        }

def _capitalize(word: str) -> str:
    if not word:
        return word
    return word[0].upper() + word[1:]

def reconstruct_text(
    words: Sequence[str],
    labels: Sequence[str],
    *,
    capitalize: bool = True,
    ensure_final_punctuation: bool = False,
) -> str:
    if len(words) != len(labels):
        raise ValueError(
            f"len(words)={len(words)} != len(labels)={len(labels)}; "
            "every word must have exactly one label."
        )
    if not words:
        return ""

    unknown = sorted({l for l in labels if l not in LABELS})
    if unknown:
        raise ValueError(f"Unknown label(s) {unknown}; expected {LABELS}")

    pieces: List[str] = []
    start_of_sentence = True
    for word, label in zip(words, labels):
        rendered = _capitalize(word) if (capitalize and start_of_sentence) else word
        pieces.append(rendered + LABEL_TO_SYMBOL.get(label, ""))
        start_of_sentence = label in SENTENCE_END_LABELS

    text = " ".join(pieces)

    if ensure_final_punctuation and labels[-1] not in SENTENCE_END_LABELS:
        text = text.rstrip(",") + "."

    return text

def reconstruct_with_details(
    words: Sequence[str],
    labels: Sequence[str],
    *,
    capitalize: bool = True,
    ensure_final_punctuation: bool = False,
) -> ReconstructionResult:
    text = reconstruct_text(
        words,
        labels,
        capitalize=capitalize,
        ensure_final_punctuation=ensure_final_punctuation,
    )
    periods = sum(1 for l in labels if l == "PERIOD")
    questions = sum(1 for l in labels if l == "QUESTION")
    if ensure_final_punctuation and labels and labels[-1] not in SENTENCE_END_LABELS:
        periods += 1
    return ReconstructionResult(
        text=text,
        num_words=len(words),
        num_commas=sum(1 for l in labels if l == "COMMA"),
        num_periods=periods,
        num_questions=questions,
        num_sentences=periods + questions,
    )

def split_into_sentences(
    words: Sequence[str], labels: Sequence[str], *, capitalize: bool = True
) -> List[str]:
    if len(words) != len(labels):
        raise ValueError(f"len(words)={len(words)} != len(labels)={len(labels)}")

    sentences: List[str] = []
    buffer_words: List[str] = []
    buffer_labels: List[str] = []
    for word, label in zip(words, labels):
        buffer_words.append(word)
        buffer_labels.append(label)
        if label in SENTENCE_END_LABELS:
            sentences.append(reconstruct_text(buffer_words, buffer_labels, capitalize=capitalize))
            buffer_words, buffer_labels = [], []
    if buffer_words:
        sentences.append(reconstruct_text(buffer_words, buffer_labels, capitalize=capitalize))
    return sentences
