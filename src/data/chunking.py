from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from src.data.constants import MAX_WORDS_PER_EXAMPLE, SENTENCE_END_LABELS

__all__ = [
    "Chunk",
    "ChunkingStats",
    "split_into_sentences",
    "pack_sentences",
    "chunk_sequence",
    "assert_chunking_invariants",
]

@dataclass
class Chunk:

    tokens: List[str]
    labels: List[str]
    segment_index: int
    chunk_index: int
    hard_cut: bool = False
    ends_with_sentence_boundary: bool = False

    def __len__(self) -> int:
        return len(self.tokens)

@dataclass
class ChunkingStats:

    input_tokens: int = 0
    sentences: int = 0
    segments: int = 0
    chunks: int = 0
    hard_cut_chunks: int = 0
    oversized_sentences: int = 0
    chunks_ending_at_boundary: int = 0

    def to_dict(self) -> dict:
        boundary_ratio = (
            self.chunks_ending_at_boundary / self.chunks if self.chunks else 0.0
        )
        return {
            "input_tokens": self.input_tokens,
            "sentences": self.sentences,
            "segments": self.segments,
            "chunks": self.chunks,
            "hard_cut_chunks": self.hard_cut_chunks,
            "oversized_sentences": self.oversized_sentences,
            "chunks_ending_at_sentence_boundary": self.chunks_ending_at_boundary,
            "boundary_ratio": round(boundary_ratio, 6),
        }

def split_into_sentences(
    tokens: Sequence[str],
    labels: Sequence[str],
    *,
    end_labels: Iterable[str] = tuple(SENTENCE_END_LABELS),
) -> List[Tuple[List[str], List[str]]]:
    if len(tokens) != len(labels):
        raise ValueError(
            f"len(tokens)={len(tokens)} != len(labels)={len(labels)}"
        )

    ends = set(end_labels)
    sentences: List[Tuple[List[str], List[str]]] = []
    start = 0
    for i, label in enumerate(labels):
        if label in ends:
            sentences.append((list(tokens[start : i + 1]), list(labels[start : i + 1])))
            start = i + 1
    if start < len(tokens):
        sentences.append((list(tokens[start:]), list(labels[start:])))
    return sentences

def pack_sentences(
    sentences: Sequence[Tuple[List[str], List[str]]],
    *,
    max_words: int = MAX_WORDS_PER_EXAMPLE,
) -> List[Tuple[List[str], List[str]]]:

    if max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words}")

    segments: List[Tuple[List[str], List[str]]] = []
    cur_tokens: List[str] = []
    cur_labels: List[str] = []

    def flush() -> None:
        nonlocal cur_tokens, cur_labels
        if cur_tokens:
            segments.append((cur_tokens, cur_labels))
            cur_tokens, cur_labels = [], []

    for s_tokens, s_labels in sentences:
        if not s_tokens:
            continue
        if len(s_tokens) > max_words:

            flush()
            segments.append((list(s_tokens), list(s_labels)))
            continue
        if len(cur_tokens) + len(s_tokens) > max_words:
            flush()
        cur_tokens.extend(s_tokens)
        cur_labels.extend(s_labels)

    flush()
    return segments

def chunk_sequence(
    tokens: Sequence[str],
    labels: Sequence[str],
    *,
    max_words: int = MAX_WORDS_PER_EXAMPLE,
    end_labels: Iterable[str] = tuple(SENTENCE_END_LABELS),
) -> Tuple[List[Chunk], ChunkingStats]:
    """Full chunking pipeline for one document / token stream."""
    if len(tokens) != len(labels):
        raise ValueError(f"len(tokens)={len(tokens)} != len(labels)={len(labels)}")

    stats = ChunkingStats(input_tokens=len(tokens))
    if not tokens:
        return [], stats

    ends = set(end_labels)
    sentences = split_into_sentences(tokens, labels, end_labels=ends)
    stats.sentences = len(sentences)
    stats.oversized_sentences = sum(1 for t, _ in sentences if len(t) > max_words)

    segments = pack_sentences(sentences, max_words=max_words)
    stats.segments = len(segments)

    chunks: List[Chunk] = []
    for seg_idx, (seg_tokens, seg_labels) in enumerate(segments):
        if len(seg_tokens) <= max_words:
            pieces = [(seg_tokens, seg_labels, False)]
        else:
            pieces = [
                (seg_tokens[i : i + max_words], seg_labels[i : i + max_words], True)
                for i in range(0, len(seg_tokens), max_words)
            ]
        for chunk_idx, (c_tokens, c_labels, hard) in enumerate(pieces):
            at_boundary = bool(c_labels) and c_labels[-1] in ends
            chunks.append(
                Chunk(
                    tokens=list(c_tokens),
                    labels=list(c_labels),
                    segment_index=seg_idx,
                    chunk_index=chunk_idx,
                    hard_cut=hard,
                    ends_with_sentence_boundary=at_boundary,
                )
            )
            if hard:
                stats.hard_cut_chunks += 1
            if at_boundary:
                stats.chunks_ending_at_boundary += 1

    stats.chunks = len(chunks)
    return chunks, stats

def assert_chunking_invariants(
    tokens: Sequence[str],
    labels: Sequence[str],
    chunks: Sequence[Chunk],
    *,
    max_words: int = MAX_WORDS_PER_EXAMPLE,
) -> None:
    """Raise ``AssertionError`` if chunking lost, duplicated or reordered data."""
    rebuilt_tokens: List[str] = []
    rebuilt_labels: List[str] = []
    for i, chunk in enumerate(chunks):
        if len(chunk.tokens) != len(chunk.labels):
            raise AssertionError(
                f"chunk[{i}]: len(tokens)={len(chunk.tokens)} != len(labels)={len(chunk.labels)}"
            )
        if not chunk.tokens:
            raise AssertionError(f"chunk[{i}] is empty")
        if len(chunk.tokens) > max_words:
            raise AssertionError(
                f"chunk[{i}] has {len(chunk.tokens)} tokens (> max_words={max_words})"
            )
        rebuilt_tokens.extend(chunk.tokens)
        rebuilt_labels.extend(chunk.labels)

    if len(rebuilt_tokens) != len(tokens):
        raise AssertionError(
            f"token count changed: input={len(tokens)} output={len(rebuilt_tokens)}"
        )
    if rebuilt_tokens != list(tokens):
        first = next(
            (i for i, (a, b) in enumerate(zip(rebuilt_tokens, tokens)) if a != b), None
        )
        raise AssertionError(f"token stream changed at position {first}")
    if rebuilt_labels != list(labels):
        first = next(
            (i for i, (a, b) in enumerate(zip(rebuilt_labels, labels)) if a != b), None
        )
        raise AssertionError(f"label stream changed at position {first}")
