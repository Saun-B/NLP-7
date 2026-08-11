from __future__ import annotations

import random
import pytest

from src.data.chunking import (
    assert_chunking_invariants,
    chunk_sequence,
    pack_sentences,
    split_into_sentences,
)
from src.data.constants import MAX_WORDS_PER_EXAMPLE

def make_stream(sentence_lengths, end_label="PERIOD"):
    tokens, labels = [], []
    counter = 0
    for n in sentence_lengths:
        for i in range(n):
            tokens.append(f"w{counter}")
            counter += 1
            labels.append(end_label if i == n - 1 else "O")
    return tokens, labels

def test_sentence_split_keeps_the_punctuated_token_inside_its_sentence():
    tokens, labels = make_stream([3, 2])
    sentences = split_into_sentences(tokens, labels)
    assert [len(t) for t, _ in sentences] == [3, 2]
    assert sentences[0][1] == ["O", "O", "PERIOD"]

def test_question_also_ends_a_sentence():
    tokens = ["a", "b", "c", "d"]
    labels = ["O", "QUESTION", "O", "PERIOD"]
    sentences = split_into_sentences(tokens, labels)
    assert [t for t, _ in sentences] == [["a", "b"], ["c", "d"]]

def test_comma_does_not_end_a_sentence():
    tokens = ["a", "b", "c"]
    labels = ["O", "COMMA", "O"]
    assert len(split_into_sentences(tokens, labels)) == 1

def test_unterminated_tail_is_kept():
    tokens, labels = make_stream([2])
    tokens += ["x", "y"]
    labels += ["O", "O"]
    sentences = split_into_sentences(tokens, labels)
    assert len(sentences) == 2
    assert sentences[1][0] == ["x", "y"]

def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="len\\(tokens\\)"):
        split_into_sentences(["a", "b"], ["O"])

def test_packing_respects_the_word_cap():
    sentences = [(["w"] * 40, ["O"] * 40) for _ in range(10)]
    segments = pack_sentences(sentences, max_words=150)
    assert all(len(t) <= 150 for t, _ in segments)
    assert sum(len(t) for t, _ in segments) == 400

def test_packing_never_splits_a_short_sentence():
    sentences = [(["a"] * 100, ["O"] * 100), (["b"] * 100, ["O"] * 100)]
    segments = pack_sentences(sentences, max_words=150)
    assert [len(t) for t, _ in segments] == [100, 100]

def test_oversized_sentence_becomes_its_own_segment():
    sentences = [
        (["a"] * 10, ["O"] * 10),
        (["b"] * 400, ["O"] * 400),
        (["c"] * 10, ["O"] * 10),
    ]
    segments = pack_sentences(sentences, max_words=150)
    assert [len(t) for t, _ in segments] == [10, 400, 10]

def test_no_token_is_lost_duplicated_or_reordered():
    tokens, labels = make_stream([17, 23, 5, 61, 44, 12, 90])
    chunks, _ = chunk_sequence(tokens, labels, max_words=150)
    assert_chunking_invariants(tokens, labels, chunks, max_words=150)

    rebuilt = [t for c in chunks for t in c.tokens]
    assert rebuilt == tokens
    assert len(rebuilt) == len(set(rebuilt))

def test_labels_stay_aligned_with_tokens():
    tokens, labels = make_stream([30, 40, 50])
    chunks, _ = chunk_sequence(tokens, labels, max_words=150)
    for chunk in chunks:
        assert len(chunk.tokens) == len(chunk.labels)
    assert [l for c in chunks for l in c.labels] == labels

def test_cap_is_never_exceeded():
    tokens, labels = make_stream([200, 7, 149, 151])
    chunks, _ = chunk_sequence(tokens, labels, max_words=150)
    assert max(len(c.tokens) for c in chunks) <= 150

def test_boundaries_are_preferred_over_hard_cuts():
    tokens, labels = make_stream([40] * 10)
    chunks, stats = chunk_sequence(tokens, labels, max_words=150)
    assert stats.hard_cut_chunks == 0
    assert all(c.labels[-1] == "PERIOD" for c in chunks)
    assert stats.chunks_ending_at_boundary == stats.chunks

def test_hard_cut_only_for_an_oversized_sentence():
    tokens, labels = make_stream([380])
    chunks, stats = chunk_sequence(tokens, labels, max_words=150)
    assert [len(c.tokens) for c in chunks] == [150, 150, 80]
    assert stats.oversized_sentences == 1
    assert stats.hard_cut_chunks == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert len({c.segment_index for c in chunks}) == 1
    assert_chunking_invariants(tokens, labels, chunks, max_words=150)

def test_chunk_index_is_zero_when_no_hard_cut_happened():
    tokens, labels = make_stream([50, 50, 50, 50])
    chunks, _ = chunk_sequence(tokens, labels, max_words=150)
    assert all(c.chunk_index == 0 for c in chunks)

def test_empty_input_produces_no_chunks():
    chunks, stats = chunk_sequence([], [], max_words=150)
    assert chunks == []
    assert stats.chunks == 0

def test_default_cap_is_150():
    tokens, labels = make_stream([1000])
    chunks, _ = chunk_sequence(tokens, labels)
    assert max(len(c.tokens) for c in chunks) <= MAX_WORDS_PER_EXAMPLE == 150

def test_invariant_checker_catches_a_dropped_token():
    tokens, labels = make_stream([10, 10])
    chunks, _ = chunk_sequence(tokens, labels, max_words=150)
    chunks[0].tokens.pop()
    chunks[0].labels.pop()
    with pytest.raises(AssertionError, match="token count changed"):
        assert_chunking_invariants(tokens, labels, chunks, max_words=150)

def test_invariant_checker_catches_reordering():
    tokens, labels = make_stream([4, 4])
    chunks, _ = chunk_sequence(tokens, labels, max_words=5)
    chunks[0].tokens.reverse()
    with pytest.raises(AssertionError, match="changed at position"):
        assert_chunking_invariants(tokens, labels, chunks, max_words=5)

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_randomised_streams_always_satisfy_the_invariants(seed):
    rng = random.Random(seed)
    lengths = [rng.randint(1, 320) for _ in range(rng.randint(1, 40))]
    tokens, labels = make_stream(lengths, end_label=rng.choice(["PERIOD", "QUESTION"]))

    for i in range(0, len(labels), 13):
        if labels[i] == "O":
            labels[i] = "COMMA"
    chunks, stats = chunk_sequence(tokens, labels, max_words=150)
    assert_chunking_invariants(tokens, labels, chunks, max_words=150)
    assert stats.input_tokens == len(tokens)
    assert sum(len(c.tokens) for c in chunks) == len(tokens)
