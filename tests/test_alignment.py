from __future__ import annotations

import pytest

from src.data.constants import IGNORE_INDEX, LABEL2ID, PHOBERT_MAX_LENGTH
from src.data.dataset import align_words_to_subwords

BOS, EOS = 0, 2


def align(word_pieces, labels, max_length=PHOBERT_MAX_LENGTH):
    label_ids = [LABEL2ID[l] for l in labels]
    return align_words_to_subwords(
        word_pieces, label_ids, bos_id=BOS, eos_id=EOS, max_length=max_length, example_id="t"
    )

def test_label_lands_on_the_last_subword_of_a_word():

    windows = align([[10], [11, 12, 13], [14, 15]], ["O", "COMMA", "PERIOD"])
    (w,) = windows
    assert w.input_ids == [BOS, 10, 11, 12, 13, 14, 15, EOS]
    assert w.labels == [
        IGNORE_INDEX,
        LABEL2ID["O"],
        IGNORE_INDEX,
        IGNORE_INDEX,
        LABEL2ID["COMMA"],
        IGNORE_INDEX,
        LABEL2ID["PERIOD"],
        IGNORE_INDEX,
    ]

def test_special_tokens_are_ignored():
    (w,) = align([[10], [11]], ["O", "PERIOD"])
    assert w.labels[0] == IGNORE_INDEX
    assert w.labels[-1] == IGNORE_INDEX
    assert w.input_ids[0] == BOS and w.input_ids[-1] == EOS

def test_every_word_supervised_exactly_once():
    pieces = [[i, i + 1] for i in range(0, 40, 2)]
    labels = ["O"] * 19 + ["PERIOD"]
    windows = align(pieces, labels)
    supervised = [l for w in windows for l in w.labels if l != IGNORE_INDEX]
    assert len(supervised) == len(pieces) == 20

def test_word_index_maps_positions_back_to_words():
    (w,) = align([[10], [11, 12], [13]], ["O", "QUESTION", "PERIOD"])
    assert w.word_index == [-1, 0, -1, 1, 2, -1]
    supervised = [i for i in w.word_index if i >= 0]
    assert supervised == [0, 1, 2]

def test_attention_mask_covers_every_real_position():
    (w,) = align([[10], [11, 12]], ["O", "PERIOD"])
    assert w.attention_mask == [1] * len(w.input_ids)

def test_long_input_is_split_not_truncated():
    n_words = 100
    pieces = [[i] for i in range(n_words)]
    labels = ["O"] * (n_words - 1) + ["PERIOD"]
    windows = align(pieces, labels, max_length=12)

    assert len(windows) == 10
    assert all(len(w.input_ids) <= 12 for w in windows)

    rebuilt = [i for w in windows for i in w.input_ids[1:-1]]
    assert rebuilt == list(range(n_words))

def test_windows_split_at_word_boundaries_only():

    pieces = [[i * 3, i * 3 + 1, i * 3 + 2] for i in range(7)]
    labels = ["O"] * 6 + ["PERIOD"]
    windows = align(pieces, labels, max_length=12)
    for w in windows:
        body = len(w.input_ids) - 2
        assert body % 3 == 0
    assert sum(w.num_words for w in windows) == 7

def test_word_indices_are_global_and_contiguous_across_windows():
    pieces = [[i] for i in range(25)]
    labels = ["O"] * 24 + ["PERIOD"]
    windows = align(pieces, labels, max_length=12)
    supervised = [i for w in windows for i in w.word_index if i >= 0]
    assert supervised == list(range(25))

def test_first_word_offset_is_recorded():
    pieces = [[i] for i in range(25)]
    labels = ["O"] * 24 + ["PERIOD"]
    windows = align(pieces, labels, max_length=12)
    offsets = [w.first_word for w in windows]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    running = 0
    for w in windows:
        assert w.first_word == running
        running += w.num_words

def test_exactly_full_window_is_not_split():
    pieces = [[i] for i in range(10)]
    labels = ["O"] * 9 + ["PERIOD"]
    windows = align(pieces, labels, max_length=12)
    assert len(windows) == 1
    assert len(windows[0].input_ids) == 12

def test_word_longer_than_the_window_is_truncated_but_never_dropped():
    pieces = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [20]]
    windows = align(pieces, ["COMMA", "PERIOD"], max_length=7)
    supervised = [i for w in windows for i in w.word_index if i >= 0]
    assert supervised == [0, 1]
    assert all(len(w.input_ids) <= 7 for w in windows)

def test_zero_piece_word_is_rejected():
    with pytest.raises(ValueError, match="zero subwords"):
        align([[10], []], ["O", "PERIOD"])

def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="!="):
        align_words_to_subwords([[1], [2]], [0], bos_id=BOS, eos_id=EOS)

def test_tiny_max_length_is_rejected():
    with pytest.raises(ValueError, match="max_length"):
        align([[1]], ["O"], max_length=2)

def test_single_word_input():
    (w,) = align([[7]], ["PERIOD"])
    assert w.num_supervised == 1
    assert w.labels == [IGNORE_INDEX, LABEL2ID["PERIOD"], IGNORE_INDEX]

def _local_phobert_tokenizer():
    transformers = pytest.importorskip("transformers")
    from src.data.constants import CHECKPOINT_DIR, EXPERIMENT_IDS

    for experiment_id in EXPERIMENT_IDS:
        directory = CHECKPOINT_DIR / experiment_id
        if (directory / "tokenizer_config.json").exists():
            return transformers.AutoTokenizer.from_pretrained(
                directory, local_files_only=True
            )
    pytest.skip("no local PhoBERT checkpoint with a saved tokenizer")

@pytest.mark.slow
def test_real_phobert_alignment_round_trip():
    from src.data.dataset import PhoBERTEncoder
    from src.data.schema import Example

    tokenizer = _local_phobert_tokenizer()

    tokens = "bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài".split()
    labels = ["O"] * 6 + ["QUESTION"] + ["O"] * 5 + ["PERIOD"]
    ex = Example(
        id="t_1", source_split="train", source_id="t_seg_1", chunk_index=0,
        tokens=tokens, labels=labels,
    )

    encoder = PhoBERTEncoder(tokenizer, max_length=PHOBERT_MAX_LENGTH)
    windows = encoder.encode_example(ex)

    supervised = [i for w in windows for i in w.word_index if i >= 0]
    assert supervised == list(range(len(tokens)))

    gold = [l for w in windows for l in w.labels if l != IGNORE_INDEX]
    assert gold == [LABEL2ID[l] for l in labels]
