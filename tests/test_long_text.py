from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.constants import IGNORE_INDEX, LABEL2ID, LABELS, PHOBERT_MAX_LENGTH
from src.data.dataset import align_words_to_subwords
from src.evaluation.selection import MODEL_SELECTION_PATH
from src.inference.predictor import PunctuationRestorationPredictor
from tests.test_predictor import FakeBiLSTM, FakeVocabulary, make_bilstm_predictor

def long_words(n: int) -> list[str]:
    base = "hôm nay trời đẹp em bị đau bụng bác sĩ ơi cho em hỏi một chút".split()
    return [base[i % len(base)] for i in range(n)]

@pytest.mark.parametrize("n_words", [1, 50, 191, 192, 193, 500, 2000])
def test_alignment_covers_every_word_regardless_of_length(n_words):
    pieces = [[i % 1000 + 5] for i in range(n_words)]
    labels = [LABEL2ID["O"]] * n_words
    windows = align_words_to_subwords(
        pieces, labels, bos_id=0, eos_id=2, max_length=PHOBERT_MAX_LENGTH
    )
    supervised = [w for win in windows for w in win.word_index if w >= 0]
    assert supervised == list(range(n_words)), "some words were dropped or reordered"
    assert all(len(w.input_ids) <= PHOBERT_MAX_LENGTH for w in windows)

def test_long_input_produces_multiple_windows():
    n = 1000
    pieces = [[i % 500 + 5] for i in range(n)]
    windows = align_words_to_subwords(
        pieces, [LABEL2ID["O"]] * n, bos_id=0, eos_id=2, max_length=PHOBERT_MAX_LENGTH
    )
    assert len(windows) > 1
    assert sum(w.num_words for w in windows) == n

def test_no_word_is_split_across_two_windows():

    pieces = [[i * 3, i * 3 + 1, i * 3 + 2] for i in range(40)]
    windows = align_words_to_subwords(
        pieces, [LABEL2ID["O"]] * 40, bos_id=0, eos_id=2, max_length=12
    )
    for w in windows:
        assert (len(w.input_ids) - 2) % 3 == 0
    assert sum(w.num_words for w in windows) == 40

@pytest.mark.parametrize("n_words", [1, 149, 150, 151, 300, 1500])
def test_predictor_returns_one_label_per_word_for_any_length(n_words):
    p = make_bilstm_predictor()
    words = long_words(n_words)
    labels = p.predict_labels(words)
    assert len(labels) == n_words
    assert all(l in LABELS for l in labels)

def test_bilstm_chunking_boundary_is_the_training_length():
    p = make_bilstm_predictor()
    r = p.restore(" ".join(long_words(451)))
    assert r.num_words == 451
    assert r.num_windows == 4

def test_restore_preserves_every_word_of_a_long_text():
    p = make_bilstm_predictor()
    words = long_words(600)
    r = p.restore(" ".join(words))
    assert r.num_words == 600
    out_words = (
        r.restored_text.lower().replace(",", "").replace(".", "").replace("?", "").split()
    )
    assert out_words == words

def test_over_limit_input_is_refused_not_truncated():
    p = make_bilstm_predictor()
    text = " ".join(long_words(1200))
    with pytest.raises(ValueError, match="NOT truncated"):
        p.restore(text, max_words=1000)

def test_under_limit_input_passes():
    p = make_bilstm_predictor()
    r = p.restore(" ".join(long_words(100)), max_words=1000)
    assert r.num_words == 100

@pytest.mark.integration
@pytest.mark.parametrize("n_words", [200, 400])
def test_real_winner_handles_text_longer_than_one_window(n_words):
    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present — run notebook 05")

    predictor = PunctuationRestorationPredictor.from_selected_model(
        device=torch.device("cpu")
    )
    words = long_words(n_words)
    labels = predictor.predict_labels(words)
    assert len(labels) == n_words, "long input lost words — silent truncation"

    result = predictor.restore(" ".join(words))
    assert result.num_words == n_words
    if predictor.model_type == "phobert":
        assert result.num_windows >= 2, "expected the text to span several windows"
