from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from src.data.constants import ID2LABEL, LABEL2ID, LABELS
from src.evaluation.selection import MODEL_SELECTION_PATH
from src.inference.predictor import PunctuationRestorationPredictor

class FakeVocabulary:
    """Minimal stand-in for the BiLSTM vocabulary."""

    def encode(self, tokens):
        return [min(len(t), 9) for t in tokens]

    def __len__(self):
        return 10

class FakeBiLSTM(torch.nn.Module):
    """Predicts PERIOD on the last word of every chunk, O elsewhere."""

    def forward(self, input_ids, attention_mask=None, lengths=None, **_):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, len(LABELS))
        logits[:, :, LABEL2ID["O"]] = 1.0
        if lengths is not None:
            for i, n in enumerate(lengths.tolist()):
                logits[i, n - 1, LABEL2ID["O"]] = 0.0
                logits[i, n - 1, LABEL2ID["PERIOD"]] = 5.0
        return logits

def make_bilstm_predictor(**kwargs) -> PunctuationRestorationPredictor:
    return PunctuationRestorationPredictor(
        model=FakeBiLSTM(),
        model_type="bilstm",
        encoder=FakeVocabulary(),
        metadata={
            "experiment_id": "EFAKE",
            "model_name": "fake-bilstm",
            "model_type": "bilstm",
            "label2id": dict(LABEL2ID),
            "best_epoch": 1,
        },
        device=torch.device("cpu"),
        **kwargs,
    )

def test_one_label_per_word():
    p = make_bilstm_predictor()
    words = "một hai ba bốn năm".split()
    labels = p.predict_labels(words)
    assert len(labels) == len(words)
    assert all(l in LABELS for l in labels)

def test_empty_input_is_handled():
    p = make_bilstm_predictor()
    assert p.predict_labels([]) == []
    result = p.restore("")
    assert result.restored_text == ""
    assert result.num_words == 0

def test_whitespace_only_input_is_handled():
    p = make_bilstm_predictor()
    assert p.restore("   \n\t  ").restored_text == ""

def test_restore_returns_a_populated_result():
    p = make_bilstm_predictor()
    r = p.restore("hôm nay trời đẹp")
    assert r.experiment_id == "EFAKE"
    assert r.model_name == "fake-bilstm"
    assert r.model_type == "bilstm"
    assert r.num_words == 4
    assert r.latency_ms >= 0
    assert r.restored_text.endswith(".")
    assert r.restored_text[0].isupper()

def test_result_serialises_to_a_stable_shape():
    p = make_bilstm_predictor()
    d = p.restore("hôm nay trời đẹp").to_dict()
    for key in ("input_text", "restored_text", "num_words", "latency_ms",
                "experiment_id", "model_name"):
        assert key in d

def test_mismatched_label_map_is_rejected():
    with pytest.raises(ValueError, match="label mapping"):
        PunctuationRestorationPredictor(
            model=FakeBiLSTM(),
            model_type="bilstm",
            encoder=FakeVocabulary(),
            metadata={"label2id": {"O": 0, "COMMA": 1, "PERIOD": 2, "EXCLAM": 3}},
            device=torch.device("cpu"),
        )

def test_describe_reports_the_model_card():
    p = make_bilstm_predictor()
    info = p.describe()
    assert info["experiment_id"] == "EFAKE"
    assert info["model_type"] == "bilstm"
    assert info["device"] == "cpu"

def test_punctuation_in_the_input_does_not_duplicate_in_the_output():
    p = make_bilstm_predictor()
    r = p.restore("Hôm nay, trời đẹp. Bạn khỏe không?")
    assert ",," not in r.restored_text
    assert ".." not in r.restored_text
    assert "?." not in r.restored_text

def test_diacritics_preserved_through_the_predictor():
    p = make_bilstm_predictor()
    r = p.restore("tiếng việt rất đẹp")
    assert "tiếng" in r.restored_text.lower()
    assert "tieng" not in r.restored_text.lower()

def test_warmup_runs_and_returns_a_duration():
    p = make_bilstm_predictor()
    assert p.warmup() >= 0

def test_restore_batch():
    p = make_bilstm_predictor()
    results = p.restore_batch(["hôm nay", "trời đẹp quá"])
    assert len(results) == 2
    assert all(r.restored_text for r in results)

@pytest.mark.integration
def test_real_winner_restores_the_spec_example():
    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present — run notebook 05")

    predictor = PunctuationRestorationPredictor.from_selected_model(
        device=torch.device("cpu")
    )
    text = "bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài"
    result = predictor.restore(text)

    assert result.num_words == 13
    assert len(result.labels) == 13

    stripped = (
        result.restored_text.lower().replace(",", "").replace(".", "").replace("?", "")
    )
    assert stripped.split() == text.split()

    assert any(c in result.restored_text for c in ".,?")

@pytest.mark.integration
def test_real_winner_matches_the_locked_selection():
    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present")
    from src.utils.io import read_json

    selection = read_json(MODEL_SELECTION_PATH)
    predictor = PunctuationRestorationPredictor.from_selected_model(
        device=torch.device("cpu")
    )
    assert predictor.experiment_id == selection["winner"]
    assert predictor.describe()["winner_locked"] is True

@pytest.mark.integration
def test_real_winner_is_deterministic():
    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present")
    predictor = PunctuationRestorationPredictor.from_selected_model(
        device=torch.device("cpu")
    )
    text = "chào bác sĩ em bị đau bụng mấy hôm nay rồi em có nên đi khám không"
    first = predictor.restore(text).restored_text
    second = predictor.restore(text).restored_text
    assert first == second
