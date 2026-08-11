from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.selection import MODEL_SELECTION_PATH
from src.inference.service import (
    PunctuationService,
    ServiceResponse,
    get_service,
    reset_service,
)
from tests.test_predictor import make_bilstm_predictor

@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_service()
    yield
    reset_service()


def service_with_fake_predictor(**kwargs) -> PunctuationService:
    service = PunctuationService(warmup=False, **kwargs)
    service._predictor = make_bilstm_predictor()
    return service

def test_successful_response_shape():
    r = service_with_fake_predictor().restore("hôm nay trời đẹp")
    assert isinstance(r, ServiceResponse)
    assert r.ok is True
    assert r.restored_text
    assert r.num_words == 4
    assert r.error is None
    assert r.latency_ms >= 0
    json.dumps(r.to_dict())

@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None])
def test_empty_input_returns_a_friendly_error_not_an_exception(bad):
    r = service_with_fake_predictor().restore(bad)
    assert r.ok is False
    assert r.error_type == "empty_input"
    assert "nhập văn bản" in r.error.lower()

def test_input_that_tokenizes_to_nothing_is_handled():
    r = service_with_fake_predictor().restore("!!! ??? ...")
    assert r.ok is True
    assert r.restored_text == ""

def test_over_limit_input_is_reported_not_truncated():
    service = service_with_fake_predictor(max_words=50)
    r = service.restore(" ".join(["từ"] * 200))
    assert r.ok is False
    assert r.error_type == "input_too_long_or_invalid"
    assert "NOT truncated" in r.error

def test_missing_model_selection_is_reported_cleanly(tmp_path):
    service = PunctuationService(
        model_selection_path=tmp_path / "does_not_exist.json", warmup=False
    )
    r = service.restore("hôm nay trời đẹp")
    assert r.ok is False
    assert r.error and "not found" in r.error.lower()

def test_unlocked_selection_is_reported_cleanly(tmp_path):
    path = tmp_path / "model_selection.json"
    path.write_text(
        json.dumps({
            "selection_split": "validation",
            "selection_metric": "punctuation_macro_f1",
            "tie_breaker": "unweighted_validation_loss",
            "winner": "E2",
            "test_was_used_for_selection": False,
            "winner_locked": False,
        }),
        encoding="utf-8",
    )
    service = PunctuationService(model_selection_path=path, warmup=False)
    r = service.restore("hôm nay trời đẹp")
    assert r.ok is False
    assert "winner_locked" in (r.error or "")

def test_predictor_failure_becomes_a_response(monkeypatch):
    service = service_with_fake_predictor()

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(service._predictor, "restore", boom)
    r = service.restore("hôm nay trời đẹp")
    assert r.ok is False
    assert r.error_type == "out_of_memory"

def test_model_is_loaded_only_once():
    service = PunctuationService(warmup=False)
    calls = {"n": 0}
    fake = make_bilstm_predictor()

    def counting_loader(*a, **k):
        calls["n"] += 1
        return fake
    from src.inference import service as service_module

    original = service_module.PunctuationRestorationPredictor.from_selected_model
    service_module.PunctuationRestorationPredictor.from_selected_model = staticmethod(
        counting_loader
    )
    try:
        for _ in range(5):
            service.restore("hôm nay trời đẹp")
        assert calls["n"] == 1, "the checkpoint was reloaded on every request"
    finally:
        service_module.PunctuationRestorationPredictor.from_selected_model = original

def test_is_loaded_flag():
    service = service_with_fake_predictor()
    assert service.is_loaded is True
    assert PunctuationService(warmup=False).is_loaded is False

def test_get_service_returns_a_singleton():
    assert get_service() is get_service()
    reset_service()

    assert get_service() is get_service()

def test_restore_many():
    service = service_with_fake_predictor()
    responses = service.restore_many(["hôm nay trời đẹp", "", "bạn khỏe không"])
    assert [r.ok for r in responses] == [True, False, True]

@pytest.mark.integration
def test_real_service_end_to_end():
    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present — run notebook 05")

    service = PunctuationService(device=torch.device("cpu"), warmup=False)
    r = service.restore("hôm nay trời đẹp bạn có muốn đi dạo không")
    assert r.ok is True
    assert r.restored_text
    assert r.num_words == 10
    assert r.experiment_id

    info = service.model_info()
    assert info["winner_locked"] is True
    assert info["selected_on"] == "validation"
