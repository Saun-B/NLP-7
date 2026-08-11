from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.constants import LABELS
from src.evaluation.selection import (
    SELECTION_METRIC,
    SELECTION_SPLIT,
    TIE_BREAKER,
    SelectionError,
    ValidationCandidate,
    load_locked_winner,
    select_winner,
    write_model_selection,
)

def candidate(exp_id: str, punct_f1: float, loss: float, **kw) -> ValidationCandidate:
    return ValidationCandidate(
        experiment_id=exp_id,
        model=kw.get("model", "test-model"),
        model_type=kw.get("model_type", "phobert"),
        weight_mode=kw.get("weight_mode", "none"),
        best_epoch=kw.get("best_epoch", 1),
        punctuation_macro_f1=punct_f1,
        unweighted_validation_loss=loss,
        accuracy=kw.get("accuracy", 0.9),
        macro_f1=kw.get("macro_f1", 0.8),
        f1_per_class={c: punct_f1 for c in LABELS},
    )

def test_highest_punctuation_macro_f1_wins():
    result = select_winner([
        candidate("EA", 0.70, 0.10),
        candidate("EB", 0.78, 0.30),
        candidate("EC", 0.65, 0.05),
    ])
    assert result.winner.experiment_id == "EB"
    assert [c.experiment_id for c in result.ranking] == ["EB", "EA", "EC"]

def test_accuracy_never_decides():

    result = select_winner([
        candidate("EA", 0.60, 0.10, accuracy=0.99),
        candidate("EB", 0.75, 0.10, accuracy=0.80),
    ])
    assert result.winner.experiment_id == "EB"

def test_exact_tie_is_broken_by_lower_unweighted_loss():
    result = select_winner([
        candidate("EA", 0.750000, 0.200),
        candidate("EB", 0.750000, 0.100),
    ])
    assert result.winner.experiment_id == "EB"
    assert result.tie_breaker_used is True
    assert result.margin_over_runner_up == pytest.approx(0.0)

def test_tie_breaker_not_flagged_when_there_is_a_real_margin():
    result = select_winner([candidate("EA", 0.80, 0.5), candidate("EB", 0.70, 0.1)])
    assert result.tie_breaker_used is False
    assert result.margin_over_runner_up == pytest.approx(0.10)

def test_selection_is_deterministic_regardless_of_input_order():
    a, b, c = candidate("EA", 0.7, 0.2), candidate("EB", 0.7, 0.2), candidate("EC", 0.9, 0.9)
    first = select_winner([a, b, c])
    second = select_winner([c, b, a])
    assert first.winner.experiment_id == second.winner.experiment_id == "EC"
    assert [x.experiment_id for x in first.ranking] == [x.experiment_id for x in second.ranking]

def test_single_candidate_has_no_runner_up():
    result = select_winner([candidate("EA", 0.5, 0.1)])
    assert result.winner.experiment_id == "EA"
    assert result.runner_up is None
    assert result.margin_over_runner_up is None

def test_empty_candidate_list_raises():
    with pytest.raises(SelectionError, match="No candidates"):
        select_winner([])

def test_candidate_has_no_test_fields():
    fields = set(ValidationCandidate.__dataclass_fields__)
    assert not [f for f in fields if "test" in f.lower()], (
        f"ValidationCandidate must not expose any test-related field, found: {fields}"
    )

def test_candidate_rejects_a_test_metric_field():
    with pytest.raises(TypeError):
        ValidationCandidate(
            experiment_id="EA",
            model="m",
            model_type="phobert",
            weight_mode="none",
            best_epoch=1,
            punctuation_macro_f1=0.9,
            unweighted_validation_loss=0.1,
            accuracy=0.9,
            macro_f1=0.9,
            test_punctuation_macro_f1=0.99,
        )

def test_select_winner_rejects_plain_dicts_carrying_test_scores():
    payload = {
        "experiment_id": "EA",
        "punctuation_macro_f1": 0.5,
        "test_punctuation_macro_f1": 0.99,
    }
    with pytest.raises(TypeError, match="ValidationCandidate"):
        select_winner([payload])

def test_candidate_is_immutable():
    c = candidate("EA", 0.5, 0.1)
    with pytest.raises(Exception):
        c.punctuation_macro_f1 = 0.99

def test_written_selection_has_the_required_schema(tmp_path):
    result = select_winner([candidate("EA", 0.8, 0.1), candidate("EB", 0.7, 0.2)])
    path = write_model_selection(result, path=tmp_path / "model_selection.json")
    blob = json.loads(Path(path).read_text(encoding="utf-8"))

    assert blob["selection_split"] == SELECTION_SPLIT == "validation"
    assert blob["selection_metric"] == SELECTION_METRIC == "punctuation_macro_f1"
    assert blob["tie_breaker"] == TIE_BREAKER == "unweighted_validation_loss"
    assert blob["winner"] == "EA"
    assert blob["test_was_used_for_selection"] is False
    assert blob["winner_locked"] is True

def test_load_locked_winner_roundtrip(tmp_path):
    result = select_winner([candidate("EA", 0.8, 0.1)])
    path = write_model_selection(result, path=tmp_path / "sel.json")
    assert load_locked_winner(path)["winner"] == "EA"

def test_load_locked_winner_rejects_unlocked(tmp_path):
    path = tmp_path / "sel.json"
    path.write_text(json.dumps({
        "selection_split": "validation", "selection_metric": "punctuation_macro_f1",
        "tie_breaker": "unweighted_validation_loss", "winner": "EA",
        "test_was_used_for_selection": False, "winner_locked": False,
    }), encoding="utf-8")
    with pytest.raises(SelectionError, match="winner_locked is not true"):
        load_locked_winner(path)

def test_load_locked_winner_rejects_test_based_selection(tmp_path):
    path = tmp_path / "sel.json"
    path.write_text(json.dumps({
        "selection_split": "test", "selection_metric": "punctuation_macro_f1",
        "tie_breaker": "unweighted_validation_loss", "winner": "EA",
        "test_was_used_for_selection": True, "winner_locked": True,
    }), encoding="utf-8")
    with pytest.raises(SelectionError, match="test_was_used_for_selection"):
        load_locked_winner(path)

def test_load_locked_winner_rejects_missing_file(tmp_path):
    with pytest.raises(SelectionError, match="not found"):
        load_locked_winner(tmp_path / "nope.json")

@pytest.mark.integration
def test_repository_selection_artifact_is_valid_and_locked():
    from src.evaluation.selection import MODEL_SELECTION_PATH

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present — run notebook 05 first")

    blob = load_locked_winner()
    assert blob["winner_locked"] is True
    assert blob["test_was_used_for_selection"] is False
    assert blob["selection_split"] == "validation"
    assert blob["selection_metric"] == "punctuation_macro_f1"

    assert blob["ranking"][0]["experiment_id"] == blob["winner"]
