from __future__ import annotations

import numpy as np
import pytest

from src.data.constants import IGNORE_INDEX, LABEL2ID, PUNCTUATION_LABELS
from src.evaluation.metrics import (
    compute_metrics,
    confusion_matrix,
    flatten_metrics,
    format_metrics_table,
    metrics_from_confusion_matrix,
    per_class_metrics,
    punctuation_macro_f1,
)

O, COMMA, PERIOD, QUESTION = 0, 1, 2, 3

def test_perfect_prediction():
    y = [O, COMMA, PERIOD, QUESTION, O]
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["punctuation_macro_f1"] == 1.0

def test_confusion_matrix_orientation_is_gold_by_predicted():
    cm = confusion_matrix([O, O, COMMA], [O, COMMA, COMMA])
    assert cm[O, O] == 1
    assert cm[O, COMMA] == 1
    assert cm[COMMA, COMMA] == 1
    assert cm.sum() == 3

def test_ignore_index_positions_are_excluded():
    y_true = [O, IGNORE_INDEX, PERIOD, IGNORE_INDEX]
    y_pred = [O, COMMA, PERIOD, QUESTION]
    m = compute_metrics(y_true, y_pred)
    assert m["num_evaluated_tokens"] == 2
    assert m["accuracy"] == 1.0

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics([O, O], [O])

def test_out_of_range_label_raises():
    with pytest.raises(ValueError, match="out of range"):
        confusion_matrix([O, 9], [O, O])

def test_per_class_values_match_hand_computation():

    y_true = [O, O, O, COMMA, COMMA, PERIOD]
    y_pred = [O, O, COMMA, COMMA, O, PERIOD]
    pc = per_class_metrics(confusion_matrix(y_true, y_pred))

    assert pc["O"]["precision"] == pytest.approx(2 / 3)
    assert pc["O"]["recall"] == pytest.approx(2 / 3)
    assert pc["O"]["f1"] == pytest.approx(2 / 3)

    assert pc["COMMA"]["precision"] == pytest.approx(0.5)
    assert pc["COMMA"]["recall"] == pytest.approx(0.5)
    assert pc["COMMA"]["f1"] == pytest.approx(0.5)
    assert pc["PERIOD"]["f1"] == pytest.approx(1.0)

def test_absent_class_scores_zero_without_dividing_by_zero():
    y_true = [O, O, COMMA]
    y_pred = [O, O, COMMA]
    pc = per_class_metrics(confusion_matrix(y_true, y_pred))
    assert pc["QUESTION"]["f1"] == 0.0
    assert pc["QUESTION"]["support"] == 0.0

def test_support_equals_gold_count():
    y_true = [O, O, O, COMMA, PERIOD, PERIOD]
    y_pred = [O] * 6
    pc = per_class_metrics(confusion_matrix(y_true, y_pred))
    assert pc["O"]["support"] == 3
    assert pc["COMMA"]["support"] == 1
    assert pc["PERIOD"]["support"] == 2

def test_punctuation_macro_f1_excludes_O():

    y_true = [O] * 90 + [COMMA] * 5 + [PERIOD] * 4 + [QUESTION]
    y_pred = [O] * 100
    m = compute_metrics(y_true, y_pred)

    assert m["accuracy"] == pytest.approx(0.90)
    assert m["punctuation_macro_f1"] == 0.0
    assert m["macro_f1"] > 0.0
    assert m["macro_f1"] == pytest.approx(m["per_class"]["O"]["f1"] / 4)

def test_punctuation_macro_f1_is_the_mean_of_three_class_f1s():
    y_true = [O, O, COMMA, COMMA, PERIOD, PERIOD, QUESTION, QUESTION]
    y_pred = [O, COMMA, COMMA, O, PERIOD, PERIOD, QUESTION, O]
    m = compute_metrics(y_true, y_pred)
    expected = np.mean([m["per_class"][c]["f1"] for c in PUNCTUATION_LABELS])
    assert m["punctuation_macro_f1"] == pytest.approx(expected)
    assert PUNCTUATION_LABELS == ["COMMA", "PERIOD", "QUESTION"]

def test_helper_matches_bundle():
    y_true = [O, COMMA, PERIOD, QUESTION] * 5
    y_pred = [O, COMMA, PERIOD, O] * 5
    m = compute_metrics(y_true, y_pred)
    assert punctuation_macro_f1(m["per_class"]) == pytest.approx(m["punctuation_macro_f1"])

def test_rare_class_is_not_drowned_out():
    y_true = [COMMA] * 50 + [PERIOD] * 49 + [QUESTION]
    good = list(y_true)
    bad = [COMMA] * 50 + [PERIOD] * 49 + [PERIOD]

    assert compute_metrics(y_true, good)["punctuation_macro_f1"] == pytest.approx(1.0)

    m_bad = compute_metrics(y_true, bad)
    assert m_bad["accuracy"] == pytest.approx(0.99)
    assert m_bad["per_class"]["QUESTION"]["f1"] == 0.0

    assert m_bad["punctuation_macro_f1"] < 0.70
    assert m_bad["punctuation_micro_f1"] > 0.98

def test_metrics_from_confusion_matrix_matches_compute_metrics():
    y_true = [O, O, COMMA, PERIOD, QUESTION, COMMA]
    y_pred = [O, COMMA, COMMA, PERIOD, O, QUESTION]
    direct = compute_metrics(y_true, y_pred)
    via_cm = metrics_from_confusion_matrix(confusion_matrix(y_true, y_pred))
    assert via_cm["punctuation_macro_f1"] == pytest.approx(direct["punctuation_macro_f1"])
    assert via_cm["confusion_matrix"] == direct["confusion_matrix"]

def test_loss_is_carried_through():
    m = compute_metrics([O], [O], loss=0.1234)
    assert m["loss"] == pytest.approx(0.1234)

def test_flatten_produces_scalar_history_columns():
    m = compute_metrics([O, COMMA, PERIOD, QUESTION], [O, COMMA, PERIOD, QUESTION], loss=0.5)
    flat = flatten_metrics(m, prefix="val_")
    assert flat["val_punctuation_macro_f1"] == 1.0
    assert flat["val_loss"] == 0.5
    for label in LABEL2ID:
        assert f"val_f1_{label}" in flat
    assert all(isinstance(v, float) for v in flat.values())

def test_confusion_matrix_is_json_serialisable():
    m = compute_metrics([O, COMMA], [O, COMMA])
    assert isinstance(m["confusion_matrix"], list)
    assert all(isinstance(x, int) for row in m["confusion_matrix"] for x in row)

def test_table_renders_and_names_the_selection_metric():
    m = compute_metrics([O, COMMA, PERIOD, QUESTION], [O, COMMA, PERIOD, O])
    table = format_metrics_table(m)
    assert "PUNCT-F1" in table
    assert "model selection metric" in table
    for label in ("O", "COMMA", "PERIOD", "QUESTION"):
        assert label in table

def test_empty_evaluation_does_not_crash():
    m = compute_metrics([IGNORE_INDEX], [O])
    assert m["num_evaluated_tokens"] == 0
    assert m["accuracy"] == 0.0
    assert m["punctuation_macro_f1"] == 0.0
