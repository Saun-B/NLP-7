from __future__ import annotations

import pytest

from src.data.constants import LABEL2ID, MAX_WORDS_PER_EXAMPLE, PROJECT_ROOT
from src.data.schema import (
    SCHEMA_FIELDS,
    Example,
    SchemaError,
    make_example_id,
    make_source_id,
    validate_row,
)
from src.utils.io import project_relative_path

def valid_row(**overrides):
    row = {
        "id": "train_000001",
        "source_split": "train",
        "source_id": "train_seg_000001",
        "chunk_index": 0,
        "tokens": ["hôm", "nay", "trời", "đẹp"],
        "labels": ["O", "O", "O", "PERIOD"],
    }
    row.update(overrides)
    return row

def test_label_space_is_exactly_the_four_project_labels():
    assert LABEL2ID == {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}

def test_valid_row_passes():
    validate_row(valid_row())

def test_field_order_is_stable():
    assert SCHEMA_FIELDS == [
        "id",
        "source_split",
        "source_id",
        "chunk_index",
        "tokens",
        "labels",
    ]

@pytest.mark.parametrize("field", SCHEMA_FIELDS)
def test_missing_field_rejected(field):
    row = valid_row()
    del row[field]
    with pytest.raises(SchemaError, match="missing field"):
        validate_row(row)

def test_unexpected_field_rejected():
    with pytest.raises(SchemaError, match="unexpected field"):
        validate_row(valid_row(capitalization=[0, 0, 0, 1]))

def test_length_mismatch_rejected():
    with pytest.raises(SchemaError, match="length mismatch"):
        validate_row(valid_row(labels=["O", "O", "PERIOD"]))

def test_empty_tokens_rejected():
    with pytest.raises(SchemaError, match="must not be empty"):
        validate_row(valid_row(tokens=[], labels=[]))

def test_whitespace_only_token_rejected():
    with pytest.raises(SchemaError, match="empty or whitespace"):
        validate_row(valid_row(tokens=["hôm", "  ", "trời", "đẹp"]))

def test_unknown_label_rejected():
    with pytest.raises(SchemaError, match="not a valid label"):
        validate_row(valid_row(labels=["O", "O", "O", "QMARK"]))

def test_exclamation_is_not_a_label():

    with pytest.raises(SchemaError):
        validate_row(valid_row(labels=["O", "O", "O", "EXCLAM"]))

def test_length_cap_enforced():
    n = MAX_WORDS_PER_EXAMPLE + 1
    with pytest.raises(SchemaError, match="exceeds"):
        validate_row(valid_row(tokens=["x"] * n, labels=["O"] * n))

def test_exactly_max_length_allowed():
    n = MAX_WORDS_PER_EXAMPLE
    validate_row(valid_row(tokens=["x"] * n, labels=["O"] * n))

def test_bad_split_rejected():
    with pytest.raises(SchemaError, match="source_split"):
        validate_row(valid_row(source_split="dev"))

def test_expected_split_mismatch_rejected():
    with pytest.raises(SchemaError, match="but this file holds"):
        validate_row(valid_row(source_split="train"), expected_split="validation")

def test_negative_chunk_index_rejected():
    with pytest.raises(SchemaError, match="chunk_index"):
        validate_row(valid_row(chunk_index=-1))

def test_bool_chunk_index_rejected():

    with pytest.raises(SchemaError, match="chunk_index"):
        validate_row(valid_row(chunk_index=True))

def test_example_roundtrip():
    row = valid_row()
    ex = Example.from_dict(row)
    assert ex.to_dict() == row
    assert len(ex) == 4
    assert ex.text == "hôm nay trời đẹp"

def test_id_helpers():
    assert make_example_id("train", 1) == "train_000001"
    assert make_example_id("validation", 42) == "validation_000042"
    assert make_source_id("test", 7) == "test_seg_000007"

def test_project_relative_path_hides_host_directory(tmp_path):
    assert project_relative_path(PROJECT_ROOT / "outputs" / "metrics.json") == "outputs/metrics.json"
    assert project_relative_path(tmp_path / "internal.json").startswith(".pytest_tmp/")
    assert project_relative_path(PROJECT_ROOT.parent / "external.json") == "external.json"
