from __future__ import annotations

import unicodedata
import pytest

from src.data.constants import LABEL2ID
from src.data.normalization import (
    canonical_text,
    map_raw_label,
    map_raw_labels,
    nfc,
    normalize_text,
    normalize_token,
    normalize_tokens,
    strip_diacritics,
)

def test_decomposed_input_is_composed_to_nfc():
    decomposed = unicodedata.normalize("NFD", "tiếng")
    assert decomposed != "tiếng"
    assert normalize_token(decomposed) == "tiếng"
    assert unicodedata.is_normalized("NFC", normalize_token(decomposed))

def test_nfc_is_idempotent():
    once = normalize_token("Việt")
    assert normalize_token(once) == once

def test_two_encodings_of_the_same_word_unify():
    a = normalize_token(unicodedata.normalize("NFC", "hoà"))
    b = normalize_token(unicodedata.normalize("NFD", "hoà"))
    assert a == b

@pytest.mark.parametrize(
    "word", ["tiếng", "việt", "đẹp", "trời", "khỏe", "nghĩa", "phường", "ạ"]
)
def test_tone_marks_are_never_stripped(word):
    out = normalize_token(word)
    assert out == word
    assert out != strip_diacritics(word)

def test_pipeline_output_is_not_ascii_folded():
    assert normalize_text("tiếng Việt") == "tiếng việt"
    assert normalize_text("tiếng Việt") != "tieng viet"

def test_strip_diacritics_helper_exists_but_is_the_counter_example():
    assert strip_diacritics("tiếng việt") == "tieng viet"
    assert strip_diacritics("đường") == "duong"

def test_whitespace_is_trimmed_and_collapsed():
    assert normalize_token("  hôm  ") == "hôm"
    assert normalize_text("hôm   nay\tthật\n đẹp") == "hôm nay thật đẹp"

def test_lowercasing_can_be_disabled():
    assert normalize_token("Việt", lowercase=False) == "Việt"
    assert normalize_token("Việt", lowercase=True) == "việt"

def test_empty_tokens_are_dropped_by_normalize_tokens():
    assert normalize_tokens(["hôm", "  ", "nay", ""]) == ["hôm", "nay"]

def test_canonical_text_is_a_stable_dedup_key():
    a = canonical_text(["Hôm", "NAY", "trời"])
    b = canonical_text(["hôm", "nay", "trời"])
    assert a == b == "hôm nay trời"

def test_qmark_maps_to_question():
    assert map_raw_label("QMARK") == "QUESTION"

@pytest.mark.parametrize("label", ["O", "COMMA", "PERIOD"])
def test_other_labels_pass_through(label):
    assert map_raw_label(label) == label

def test_mapping_is_case_insensitive_on_input():
    assert map_raw_label("qmark") == "QUESTION"
    assert map_raw_label(" period ") == "PERIOD"

def test_every_mapped_label_is_in_the_label_space():
    for raw in ["O", "COMMA", "PERIOD", "QMARK"]:
        assert map_raw_label(raw) in LABEL2ID

def test_unknown_label_raises_instead_of_defaulting_to_O():
    with pytest.raises(ValueError, match="Unknown raw punctuation label"):
        map_raw_label("EXCLAM")
    with pytest.raises(ValueError):
        map_raw_label("")

def test_map_raw_labels_sequence():
    assert map_raw_labels(["O", "QMARK", "COMMA"]) == ["O", "QUESTION", "COMMA"]

def test_nfc_helper():
    assert nfc(unicodedata.normalize("NFD", "ế")) == "ế"
