from __future__ import annotations

import pytest
from src.inference.reconstruction import (
    reconstruct_text,
    reconstruct_with_details,
    split_into_sentences,
)
from src.inference.tokenizer import InferenceTokenizer, tokenize_for_inference

def test_the_worked_example_from_the_spec():
    words = "bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài".split()
    labels = ["O"] * 6 + ["QUESTION"] + ["O"] * 5 + ["PERIOD"]
    assert reconstruct_text(words, labels) == (
        "Bạn đã hoàn thành bài tập chưa? Ngày mai chúng ta nộp bài."
    )

@pytest.mark.parametrize(
    "label,symbol", [("O", ""), ("COMMA", ","), ("PERIOD", "."), ("QUESTION", "?")]
)
def test_each_label_renders_its_symbol(label, symbol):
    assert reconstruct_text(["từ"], [label], capitalize=False) == f"từ{symbol}"

def test_comma_does_not_start_a_new_sentence():
    out = reconstruct_text(["a", "b", "c"], ["COMMA", "O", "PERIOD"])
    assert out == "A, b c."

def test_capitalisation_after_period_and_question():
    words = ["một", "hai", "ba", "bốn"]
    labels = ["PERIOD", "O", "QUESTION", "PERIOD"]
    assert reconstruct_text(words, labels) == "Một. Hai ba? Bốn."

def test_capitalisation_can_be_disabled():
    assert reconstruct_text(["xin", "chào"], ["O", "PERIOD"], capitalize=False) == "xin chào."

def test_capitalisation_preserves_the_rest_of_the_word():

    assert reconstruct_text(["COVID", "nhé"], ["O", "PERIOD"]) == "COVID nhé."

def test_diacritics_survive_reconstruction():
    out = reconstruct_text(["tiếng", "việt"], ["O", "PERIOD"])
    assert out == "Tiếng việt."
    assert "tieng" not in out

def test_length_mismatch_raises_instead_of_silently_shifting():
    with pytest.raises(ValueError, match="exactly one label"):
        reconstruct_text(["a", "b", "c"], ["O", "PERIOD"])

def test_unknown_label_raises():
    with pytest.raises(ValueError, match="Unknown label"):
        reconstruct_text(["a"], ["EXCLAM"])

def test_empty_input_returns_empty_string():
    assert reconstruct_text([], []) == ""

def test_every_word_appears_in_the_output():
    words = "một hai ba bốn năm sáu bảy".split()
    labels = ["O", "COMMA", "O", "PERIOD", "O", "O", "QUESTION"]
    out = reconstruct_text(words, labels)
    for w in words:
        assert w in out.lower()

def test_details_count_punctuation():
    labels = ["COMMA", "O", "PERIOD", "O", "QUESTION"]
    d = reconstruct_with_details(["a", "b", "c", "d", "e"], labels)
    assert (d.num_words, d.num_commas, d.num_periods, d.num_questions) == (5, 1, 1, 1)
    assert d.num_sentences == 2

def test_ensure_final_punctuation_appends_a_period():
    d = reconstruct_with_details(["a", "b"], ["O", "O"], ensure_final_punctuation=True)
    assert d.text.endswith(".")
    assert d.num_periods == 1

def test_ensure_final_punctuation_is_a_noop_when_already_terminated():
    d = reconstruct_with_details(["a", "b"], ["O", "QUESTION"], ensure_final_punctuation=True)
    assert d.text == "A b?"
    assert d.num_periods == 0

def test_ensure_final_punctuation_replaces_a_trailing_comma():
    out = reconstruct_text(["a", "b"], ["O", "COMMA"], ensure_final_punctuation=True)
    assert out.endswith(".")
    assert not out.endswith(",.")

def test_split_into_sentences():
    words = "một hai ba bốn năm".split()
    labels = ["O", "PERIOD", "O", "QUESTION", "O"]
    sentences = split_into_sentences(words, labels)
    assert sentences == ["Một hai.", "Ba bốn?", "Năm"]

def test_split_mismatch_raises():
    with pytest.raises(ValueError):
        split_into_sentences(["a"], ["O", "O"])

def test_tokenizer_splits_on_whitespace():
    t = tokenize_for_inference("hôm nay trời đẹp")
    assert t.words == ["hôm", "nay", "trời", "đẹp"]
    assert t.model_words == ["hôm", "nay", "trời", "đẹp"]

def test_tokenizer_strips_existing_punctuation():
    t = tokenize_for_inference("Hôm nay, trời đẹp. Bạn khỏe không?")
    assert t.words == ["Hôm", "nay", "trời", "đẹp", "Bạn", "khỏe", "không"]
    assert t.num_removed_punctuation > 0

def test_tokenizer_lowercases_only_the_model_view():
    t = tokenize_for_inference("Hôm Nay")
    assert t.words == ["Hôm", "Nay"]
    assert t.model_words == ["hôm", "nay"]

def test_tokenizer_keeps_diacritics():
    t = tokenize_for_inference("tiếng Việt đẹp")
    assert t.model_words == ["tiếng", "việt", "đẹp"]

@pytest.mark.parametrize("protected", ["38.5", "1,5", "2.500.000", "12:30", "3/4"])
def test_tokenizer_keeps_numbers_whole(protected):
    t = tokenize_for_inference(f"em bị sốt {protected} độ")
    assert protected in t.words, f"{protected} was split apart: {t.words}"

def test_tokenizer_keeps_urls_whole():
    t = tokenize_for_inference("xem tại https://vnexpress.net/tin-tuc nhé")
    assert any("vnexpress.net" in w for w in t.words)
    assert len([w for w in t.words if "vnexpress" in w]) == 1

def test_tokenizer_keeps_emails_whole():
    t = tokenize_for_inference("liên hệ bacsi@benhvien.vn để đặt lịch")
    assert "bacsi@benhvien.vn" in t.words

def test_tokenizer_keeps_hyphenated_words():
    t = tokenize_for_inference("chụp x-quang và xét nghiệm covid-19")
    assert "x-quang" in t.words
    assert "covid-19" in t.words

def test_tokenizer_collapses_messy_whitespace():
    t = tokenize_for_inference("  hôm   nay\t\ttrời\n\nđẹp  ")
    assert t.words == ["hôm", "nay", "trời", "đẹp"]

def test_tokenizer_on_empty_input():
    for text in ("", "   ", "\n\t", "..., ?!"):
        assert tokenize_for_inference(text).is_empty

def test_tokenizer_rejects_none():
    with pytest.raises(TypeError):
        tokenize_for_inference(None)

def test_tokenizer_word_count_matches_label_count_contract():
    text = "chào bác sĩ em bị sốt 38.5 độ từ hôm qua"
    t = tokenize_for_inference(text)
    labels = ["O"] * len(t.words)
    out = reconstruct_text(t.words, labels)
    assert len(out.split()) == len(t.words)

def test_tokenizer_option_to_lowercase_display_words():
    t = InferenceTokenizer(keep_original_case=False).tokenize("Hôm Nay")
    assert t.words == ["hôm", "nay"]
