from __future__ import annotations

from pathlib import Path
import pytest

from src.data.jointcappunc_parser import (
    ParseError,
    parse_file,
    parse_line,
    parse_split,
)

def write_raw(tmp_path: Path, lines, name: str = "mini.txt", newline: str = "\r\n") -> Path:
    path = tmp_path / name
    path.write_text(newline.join(lines) + newline, encoding="utf-8", newline="")
    return path

GOOD_LINES = [
    "hôm\t1\tO",
    "nay\t0\tO",
    "trời\t0\tO",
    "đẹp\t0\tPERIOD",
    "bạn\t1\tO",
    "khỏe\t0\tO",
    "không\t0\tQMARK",
]

def test_parses_three_column_tsv(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES)
    docs, report = parse_file(path, "train")

    assert len(docs) == 1
    assert docs[0].tokens == ["hôm", "nay", "trời", "đẹp", "bạn", "khỏe", "không"]
    assert docs[0].labels == ["O", "O", "O", "PERIOD", "O", "O", "QUESTION"]
    assert report.parsed_tokens == 7
    assert report.total_lines == 7
    assert len(report.errors) == 0

def test_qmark_is_mapped_during_parsing(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES)
    docs, report = parse_file(path, "train")
    assert "QMARK" not in docs[0].labels
    assert "QUESTION" in docs[0].labels
    assert report.raw_label_counts["QMARK"] == 1
    assert report.mapped_label_counts["QUESTION"] == 1

def test_crlf_and_lf_produce_the_same_result(tmp_path):
    crlf = write_raw(tmp_path, GOOD_LINES, name="crlf.txt", newline="\r\n")
    lf = write_raw(tmp_path, GOOD_LINES, name="lf.txt", newline="\n")
    a, _ = parse_file(crlf, "train")
    b, _ = parse_file(lf, "train")
    assert a[0].tokens == b[0].tokens
    assert a[0].labels == b[0].labels

def test_capitalization_column_is_parsed_then_dropped(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES)
    docs, report = parse_file(path, "train")
    assert report.capitalization_label_counts == {"1": 2, "0": 5}
    assert set(vars(docs[0])) == {"tokens", "labels", "start_line", "end_line"}

def test_blank_line_splits_documents(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES[:4] + [""] + GOOD_LINES[4:])
    docs, report = parse_file(path, "train")
    assert len(docs) == 2
    assert report.blank_lines == 1
    assert docs[0].tokens == ["hôm", "nay", "trời", "đẹp"]
    assert docs[1].tokens == ["bạn", "khỏe", "không"]

def test_no_empty_document_is_emitted(tmp_path):
    path = write_raw(tmp_path, ["", ""] + GOOD_LINES[:2] + ["", ""])
    docs, _ = parse_file(path, "train")
    assert len(docs) == 1
    assert docs[0].tokens == ["hôm", "nay"]

def test_wrong_column_count_raises_with_file_and_line(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES[:2] + ["broken\t0"] + GOOD_LINES[2:])
    with pytest.raises(ParseError) as exc:
        parse_file(path, "train")
    assert exc.value.line_no == 3
    assert "expected 3 tab-separated columns" in str(exc.value)
    assert path.name in str(exc.value)

def test_unknown_punctuation_label_raises(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES[:2] + ["ôi\t0\tEXCLAM"])
    with pytest.raises(ParseError, match="unknown punctuation label"):
        parse_file(path, "train")

def test_unknown_capitalization_label_raises(tmp_path):
    path = write_raw(tmp_path, ["từ\t9\tO"])
    with pytest.raises(ParseError, match="unknown capitalization label"):
        parse_file(path, "train")

def test_empty_token_raises_in_strict_mode(tmp_path):
    path = write_raw(tmp_path, ["  \t0\tO"])
    with pytest.raises(ParseError, match="token is empty after normalization"):
        parse_file(path, "train")

def test_non_strict_mode_records_errors_instead_of_hiding_them(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES[:2] + ["broken\t0"] + GOOD_LINES[2:])
    docs, report = parse_file(path, "train", strict=False)
    assert len(report.errors) == 1
    assert ":3:" in report.errors[0]

    assert report.total_lines == 8
    assert report.parsed_tokens == 7


def test_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        parse_file(tmp_path / "nope.txt", "train")

def test_unknown_split_name_raises(tmp_path):
    with pytest.raises(KeyError):
        parse_split("dev", raw_data_dir=tmp_path)

def test_parse_line_returns_raw_and_mapped_labels():
    token, cap, raw, mapped = parse_line("f.txt", 1, "Không\t1\tqmark")
    assert token == "không"
    assert cap == "1"
    assert raw == "QMARK"
    assert mapped == "QUESTION"

def test_max_lines_truncates(tmp_path):
    path = write_raw(tmp_path, GOOD_LINES)
    docs, report = parse_file(path, "train", max_lines=3)
    assert report.parsed_tokens == 3
    assert docs[0].tokens == ["hôm", "nay", "trời"]
