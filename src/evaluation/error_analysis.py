from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from src.data.constants import LABELS, LABEL_TO_SYMBOL, PUNCTUATION_LABELS

__all__ = [
    "ErrorCase",
    "ConfusionPair",
    "analyze_errors",
    "KEY_CONFUSION_PAIRS",
    "error_rows_for_csv",
    "ERROR_CSV_HEADER",
    "describe_pair",
]

KEY_CONFUSION_PAIRS: List[Tuple[str, str]] = [
    ("COMMA", "O"),
    ("O", "COMMA"),
    ("COMMA", "PERIOD"),
    ("PERIOD", "COMMA"),
    ("PERIOD", "O"),
    ("O", "PERIOD"),
    ("QUESTION", "O"),
    ("O", "QUESTION"),
]

PAIR_MEANING: Dict[Tuple[str, str], str] = {
    ("COMMA", "O"): "bỏ sót dấu phẩy — hai mệnh đề bị dính vào nhau",
    ("O", "COMMA"): "thêm dấu phẩy thừa — chèn ranh giới mệnh đề không có thật",
    ("COMMA", "PERIOD"): "cắt câu quá mạnh — đáng lẽ chỉ là dấu phẩy",
    ("PERIOD", "COMMA"): "kết thúc câu quá yếu — hai câu bị nối thành câu ghép",
    ("PERIOD", "O"): "bỏ sót dấu chấm — hai câu bị nhập làm một (run-on)",
    ("O", "PERIOD"): "chấm câu quá sớm — một câu bị chẻ đôi",
    ("QUESTION", "O"): "mất hẳn câu hỏi — lỗi nghiêm trọng nhất về mặt ngữ nghĩa",
    ("O", "QUESTION"): "thêm dấu hỏi thừa — biến câu trần thuật thành câu hỏi",
}

def describe_pair(gold: str, predicted: str) -> str:
    if gold == predicted:
        return "dự đoán đúng"
    return PAIR_MEANING.get((gold, predicted), f"{gold} bị dự đoán thành {predicted}")

@dataclass
class ErrorCase:
    example_id: str
    word_index: int
    word: str
    gold_label: str
    predicted_label: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "word_index": self.word_index,
            "word": self.word,
            "gold_label": self.gold_label,
            "predicted_label": self.predicted_label,
            "context": self.context,
        }

@dataclass
class ConfusionPair:
    gold_label: str
    predicted_label: str
    count: int = 0
    examples: List[ErrorCase] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.gold_label != self.predicted_label

def _context_snippet(
    tokens: Sequence[str],
    gold: Sequence[str],
    pred: Sequence[str],
    index: int,
    *,
    window: int = 6,
) -> str:
    start = max(0, index - window)
    end = min(len(tokens), index + window + 1)

    pieces: List[str] = []
    if start > 0:
        pieces.append("…")
    for i in range(start, end):
        word = tokens[i]
        if i == index:
            g = gold[i]
            p = pred[i]
            pieces.append(
                f"**{word}**[gold={g}{LABEL_TO_SYMBOL.get(g, '')!r} "
                f"pred={p}{LABEL_TO_SYMBOL.get(p, '')!r}]"
            )
        else:
            pieces.append(word + LABEL_TO_SYMBOL.get(gold[i], ""))
    if end < len(tokens):
        pieces.append("…")
    return " ".join(pieces)

def analyze_errors(
    examples: Sequence[Any],
    predictions: Dict[str, Sequence[str]],
    *,
    max_examples_per_pair: int = 5,
    context_window: int = 6,
    include_correct: bool = False,
) -> Dict[str, Any]:
    pairs: Dict[Tuple[str, str], ConfusionPair] = {}
    total_positions = 0
    total_errors = 0
    skipped = 0

    for ex in examples:
        pred = predictions.get(ex.id)
        if pred is None:
            skipped += 1
            continue
        if len(pred) != len(ex.tokens):
            raise AssertionError(
                f"{ex.id}: {len(pred)} predicted labels for {len(ex.tokens)} words"
            )

        for i, (gold_label, pred_label) in enumerate(zip(ex.labels, pred)):
            total_positions += 1
            if gold_label == pred_label and not include_correct:
                continue
            if gold_label != pred_label:
                total_errors += 1

            key = (gold_label, pred_label)
            pair = pairs.get(key)
            if pair is None:
                pair = pairs[key] = ConfusionPair(gold_label, pred_label)
            pair.count += 1
            if len(pair.examples) < max_examples_per_pair:
                pair.examples.append(
                    ErrorCase(
                        example_id=ex.id,
                        word_index=i,
                        word=ex.tokens[i],
                        gold_label=gold_label,
                        predicted_label=pred_label,
                        context=_context_snippet(
                            ex.tokens, ex.labels, pred, i, window=context_window
                        ),
                    )
                )

    ordered = sorted(pairs.values(), key=lambda p: (-p.count, p.gold_label, p.predicted_label))

    key_pairs = []
    for gold, predicted in KEY_CONFUSION_PAIRS:
        pair = pairs.get((gold, predicted))
        count = pair.count if pair else 0
        key_pairs.append(
            {
                "gold_label": gold,
                "predicted_label": predicted,
                "count": count,
                "share_of_all_errors": (count / total_errors) if total_errors else 0.0,
                "meaning": describe_pair(gold, predicted),
                "representative_examples": (
                    [e.to_dict() for e in pair.examples] if pair else []
                ),
            }
        )

    per_gold_error_counts = {label: 0 for label in LABELS}
    for pair in ordered:
        if pair.is_error:
            per_gold_error_counts[pair.gold_label] += pair.count

    return {
        "num_examples_analyzed": len(examples) - skipped,
        "num_examples_skipped": skipped,
        "total_positions": total_positions,
        "total_errors": total_errors,
        "error_rate": (total_errors / total_positions) if total_positions else 0.0,
        "errors_by_gold_label": per_gold_error_counts,
        "key_confusion_pairs": key_pairs,
        "all_pairs": [
            {
                "gold_label": p.gold_label,
                "predicted_label": p.predicted_label,
                "count": p.count,
                "share_of_all_errors": (p.count / total_errors) if total_errors else 0.0,
                "is_error": p.is_error,
                "meaning": describe_pair(p.gold_label, p.predicted_label),
                "representative_examples": [e.to_dict() for e in p.examples],
            }
            for p in ordered
        ],
        "punctuation_labels": list(PUNCTUATION_LABELS),
    }

ERROR_CSV_HEADER = [
    "gold_label",
    "predicted_label",
    "count",
    "share_of_all_errors",
    "meaning",
    "representative_examples",
]

def error_rows_for_csv(
    analysis: Dict[str, Any], *, max_examples: int = 3, separator: str = " ||| "
) -> List[List[Any]]:
    """Flatten the analysis into ``final_error_analysis.csv`` rows."""
    rows: List[List[Any]] = []
    for pair in analysis["all_pairs"]:
        if not pair["is_error"]:
            continue
        contexts = separator.join(
            e["context"] for e in pair["representative_examples"][:max_examples]
        )
        rows.append(
            [
                pair["gold_label"],
                pair["predicted_label"],
                pair["count"],
                round(pair["share_of_all_errors"], 6),
                pair["meaning"],
                contexts,
            ]
        )
    return rows
