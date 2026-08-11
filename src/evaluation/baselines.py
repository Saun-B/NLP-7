from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set
from src.data.constants import LABELS

__all__ = [
    "Baseline",
    "MajorityClassBaseline",
    "CueWordHeuristicBaseline",
    "build_baseline",
    "BASELINES",
    "B1_RULES_DESCRIPTION",
]

QUESTION_FINAL_CUES: Set[str] = {
    "không", "ko", "chưa", "hả", "hử", "à", "ạ", "nhỉ", "nhé",
    "chứ", "sao", "gì", "đâu", "nào", "vậy", "thế",
}

SENTENCE_END_CUES: Set[str] = {
    "rồi", "nhé", "nha", "nhá", "đấy", "đó", "ấy", "luôn", "cả", "được",
}

CLAUSE_START_CUES: Set[str] = {
    "nhưng", "tuy", "song", "còn", "mà", "nếu", "thì", "vì", "bởi", "nên",
    "do", "và", "hoặc", "hay", "tức", "ngoài", "trong", "sau", "trước",
    "khi", "để", "vậy", "thế",
}

MIN_WORDS_BEFORE_QUESTION = 4

MIN_WORDS_BEFORE_PERIOD = 6

MIN_WORDS_BEFORE_COMMA = 4

MAX_WORDS_PER_SENTENCE = 25

B1_RULES_DESCRIPTION = f"""
B1 — Vietnamese cue-word heuristic (deterministic, no training data)

Scanning left to right and tracking `n` = number of words since the last
predicted PERIOD/QUESTION, the label of word `w[i]` is the first rule that
matches:

  1. QUESTION  if w[i] ∈ QUESTION_FINAL_CUES and n >= {MIN_WORDS_BEFORE_QUESTION}
                 and (i is the last word, or w[i+1] ∉ QUESTION_FINAL_CUES)
  2. PERIOD    if w[i] ∈ SENTENCE_END_CUES and n >= {MIN_WORDS_BEFORE_PERIOD}
  3. PERIOD    if n >= {MAX_WORDS_PER_SENTENCE}                (run-on guard)
  4. COMMA     if w[i+1] ∈ CLAUSE_START_CUES and n >= {MIN_WORDS_BEFORE_COMMA}
  5. O         otherwise

The final word of a chunk is forced to PERIOD if it would otherwise be O,
because a chunk boundary is (almost always) a sentence boundary in this data.

Cue lists were written from Vietnamese grammar, not tuned on any split.
""".strip()

@dataclass
class Baseline:
    baseline_id: str
    name: str
    description: str

    def predict(self, tokens: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def predict_batch(self, batch: Iterable[Sequence[str]]) -> List[List[str]]:
        return [self.predict(tokens) for tokens in batch]

    def to_dict(self) -> Dict[str, str]:
        return {
            "baseline_id": self.baseline_id,
            "name": self.name,
            "description": self.description,
        }

class MajorityClassBaseline(Baseline):
    def __init__(self) -> None:
        super().__init__(
            baseline_id="B0",
            name="Majority class (all O)",
            description=(
                "Predicts O for every word. Restores no punctuation at all; exists to "
                "show how high accuracy can be while Punctuation Macro-F1 is 0."
            ),
        )

    def predict(self, tokens: Sequence[str]) -> List[str]:
        return ["O"] * len(tokens)

class CueWordHeuristicBaseline(Baseline):
    def __init__(
        self,
        *,
        question_cues: Set[str] = QUESTION_FINAL_CUES,
        end_cues: Set[str] = SENTENCE_END_CUES,
        clause_cues: Set[str] = CLAUSE_START_CUES,
        min_words_before_question: int = MIN_WORDS_BEFORE_QUESTION,
        min_words_before_period: int = MIN_WORDS_BEFORE_PERIOD,
        min_words_before_comma: int = MIN_WORDS_BEFORE_COMMA,
        max_words_per_sentence: int = MAX_WORDS_PER_SENTENCE,
        force_final_period: bool = True,
    ) -> None:
        super().__init__(
            baseline_id="B1",
            name="Vietnamese cue-word heuristic",
            description=B1_RULES_DESCRIPTION,
        )
        self.question_cues = question_cues
        self.end_cues = end_cues
        self.clause_cues = clause_cues
        self.min_q = min_words_before_question
        self.min_p = min_words_before_period
        self.min_c = min_words_before_comma
        self.max_sentence = max_words_per_sentence
        self.force_final_period = force_final_period

    def predict(self, tokens: Sequence[str]) -> List[str]:
        n_tokens = len(tokens)
        labels = ["O"] * n_tokens
        since_end = 0

        for i, raw in enumerate(tokens):
            word = raw.lower()
            nxt = tokens[i + 1].lower() if i + 1 < n_tokens else None
            since_end += 1

            if (
                word in self.question_cues
                and since_end >= self.min_q
                and (nxt is None or nxt not in self.question_cues)
            ):
                labels[i] = "QUESTION"
                since_end = 0
            elif word in self.end_cues and since_end >= self.min_p:
                labels[i] = "PERIOD"
                since_end = 0
            elif since_end >= self.max_sentence:
                labels[i] = "PERIOD"
                since_end = 0
            elif nxt is not None and nxt in self.clause_cues and since_end >= self.min_c:
                labels[i] = "COMMA"

        if self.force_final_period and n_tokens and labels[-1] == "O":
            labels[-1] = "PERIOD"

        assert len(labels) == n_tokens
        assert all(l in LABELS for l in labels)
        return labels

BASELINES: Dict[str, type] = {
    "B0": MajorityClassBaseline,
    "B1": CueWordHeuristicBaseline,
}

def build_baseline(baseline_id: str) -> Baseline:
    try:
        return BASELINES[baseline_id.upper()]()
    except KeyError:
        raise ValueError(
            f"Unknown baseline {baseline_id!r}; available: {sorted(BASELINES)}"
        ) from None

def evaluate_baseline(baseline: Baseline, examples: Sequence) -> Dict[str, object]:
    """Score a baseline over processed examples using the project metrics."""
    from src.data.constants import LABEL2ID
    from src.evaluation.metrics import compute_metrics

    y_true: List[int] = []
    y_pred: List[int] = []
    for ex in examples:
        pred = baseline.predict(ex.tokens)
        if len(pred) != len(ex.labels):
            raise AssertionError(
                f"{baseline.baseline_id} produced {len(pred)} labels for "
                f"{len(ex.labels)} words in {ex.id}"
            )
        y_true.extend(LABEL2ID[l] for l in ex.labels)
        y_pred.extend(LABEL2ID[l] for l in pred)

    metrics = compute_metrics(y_true, y_pred)
    metrics["baseline"] = baseline.to_dict()
    return metrics
