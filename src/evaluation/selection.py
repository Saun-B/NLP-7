"""Model selection — **validation only**, by construction.

The rule
--------
1. Rank by validation **Punctuation Macro-F1** (mean F1 of COMMA/PERIOD/
   QUESTION), descending. Accuracy is never used: with 91% ``O`` it rewards a
   model that predicts nothing.
2. Tie-break by validation **unweighted** loss, ascending — the loss computed
   with one shared unweighted criterion for every model
   (:file:`scripts/compute_unweighted_validation_loss.py`), because each
   experiment's own logged loss is on its own weighted scale.
3. Final tie-break by experiment id, so the outcome is deterministic.

How test leakage is made structurally impossible
------------------------------------------------
:class:`ValidationCandidate` is a frozen dataclass whose fields are *only*
validation quantities. There is nowhere to put a test metric. Constructing one
with a test-looking field raises ``TypeError``, and
:meth:`ValidationCandidate.from_experiment` refuses any artifact that was not
evaluated on ``validation.jsonl``. :func:`select_winner` takes nothing but
those candidates.

Once :func:`write_model_selection` has run, ``winner_locked`` is ``true`` and
the winner must not change — not even if the official test later favours a
different model. :func:`load_locked_winner` enforces that contract for every
downstream consumer (notebook 06/07, inference, UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.data.constants import (
    CHECKPOINT_DIR,
    EXPERIMENT_DIR,
    EXPERIMENT_IDS,
    LABELS,
    MONITOR_METRIC,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    PUNCTUATION_LABELS,
)
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
MODEL_SELECTION_PATH = EVALUATION_DIR / "model_selection.json"
UNWEIGHTED_LOSS_PATH = EVALUATION_DIR / "validation_unweighted_loss.json"

SELECTION_SPLIT = "validation"
SELECTION_METRIC = "punctuation_macro_f1"
TIE_BREAKER = "unweighted_validation_loss"


TIE_TOLERANCE = 1e-6

__all__ = [
    "ValidationCandidate",
    "SelectionResult",
    "load_validation_candidates",
    "select_winner",
    "write_model_selection",
    "load_locked_winner",
    "resolve_winner_checkpoint",
    "MODEL_SELECTION_PATH",
]


class SelectionError(RuntimeError):
    """Raised when selection inputs are unusable or the lock is violated."""


@dataclass(frozen=True)
class ValidationCandidate:
    """One experiment's **validation** results. Contains no test information.

    Frozen and closed: there is no field a test metric could be written into.
    """

    experiment_id: str
    model: str
    model_type: str
    weight_mode: str
    best_epoch: int
    punctuation_macro_f1: float
    unweighted_validation_loss: float
    accuracy: float
    macro_f1: float
    f1_per_class: Dict[str, float] = field(default_factory=dict)
    precision_per_class: Dict[str, float] = field(default_factory=dict)
    recall_per_class: Dict[str, float] = field(default_factory=dict)
    own_loss: Optional[float] = None
    num_evaluated_tokens: int = 0
    checkpoint_dir: str = ""
    total_training_seconds: Optional[float] = None


    @classmethod
    def from_experiment(
        cls,
        experiment_id: str,
        *,
        experiment_dir: PathLike = EXPERIMENT_DIR,
        unweighted_losses: Optional[Dict[str, Any]] = None,
    ) -> "ValidationCandidate":
        exp_dir = Path(experiment_dir) / experiment_id
        summary = read_json(exp_dir / "experiment_summary.json")
        best = read_json(exp_dir / "best_validation_metrics.json")

        if summary.get("status") != "COMPLETED":
            raise SelectionError(
                f"{experiment_id}: status is {summary.get('status')!r}; only COMPLETED "
                "experiments can enter model selection."
            )
        evaluated_on = str(best.get("evaluated_on", ""))
        if not evaluated_on.startswith("validation.jsonl"):
            raise SelectionError(
                f"{experiment_id}: metrics were evaluated on {evaluated_on!r}, not "
                "validation.jsonl. Refusing to use them for model selection."
            )
        if summary.get("test_split_used") is True:
            raise SelectionError(
                f"{experiment_id}: the run recorded test_split_used=true; it cannot "
                "take part in a validation-only selection."
            )

        metrics = best["metrics"]
        per_class = metrics["per_class"]

        loss = None
        if unweighted_losses:
            entry = unweighted_losses.get("results", {}).get(experiment_id)
            if entry is not None:
                loss = float(entry["unweighted_validation_loss"])
        if loss is None:
            raise SelectionError(
                f"{experiment_id}: no comparable unweighted validation loss available. "
                "Run scripts/compute_unweighted_validation_loss.py first."
            )

        meta_path = CHECKPOINT_DIR / experiment_id / "checkpoint_metadata.json"
        model_type = read_json(meta_path)["model_type"] if meta_path.exists() else "unknown"

        return cls(
            experiment_id=experiment_id,
            model=str(summary.get("model", "")),
            model_type=model_type,
            weight_mode=str(summary.get("weight_mode", "")),
            best_epoch=int(summary.get("best_epoch", -1)),
            punctuation_macro_f1=float(metrics["punctuation_macro_f1"]),
            unweighted_validation_loss=loss,
            accuracy=float(metrics["accuracy"]),
            macro_f1=float(metrics["macro_f1"]),
            f1_per_class={c: float(per_class[c]["f1"]) for c in LABELS},
            precision_per_class={c: float(per_class[c]["precision"]) for c in LABELS},
            recall_per_class={c: float(per_class[c]["recall"]) for c in LABELS},
            own_loss=(None if metrics.get("loss") is None else float(metrics["loss"])),
            num_evaluated_tokens=int(metrics.get("num_evaluated_tokens", 0)),
            checkpoint_dir=(CHECKPOINT_DIR / experiment_id).relative_to(PROJECT_ROOT).as_posix(),
            total_training_seconds=summary.get("total_training_seconds"),
        )


    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "weight_mode": self.weight_mode,
            "best_epoch": self.best_epoch,
            "validation_loss_unweighted": round(self.unweighted_validation_loss, 6),
            "validation_accuracy": round(self.accuracy, 6),
            "validation_macro_f1": round(self.macro_f1, 6),
            "validation_punctuation_macro_f1": round(self.punctuation_macro_f1, 6),
        }
        for label in LABELS:
            row[f"f1_{label.lower()}"] = round(self.f1_per_class.get(label, 0.0), 6)
        row["validation_loss_own_weighting"] = (
            None if self.own_loss is None else round(self.own_loss, 6)
        )
        return row


@dataclass
class SelectionResult:
    winner: ValidationCandidate
    ranking: List[ValidationCandidate]
    tie_breaker_used: bool
    margin_over_runner_up: Optional[float]

    @property
    def runner_up(self) -> Optional[ValidationCandidate]:
        return self.ranking[1] if len(self.ranking) > 1 else None


def load_validation_candidates(
    experiment_ids: Sequence[str] = tuple(EXPERIMENT_IDS),
    *,
    experiment_dir: PathLike = EXPERIMENT_DIR,
    unweighted_loss_path: PathLike = UNWEIGHTED_LOSS_PATH,
) -> List[ValidationCandidate]:
    """Read the validation artifacts of every experiment. Reads no test data."""
    path = Path(unweighted_loss_path)
    if not path.exists():
        raise SelectionError(
            f"{path} not found. Run scripts/compute_unweighted_validation_loss.py first."
        )
    losses = read_json(path)
    if losses.get("test_split_used") is True:
        raise SelectionError(f"{path} was produced using the test split; refusing to use it.")

    return [
        ValidationCandidate.from_experiment(
            e, experiment_dir=experiment_dir, unweighted_losses=losses
        )
        for e in experiment_ids
    ]


def select_winner(
    candidates: Sequence[ValidationCandidate], *, tie_tolerance: float = TIE_TOLERANCE
) -> SelectionResult:
    """Pick the winner from validation results alone.

    Deterministic: sorts by ``(-punctuation_macro_f1, unweighted_validation_loss,
    experiment_id)``.
    """
    if not candidates:
        raise SelectionError("No candidates supplied to select_winner().")
    for c in candidates:
        if not isinstance(c, ValidationCandidate):
            raise TypeError(
                "select_winner() accepts ValidationCandidate objects only. This is what "
                "keeps test metrics structurally out of model selection; got "
                f"{type(c).__name__}."
            )

    ranking = sorted(
        candidates,
        key=lambda c: (-c.punctuation_macro_f1, c.unweighted_validation_loss, c.experiment_id),
    )
    winner = ranking[0]

    tie_used = False
    margin = None
    if len(ranking) > 1:
        margin = winner.punctuation_macro_f1 - ranking[1].punctuation_macro_f1
        tie_used = abs(margin) < tie_tolerance

    logger.info(
        "Winner: %s (validation %s = %.6f, unweighted loss = %.6f, tie_breaker_used=%s)",
        winner.experiment_id,
        SELECTION_METRIC,
        winner.punctuation_macro_f1,
        winner.unweighted_validation_loss,
        tie_used,
    )
    return SelectionResult(
        winner=winner, ranking=list(ranking), tie_breaker_used=tie_used, margin_over_runner_up=margin
    )


def write_model_selection(
    result: SelectionResult, *, path: PathLike = MODEL_SELECTION_PATH
) -> Path:
    """Write ``model_selection.json`` and **lock** the winner."""
    winner = result.winner
    blob: Dict[str, Any] = {

        "selection_split": SELECTION_SPLIT,
        "selection_metric": SELECTION_METRIC,
        "tie_breaker": TIE_BREAKER,
        "winner": winner.experiment_id,
        "test_was_used_for_selection": False,
        "winner_locked": True,

        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "winner_model": winner.model,
        "winner_model_type": winner.model_type,
        "winner_weight_mode": winner.weight_mode,
        "winner_best_epoch": winner.best_epoch,
        "winner_validation_punctuation_macro_f1": winner.punctuation_macro_f1,
        "winner_validation_unweighted_loss": winner.unweighted_validation_loss,
        "winner_validation_accuracy": winner.accuracy,
        "winner_validation_macro_f1": winner.macro_f1,
        "winner_validation_f1_per_class": winner.f1_per_class,
        "checkpoint_path": winner.checkpoint_dir,
        "tie_breaker_used": result.tie_breaker_used,
        "margin_over_runner_up": result.margin_over_runner_up,
        "runner_up": (result.runner_up.experiment_id if result.runner_up else None),
        "ranking": [
            {
                "rank": i + 1,
                "experiment_id": c.experiment_id,
                "validation_punctuation_macro_f1": c.punctuation_macro_f1,
                "unweighted_validation_loss": c.unweighted_validation_loss,
            }
            for i, c in enumerate(result.ranking)
        ],
        "policy": (
            "The winner was chosen using the validation split only, before any test "
            "evaluation was run. It is locked: a post-hoc test comparison may not change "
            "it, even if another model scores higher on the official test."
        ),
    }
    out = write_json(path, blob)
    logger.info("Winner %s locked in %s", winner.experiment_id, out)
    return out


def load_locked_winner(path: PathLike = MODEL_SELECTION_PATH) -> Dict[str, Any]:
    """Load ``model_selection.json`` and enforce the lock contract."""
    p = Path(path)
    if not p.exists():
        raise SelectionError(
            f"{p} not found. Run notebooks/05_Model_Comparison_Selection.ipynb "
            "(or scripts/select_model.py) first."
        )
    blob = read_json(p)
    if not blob.get("winner"):
        raise SelectionError(f"{p} has no 'winner'.")
    if blob.get("winner_locked") is not True:
        raise SelectionError(f"{p}: winner_locked is not true — the winner is not final.")
    if blob.get("test_was_used_for_selection") is not False:
        raise SelectionError(
            f"{p}: test_was_used_for_selection is not false. The winner must be chosen "
            "on validation only."
        )
    if blob.get("selection_split") != SELECTION_SPLIT:
        raise SelectionError(
            f"{p}: selection_split is {blob.get('selection_split')!r}, expected "
            f"{SELECTION_SPLIT!r}."
        )
    return blob


def resolve_winner_checkpoint(
    selection: Optional[Dict[str, Any]] = None, *, path: PathLike = MODEL_SELECTION_PATH
) -> Path:
    """Absolute path of the locked winner's checkpoint directory."""
    blob = selection or load_locked_winner(path)
    raw = blob.get("checkpoint_path") or f"outputs/checkpoints/{blob['winner']}"
    ckpt = Path(raw)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise SelectionError(f"Winner checkpoint directory does not exist: {ckpt}")
    return ckpt
