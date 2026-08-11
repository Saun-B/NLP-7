from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import _bootstrap

import torch
from torch.utils.data import DataLoader

from src.data.constants import (
    CHECKPOINT_DIR,
    EXPERIMENT_DIR,
    EXPERIMENT_IDS,
    IGNORE_INDEX,
    OUTPUTS_DIR,
    PAD_ID,
    PROCESSED_FILES,
    PROJECT_ROOT,
)
from src.data.dataset import (
    BiLSTMDataset,
    PhoBERTDataset,
    PhoBERTEncoder,
    Vocabulary,
    collate_bilstm,
    collate_phobert,
    load_examples,
)
from src.evaluation.evaluator import evaluate
from src.models.factory import load_model_from_checkpoint
from src.models.phobert import load_phobert_tokenizer
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("unweighted_val_loss")

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
REPRODUCTION_TOLERANCE = 1e-3

def build_validation_loader(
    experiment_id: str, meta: Dict[str, Any], examples, *, batch_size: int
) -> DataLoader:
    """Build the validation loader matching the checkpoint's architecture."""
    ckpt_dir = CHECKPOINT_DIR / experiment_id
    model_type = meta["model_type"]

    if model_type == "bilstm":
        vocab = Vocabulary.load(ckpt_dir / "vocabulary.json")
        logger.info("%s: loaded vocabulary of %d types", experiment_id, len(vocab))
        return DataLoader(
            BiLSTMDataset(examples, vocab),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_bilstm(b, pad_id=PAD_ID),
        )

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    except Exception as exc:
        logger.warning("%s: tokenizer not loadable from checkpoint (%s); using hub", experiment_id, exc)
        tokenizer = load_phobert_tokenizer()

    config = read_json(EXPERIMENT_DIR / experiment_id / "config.json")
    max_length = int(config["config"]["model"]["max_length"])
    encoder = PhoBERTEncoder(tokenizer, max_length=max_length)
    return DataLoader(
        PhoBERTDataset(examples, encoder),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_phobert(b, pad_id=encoder.pad_id),
    )

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENT_IDS))
    parser.add_argument("--phobert-batch-size", type=int, default=32)
    parser.add_argument("--bilstm-batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    section("PHASE 2 · STEP 2 — COMPARABLE (UNWEIGHTED) VALIDATION LOSS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Split : validation.jsonl  (test.jsonl is NOT touched)\n")

    validation_examples = load_examples(PROCESSED_FILES["validation"])

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    results: Dict[str, Any] = {}
    for experiment_id in args.experiments:
        ckpt_dir = CHECKPOINT_DIR / experiment_id
        started = time.time()
        model, meta = load_model_from_checkpoint(ckpt_dir, device=device)
        batch_size = (
            args.bilstm_batch_size if meta["model_type"] == "bilstm" else args.phobert_batch_size
        )
        loader = build_validation_loader(
            experiment_id, meta, validation_examples, batch_size=batch_size
        )

        metrics = evaluate(
            model,
            loader,
            device=device,
            loss_fn=criterion,
            use_amp=(device.type == "cuda"),
            desc=f"{experiment_id} unweighted validation",
        )

        reported = float(
            read_json(EXPERIMENT_DIR / experiment_id / "experiment_summary.json")[
                "best_validation_punctuation_macro_f1"
            ]
        )
        reproduced = float(metrics["punctuation_macro_f1"])
        delta = abs(reproduced - reported)
        matches = delta < REPRODUCTION_TOLERANCE

        results[experiment_id] = {
            "experiment_id": experiment_id,
            "model_type": meta["model_type"],
            "weight_mode": meta.get("training_config", {}).get("loss", {}).get("weight_mode"),
            "unweighted_validation_loss": float(metrics["loss"]),
            "reproduced_punctuation_macro_f1": reproduced,
            "reported_punctuation_macro_f1": reported,
            "abs_difference": delta,
            "checkpoint_reproduces_reported_score": matches,
            "num_evaluated_tokens": int(metrics["num_evaluated_tokens"]),
            "seconds": round(time.time() - started, 1),
        }

        status = "MATCH" if matches else "MISMATCH"
        print(
            f"  {experiment_id}: unweighted loss = {metrics['loss']:.6f} | "
            f"PUNCT-F1 reproduced {reproduced:.6f} vs reported {reported:.6f} "
            f"[{status}]  ({results[experiment_id]['seconds']:.0f}s)"
        )

        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_match = all(r["checkpoint_reproduces_reported_score"] for r in results.values())

    blob = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "validation",
        "criterion": "CrossEntropyLoss(weight=None, ignore_index=-100)",
        "purpose": (
            "A single unweighted criterion applied to all four checkpoints, so the "
            "losses are comparable across experiments and can serve as the documented "
            "tie-breaker for model selection."
        ),
        "test_split_used": False,
        "all_checkpoints_reproduce_reported_scores": all_match,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "results": results,
    }
    out = write_json(EVALUATION_DIR / "validation_unweighted_loss.json", blob)
    print(f"\n  Written: {out.relative_to(PROJECT_ROOT).as_posix()}")

    if not all_match:
        print("\n  WARNING: at least one checkpoint did NOT reproduce its reported score.")
        return 1
    print("  All four checkpoints load and reproduce their reported validation scores.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
