from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap

from src.data.constants import (
    CONFIG_DIR,
    EXPERIMENT_CONFIG_FILES,
    PROCESSED_FILES,
    SMOKE_DIR,
)
from src.utils.io import read_yaml, write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("smoke_test")

BANNER = "SMOKE TEST ONLY — NOT FINAL TRAINING — NOT FINAL RESULT"

def shrink_config(config: Dict[str, Any], *, train_batch_size: int) -> Dict[str, Any]:
    """Cut the config down to something that finishes in seconds."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    training = dict(cfg.get("training", {}))
    training["epochs"] = 1
    training["train_batch_size"] = train_batch_size
    training["eval_batch_size"] = max(2, train_batch_size)
    training["log_every_n_steps"] = 1
    training["num_workers"] = 0
    cfg["training"] = training
    return cfg

def run_one(experiment_id: str, *, train_size: int, val_size: int) -> Dict[str, Any]:
    import torch

    from src.data.dataset import load_examples
    from src.training.trainer_bilstm import build_bilstm_trainer
    from src.training.trainer_phobert import build_phobert_trainer

    section(f"{BANNER} — {experiment_id}")

    config = read_yaml(CONFIG_DIR / EXPERIMENT_CONFIG_FILES[experiment_id])
    model_type = config["model"]["type"]
    batch = 8 if model_type == "bilstm" else 2
    config = shrink_config(config, train_batch_size=batch)

    train_examples = load_examples(PROCESSED_FILES["train"], limit=train_size)
    val_examples = load_examples(PROCESSED_FILES["validation"], limit=val_size)

    smoke_ckpt = SMOKE_DIR / "checkpoints" / experiment_id
    smoke_exp = SMOKE_DIR / "experiments"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "bilstm":
        trainer, _ = build_bilstm_trainer(
            config,
            train_examples,
            val_examples,
            device=device,
            checkpoint_dir=smoke_ckpt,
            experiment_base_dir=smoke_exp,
        )
    else:
        trainer, _ = build_phobert_trainer(
            config,
            train_examples,
            val_examples,
            device=device,
            checkpoint_dir=smoke_ckpt,
            experiment_base_dir=smoke_exp,
        )

    result = trainer.fit()
    trainer.artifacts.write_best_validation(
        trainer.checkpoints.best_metrics, best_epoch=result.best_epoch
    )
    trainer.artifacts.write_per_class(trainer.checkpoints.best_metrics)
    trainer.artifacts.write_confusion_matrix(trainer.checkpoints.best_metrics)
    write_json(
        trainer.artifacts.path("experiment_summary.json"),
        {
            "status": "SMOKE_TEST_ONLY",
            "warning": BANNER,
            "experiment_id": experiment_id,
            "train_examples_used": len(train_examples),
            "validation_examples_used": len(val_examples),
            "epochs": 1,
            "best_validation_punctuation_macro_f1": result.best_score,
            "note": (
                "These numbers come from a few dozen examples and one epoch. "
                "They are a wiring check, not an experiment result. The real run "
                "happens in notebooks/ and writes to outputs/experiments/."
            ),
        },
    )

    return {
        "experiment_id": experiment_id,
        "model_type": model_type,
        "seconds": round(result.total_seconds, 1),
        "best_score_meaningless": round(result.best_score, 6),
        "checkpoint_written": smoke_ckpt.exists(),
    }

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=BANNER)
    parser.add_argument("--experiments", nargs="+", default=["E1", "E2"])
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--clean", action="store_true", help="wipe outputs/smoke first")
    args = parser.parse_args(argv)

    if args.clean and SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    print(BANNER)
    print(f"Writing everything to {SMOKE_DIR} — the real experiment folders are untouched.\n")

    results = []
    for experiment_id in args.experiments:
        results.append(run_one(experiment_id, train_size=args.train_size, val_size=args.val_size))

    write_json(
        SMOKE_DIR / "smoke_test_report.json",
        {"warning": BANNER, "results": results},
    )

    section(BANNER + " — SUMMARY")
    for r in results:
        print(
            f"  {r['experiment_id']:<4} {r['model_type']:<8} "
            f"{r['seconds']:>7.1f}s  checkpoint_written={r['checkpoint_written']}"
        )
    print(
        "\nWiring verified. These scores are NOT results — train for real via the "
        "four notebooks."
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
