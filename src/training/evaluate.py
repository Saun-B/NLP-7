"""Command-line evaluation wrapper for saved checkpoints.

The canonical evaluation implementation lives in ``src.evaluation``. This file
can recompute metrics from any saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.data.dataset import load_examples
from src.evaluation.evaluator import evaluate
from src.evaluation.loaders import build_eval_dataloader
from src.models.factory import load_model_from_checkpoint
from src.utils.io import project_relative_path, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_saved_checkpoint(
    checkpoint_dir: str | Path,
    split_file: str | Path,
    *,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint, evaluate a split, and include throughput numbers."""
    checkpoint_dir = Path(checkpoint_dir)
    split_file = Path(split_file)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = load_examples(split_file, limit=limit)
    model, meta = load_model_from_checkpoint(checkpoint_dir, device=device)
    model_type = meta["model_type"]
    if batch_size is None:
        batch_size = 8 if model_type == "phobert" else 256

    dataloader, _, _ = build_eval_dataloader(
        checkpoint_dir,
        examples,
        batch_size=batch_size,
        num_workers=0,
    )

    started = time.perf_counter()
    metrics = evaluate(
        model,
        dataloader,
        device=device,
        use_amp=(device.type == "cuda" and model_type == "phobert"),
        progress=True,
        desc=f"{meta.get('experiment_id', checkpoint_dir.name)} evaluation",
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "checkpoint_dir": project_relative_path(checkpoint_dir),
        "split_file": project_relative_path(split_file),
        "experiment_id": meta.get("experiment_id"),
        "model_name": meta.get("model_name"),
        "model_type": model_type,
        "num_examples": len(examples),
        "num_words": int(metrics["num_evaluated_tokens"]),
        "evaluation_seconds": elapsed,
        "average_inference_ms_per_example": elapsed * 1000.0 / len(examples),
        "examples_per_second": len(examples) / elapsed if elapsed else None,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--split",
        default=str(PROJECT_ROOT / "data" / "processed" / "test.jsonl"),
        help="Processed JSONL split.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    report = evaluate_saved_checkpoint(
        args.checkpoint,
        args.split,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
