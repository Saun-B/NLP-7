"""CLI demo — restore punctuation with the locked winner, no Streamlit needed.

    python scripts/demo_inference.py
    python scripts/demo_inference.py --interactive
    python scripts/demo_inference.py --text "hôm nay trời đẹp bạn khỏe không"
    python scripts/demo_inference.py --file input.txt
    python scripts/demo_inference.py --save

Latency note
------------
The first call pays for CUDA context creation and lazy kernel compilation, so a
warm-up pass runs before anything is timed. Even then, the numbers reported are
**demo latency on this machine**, measured over a handful of short inputs — not
a benchmark. A real benchmark would need many inputs, fixed batch sizes,
repeated runs and reported variance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap

from src.data.constants import OUTPUTS_DIR, PROJECT_ROOT
from src.utils.io import write_json
from src.utils.logging_utils import get_logger, section

sys.path.insert(0, str(PROJECT_ROOT / "app"))
from examples import EXAMPLES

logger = get_logger("demo_inference")

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"


def print_result(index: int, label: str, note: str, response) -> None:
    print(f"\n[{index}] {label}")
    if note:
        print(f"    ({note})")
    print(f"    input    : {response.input_text}")
    if response.ok:
        print(f"    restored : {response.restored_text}")
        print(
            f"    stats    : {response.num_words} từ · {response.num_sentences} câu · "
            f"{response.latency_ms:.1f} ms"
        )
    else:
        print(f"    ERROR [{response.error_type}]: {response.error}")


def run_examples(service, *, save: bool) -> int:
    section("DEMO — restore punctuation using the locked winner")

    info = service.model_info()
    print(f"  Winner experiment : {info['experiment_id']}")
    print(f"  Model             : {info['model_name']}")
    print(f"  Architecture      : {info['model_type']}")
    print(f"  Selected on       : {info.get('selected_on')} / {info.get('selection_metric')}")
    print(f"  Winner locked     : {info.get('winner_locked')}")
    print(f"  Validation score  : {info.get('validation_punctuation_macro_f1')}")
    print(f"  Device            : {info['device']} (fp16={info['fp16']})")
    print(f"  Warm-up           : {info['warmup_ms']:.0f} ms" if info.get("warmup_ms") else "")

    rows: List[Dict[str, Any]] = []
    failures = 0
    for i, example in enumerate(EXAMPLES, start=1):
        response = service.restore(example["text"])
        print_result(i, example["label"], example.get("note", ""), response)
        if not response.ok:
            failures += 1
            continue
        rows.append(
            {
                "label": example["label"],
                "input": response.input_text,
                "restored": response.restored_text,
                "num_words": response.num_words,
                "num_sentences": response.num_sentences,
                "latency_ms": round(response.latency_ms, 2),
                "winner": response.experiment_id,
                "model": response.model_name,
            }
        )

    if rows:
        latencies = [r["latency_ms"] for r in rows]
        words = [r["num_words"] for r in rows]
        print(
            f"\n  Demo latency over {len(rows)} inputs "
            f"({min(words)}–{max(words)} words): "
            f"min {min(latencies):.1f} ms · median "
            f"{sorted(latencies)[len(latencies) // 2]:.1f} ms · max {max(latencies):.1f} ms"
        )
        print("  (demo latency on this machine — not a production benchmark)")

    if save and rows:
        blob = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "winner": info["experiment_id"],
            "model": info["model_name"],
            "model_type": info["model_type"],
            "device": info["device"],
            "warmup_ms": info.get("warmup_ms"),
            "measurement_note": (
                "Demo inference latency measured on a handful of short inputs after a "
                "warm-up pass. Not a production benchmark: no repeated runs, no batching "
                "study, no variance reported."
            ),
            "results": rows,
        }
        out = write_json(EVALUATION_DIR / "inference_demo.json", blob)
        print(f"\n  Written: {out.relative_to(PROJECT_ROOT).as_posix()}")

    return 1 if failures else 0


def run_interactive(service) -> int:
    section("INTERACTIVE MODE — nhập văn bản tiếng Việt không dấu câu")
    info = service.model_info()
    print(f"  Winner: {info['experiment_id']} ({info['model_name']}) on {info['device']}")
    print("  Gõ văn bản rồi Enter. Gõ 'quit' hoặc Ctrl-C để thoát.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if text.lower() in {"quit", "exit", "q"}:
            return 0
        if not text:
            continue
        response = service.restore(text)
        if response.ok:
            print(f"  {response.restored_text}")
            print(
                f"  [{response.num_words} từ · {response.num_sentences} câu · "
                f"{response.latency_ms:.1f} ms]\n"
            )
        else:
            print(f"  ERROR [{response.error_type}]: {response.error}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="restore a single string and exit")
    parser.add_argument("--file", help="restore the contents of a UTF-8 text file")
    parser.add_argument("--interactive", action="store_true", help="type text in a loop")
    parser.add_argument(
        "--save",
        action="store_true",
        help="write outputs/evaluation/inference_demo.json from the built-in examples",
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = parser.parse_args(argv)

    import torch

    from src.inference.service import PunctuationService

    device = torch.device("cpu") if args.cpu else None
    service = PunctuationService(device=device)

    try:
        service.predictor
    except Exception as exc:
        print(f"\nCannot load the model: {exc}")
        print(
            "\nMake sure notebooks/05_Model_Comparison_Selection.ipynb has been run so "
            "outputs/evaluation/model_selection.json exists and the winner checkpoint "
            "is present."
        )
        return 1

    if args.text:
        response = service.restore(args.text)
        print_result(1, "input", "", response)
        return 0 if response.ok else 1

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        response = service.restore(path.read_text(encoding="utf-8"))
        print_result(1, path.name, "", response)
        return 0 if response.ok else 1

    if args.interactive:
        return run_interactive(service)

    return run_examples(service, save=args.save)


if __name__ == "__main__":
    sys.exit(main())
