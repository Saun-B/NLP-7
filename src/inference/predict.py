"""CLI entry point for punctuation restoration.

It wraps :class:`src.inference.predictor.PunctuationRestorationPredictor` so the
project can run ``python -m src.inference.predict`` or execute this file
directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.inference.predictor import PunctuationRestorationPredictor


def predict_text(text: str, *, checkpoint: Optional[str] = None) -> dict:
    predictor = (
        PunctuationRestorationPredictor.from_checkpoint(checkpoint)
        if checkpoint
        else PunctuationRestorationPredictor.from_selected_model()
    )
    result = predictor.restore(text)
    data = result.to_dict()
    data["labels"] = result.labels
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore Vietnamese punctuation.")
    parser.add_argument("--text", help="Input text without punctuation.")
    parser.add_argument("--file", help="UTF-8 text file to read input from.")
    parser.add_argument("--checkpoint", help="Optional checkpoint directory.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    parser.add_argument("--output", help="Optional path to write restored text.")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide --text or --file.")

    data = predict_text(text, checkpoint=args.checkpoint)
    if args.output:
        Path(args.output).write_text(data["restored_text"], encoding="utf-8")

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(data["restored_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
