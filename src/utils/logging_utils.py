from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_root = False

def configure_stdout_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

def setup_logging(
    level: int = logging.INFO, log_file: Optional[PathLike] = None
) -> None:
    global _configured_root
    configure_stdout_utf8()

    root = logging.getLogger()
    if _configured_root:
        if log_file is not None:
            _attach_file_handler(root, log_file, level)
        return

    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        _attach_file_handler(root, log_file, level)

    _configured_root = True

def _attach_file_handler(root: logging.Logger, log_file: PathLike, level: int) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    setup_logging(level=level)
    return logging.getLogger(name)

def section(title: str, width: int = 78) -> None:
    configure_stdout_utf8()
    print("\n" + "=" * width)
    print(title)
    print("=" * width, flush=True)
