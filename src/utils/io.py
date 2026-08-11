"""Deterministic IO helpers.

Every writer in this project goes through these functions so that:

* files are always UTF-8 (Windows default is cp1252 — that would corrupt
  Vietnamese diacritics);
* newlines are always ``\\n`` (``newline=""`` / ``newline="\\n"``) so a file
  hashed on Windows equals the same file hashed on Linux;
* JSON is written with ``ensure_ascii=False`` (keep the real characters) and
  ``sort_keys`` where the content is a mapping, so byte-level hashes are stable
  across runs.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Union

PathLike = Union[str, os.PathLike]


def project_relative_path(path: PathLike) -> str:
    """Return a portable project-relative path without exposing host directories."""
    from src.data.constants import PROJECT_ROOT

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.name


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (a directory) and its parents if missing; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: PathLike) -> Path:
    """Create the parent directory of a file path; return the file path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p





def write_json(path: PathLike, obj: Any, *, sort_keys: bool = False) -> Path:
    """Write ``obj`` as pretty UTF-8 JSON with LF newlines."""
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        f.write("\n")
    return p


def read_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)





def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> int:
    """Write an iterable of dicts as JSON Lines. Returns the number of rows.

    Keys are written in insertion order (not sorted) so the schema field order
    declared in :mod:`src.data.schema` is what lands on disk.
    """
    p = ensure_parent(path)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file row by row (memory friendly for large splits)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Load a whole JSONL file into a list."""
    return list(iter_jsonl(path))


def count_jsonl(path: PathLike) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n





def write_csv(
    path: PathLike, header: Sequence[str], rows: Iterable[Sequence[Any]]
) -> Path:
    """Write a CSV with UTF-8 BOM so Excel opens Vietnamese text correctly."""
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return p





def read_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data
