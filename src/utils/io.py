from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Union

PathLike = Union[str, os.PathLike]

def project_relative_path(path: PathLike) -> str:
    from src.data.constants import PROJECT_ROOT

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.name

def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_parent(path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: PathLike, obj: Any, *, sort_keys: bool = False) -> Path:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        f.write("\n")
    return p

def read_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> int:
    p = ensure_parent(path)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n

def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
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
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return p

def read_yaml(path: PathLike) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data
