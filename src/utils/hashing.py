from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Union

from src.utils.io import iter_jsonl, project_relative_path

PathLike = Union[str, os.PathLike]

_CHUNK = 1024 * 1024

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_file(path: PathLike) -> str:
    """Streaming SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def content_sha256_jsonl(path: PathLike) -> str:
    """Serialisation-independent hash of a processed split."""
    h = hashlib.sha256()
    for row in iter_jsonl(path):
        canonical = "{}\t{}\t{}\n".format(
            row.get("id", ""),
            " ".join(row.get("tokens", [])),
            " ".join(row.get("labels", [])),
        )
        h.update(canonical.encode("utf-8"))
    return h.hexdigest()

def hash_jsonl_dataset(path: PathLike) -> Dict[str, object]:
    """Full hash record for one processed split."""
    p = Path(path)
    n_rows = 0
    n_tokens = 0
    for row in iter_jsonl(p):
        n_rows += 1
        n_tokens += len(row.get("tokens", []))
    return {
        "path": project_relative_path(p),
        "exists": True,
        "bytes": p.stat().st_size,
        "num_examples": n_rows,
        "num_tokens": n_tokens,
        "file_sha256": sha256_file(p),
        "content_sha256": content_sha256_jsonl(p),
    }

def missing_hash_record(path: PathLike) -> Dict[str, object]:
    return {"path": project_relative_path(path), "exists": False}
