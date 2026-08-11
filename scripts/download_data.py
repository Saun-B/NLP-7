from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap

from src.data.constants import (
    DATASET_COMMIT,
    DATASET_LICENSE,
    DATASET_LICENSE_FILE,
    DATASET_NAME,
    DATASET_REPO_URL,
    OUTPUT_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RAW_DIR,
    RAW_REPO_DIR,
    SOURCE_SPLIT_FILES,
)
from src.utils.hashing import sha256_file
from src.utils.io import write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("download_data")


def run_git(args: List[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    """Run a git command and return stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return (proc.stdout or "").strip()

def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()

def head_commit(path: Path) -> str:
    return run_git(["rev-parse", "HEAD"], cwd=path)

def clone_repository(target: Path) -> None:
    logger.info("Cloning %s -> %s", DATASET_REPO_URL, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    run_git(["clone", DATASET_REPO_URL, str(target)])

def checkout_commit(repo: Path, commit: str) -> None:
    current = head_commit(repo)
    if current == commit:
        logger.info("Already at pinned commit %s", commit)
        return
    logger.info("Checking out pinned commit %s (currently %s)", commit, current)
    try:
        run_git(["checkout", "--force", commit], cwd=repo)
    except RuntimeError:
        logger.info("Commit not present locally — fetching it explicitly")
        run_git(["fetch", "--all", "--tags"], cwd=repo)
        run_git(["checkout", "--force", commit], cwd=repo)

def verify_commit(repo: Path, expected: str) -> str:
    actual = head_commit(repo)
    if actual != expected:
        raise RuntimeError(
            f"Commit verification FAILED.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            f"The dataset must be pinned to the expected commit. "
            f"Re-run with --force to re-clone."
        )
    logger.info("Commit verified: %s", actual)
    return actual

def verify_raw_files(raw_data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Check every expected raw file exists and hash it."""
    records: Dict[str, Dict[str, Any]] = {}
    for split, filename in SOURCE_SPLIT_FILES.items():
        path = raw_data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected raw file missing: {path}")
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f"Raw file is empty: {path}")
        logger.info("Hashing %s (%.1f MB) …", filename, size / 1024 / 1024)
        records[split] = {
            "split": split,
            "upstream_filename": filename,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": size,
            "sha256": sha256_file(path),
        }
    return records

def copy_license(repo: Path) -> Optional[str]:
    src = repo / "LICENSE"
    if not src.exists():
        logger.warning("Upstream LICENSE not found at %s", src)
        return None
    dst = PROJECT_ROOT / DATASET_LICENSE_FILE
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    logger.info("License copied to %s", dst.name)
    return dst.name

def write_manifest(commit: str, raw_files: Dict[str, Dict[str, Any]], license_file: Optional[str]) -> Path:
    manifest = {
        "dataset_name": DATASET_NAME,
        "repository_url": DATASET_REPO_URL,
        "pinned_commit": DATASET_COMMIT,
        "verified_commit": commit,
        "commit_verified": commit == DATASET_COMMIT,
        "license": DATASET_LICENSE,
        "license_file": license_file,
        "official_split_mapping": {
            "train": "train.txt -> train",
            "validation": "dev.txt -> validation",
            "test": "test.txt -> test",
        },
        "split_policy": (
            "The official JointCapPunc split is preserved exactly: no merging, "
            "no re-randomisation, no movement of examples between splits."
        ),
        "raw_files": raw_files,
        "local_repo_path": RAW_REPO_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = write_json(OUTPUT_DATA_DIR / "data_source_manifest.json", manifest)
    logger.info("Manifest written to %s", path.relative_to(PROJECT_ROOT).as_posix())
    return path

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and pin the JointCapPunc dataset.")
    parser.add_argument(
        "--force", action="store_true", help="delete any existing checkout and re-clone"
    )
    parser.add_argument(
        "--repo-dir", default=str(RAW_REPO_DIR), help="where to place the checkout"
    )
    args = parser.parse_args(argv)

    section("STAGE 1/6 — DOWNLOAD & VERIFY JointCapPunc")
    repo = Path(args.repo_dir)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.force and repo.exists():
        logger.info("--force: removing %s", repo)
        shutil.rmtree(repo, ignore_errors=True)

    if repo.exists() and not is_git_repo(repo):
        raise RuntimeError(
            f"{repo} exists but is not a git checkout. Re-run with --force to replace it."
        )

    if not repo.exists():
        clone_repository(repo)

    checkout_commit(repo, DATASET_COMMIT)
    commit = verify_commit(repo, DATASET_COMMIT)

    raw_data_dir = repo / "data"
    if raw_data_dir != RAW_DATA_DIR:
        logger.info("Using raw data directory %s", raw_data_dir)
    raw_files = verify_raw_files(raw_data_dir)
    license_file = copy_license(repo)
    write_manifest(commit, raw_files, license_file)

    print("\nRaw dataset ready:")
    for split, rec in raw_files.items():
        print(f"  {split:<11} {rec['upstream_filename']:<10} "
              f"{rec['bytes'] / 1024 / 1024:8.1f} MB  sha256={rec['sha256'][:16]}…")
    return 0

if __name__ == "__main__":
    sys.exit(main())
