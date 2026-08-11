"""Đóng gói dự án thành file .zip """

from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import _bootstrap

from src.data.constants import EXPERIMENT_IDS, PROJECT_ROOT
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger, section

logger = get_logger("package_release")

ALWAYS_EXCLUDE = [
    ".git/*", ".git",
    ".venv/*", "venv/*", "env/*",
    "**/__pycache__/*", "**/*.pyc", "**/*.pyo",
    ".pytest_cache/*", ".pytest_tmp/*",
    ".ipynb_checkpoints/*", "**/.ipynb_checkpoints/*",
    ".mypy_cache/*", ".ruff_cache/*",
    "data/raw/*",
    "outputs/smoke/*",
    "outputs/nbcheck/*",
    "*.zip",
    "*.log",
    ".DS_Store", "Thumbs.db", "desktop.ini",
]

REPORT_DIRS = [
    "outputs/data",
    "outputs/experiments",
    "outputs/evaluation",
    "outputs/figures",
]

CODE_PATTERNS = [
    "README.md", "PROJECT_AUDIT.md", "JOINTCAPPUNC_LICENSE.txt",
    "requirements.txt", "requirements-lock.txt", "pytest.ini", ".gitignore",
    "configs/*", "scripts/*", "src/**", "tests/*", "notebooks/*", "app/*",
    "report/**", "slides/**",
    "outputs/metrics.json", "outputs/model_comparison.csv",
    "outputs/confusion_matrix.png", "outputs/error_analysis.csv",
    "outputs/predictions.csv", "outputs/asr_evaluation_status.md",
]

def matches(rel: str, patterns: List[str]) -> bool:
    rel_posix = rel.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(rel_posix, pattern):
            return True

        if pattern.endswith("/**") and rel_posix.startswith(pattern[:-3] + "/"):
            return True
        if pattern.endswith("/*") and rel_posix.startswith(pattern[:-2] + "/"):
            return True
    return False

def winner_checkpoint_dir() -> Tuple[Optional[str], Optional[str]]:
    """Trả về (experiment_id, đường dẫn tương đối) của winner, nếu đã khoá."""
    path = PROJECT_ROOT / "outputs" / "evaluation" / "model_selection.json"
    if not path.exists():
        return None, None
    try:
        blob = read_json(path)
    except Exception:
        return None, None
    winner = blob.get("winner")
    if not winner:
        return None, None
    return winner, blob.get("checkpoint_path") or f"outputs/checkpoints/{winner}"

def collect_files(profile: str) -> Tuple[List[Path], Dict[str, Any]]:
    """Chọn file theo profile. Trả về (danh sách file, thông tin profile)."""
    winner, winner_ckpt = winner_checkpoint_dir()

    include: List[str] = list(CODE_PATTERNS)
    for d in REPORT_DIRS:
        include.append(f"{d}/**")

    info: Dict[str, Any] = {"profile": profile, "winner": winner}

    if profile == "code":
        info["checkpoints"] = "none"
        info["processed_data"] = "none"
    elif profile == "submission":
        if winner_ckpt:
            include.append(f"{winner_ckpt}/**")
            info["checkpoints"] = f"winner only ({winner})"
        else:
            info["checkpoints"] = "none (model_selection.json chưa có)"
            logger.warning(
                "Chưa có model_selection.json — không biết winner là ai, "
                "gói sẽ không kèm checkpoint nào."
            )
        info["processed_data"] = "none"
    elif profile == "full":
        for e in EXPERIMENT_IDS:
            include.append(f"outputs/checkpoints/{e}/**")
        include.append("data/processed/**")
        info["checkpoints"] = f"all ({', '.join(EXPERIMENT_IDS)})"
        info["processed_data"] = "included"
    else:
        raise ValueError(f"Profile không hợp lệ: {profile}")

    files: List[Path] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if matches(rel, ALWAYS_EXCLUDE):
            continue
        if not matches(rel, include):
            continue
        files.append(path)

    return files, info

def build_manifest(profile: str, files: List[Path], info: Dict[str, Any]) -> Dict[str, Any]:
    total = sum(f.stat().st_size for f in files)
    by_top: Dict[str, Dict[str, Any]] = {}
    for f in files:
        top = f.relative_to(PROJECT_ROOT).parts[0]
        entry = by_top.setdefault(top, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += f.stat().st_size

    manifest: Dict[str, Any] = {
        "project": "Vietnamese Punctuation Restoration",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "profile_contents": {
            "code_and_notebooks": True,
            "reports_and_figures": True,
            "checkpoints": info.get("checkpoints"),
            "processed_data": info.get("processed_data"),
            "raw_dataset": "not included — chạy `python scripts/download_data.py`",
        },
        "num_files": len(files),
        "total_bytes": total,
        "total_mb": round(total / 1024 / 1024, 1),
        "by_top_level": {
            k: {"files": v["files"], "mb": round(v["bytes"] / 1024 / 1024, 2)}
            for k, v in sorted(by_top.items())
        },
    }

    selection_path = PROJECT_ROOT / "outputs" / "evaluation" / "model_selection.json"
    if selection_path.exists():
        sel = read_json(selection_path)
        manifest["winner"] = {
            "experiment_id": sel.get("winner"),
            "model": sel.get("winner_model"),
            "validation_punctuation_macro_f1": sel.get(
                "winner_validation_punctuation_macro_f1"
            ),
            "winner_locked": sel.get("winner_locked"),
            "test_was_used_for_selection": sel.get("test_was_used_for_selection"),
        }

    test_path = PROJECT_ROOT / "outputs" / "evaluation" / "final_test_results.json"
    if test_path.exists():
        res = read_json(test_path)
        manifest["official_test"] = {
            "punctuation_macro_f1": res["metrics"]["punctuation_macro_f1"],
            "accuracy": res["metrics"]["accuracy"],
            "evaluation_runs": res.get("evaluation_runs_on_test"),
        }

    hashes_path = PROJECT_ROOT / "outputs" / "data" / "data_hashes.json"
    if hashes_path.exists():
        manifest["data_hashes"] = {
            k: v.get("content_sha256")
            for k, v in read_json(hashes_path).items()
            if isinstance(v, dict)
        }

    manifest["how_to_restore"] = [
        "pip install torch --index-url https://download.pytorch.org/whl/cu126",
        "pip install -r requirements.txt",
        "python scripts/run_data_pipeline.py",
        "python -m pytest -q",
        "python scripts/demo_inference.py",
        "streamlit run app/app.py",
    ]
    return manifest

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["code", "submission", "full"], default="submission"
    )
    parser.add_argument("--output", help="đường dẫn file .zip (mặc định: tự đặt tên)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ liệt kê, không nén")
    args = parser.parse_args(argv)

    section(f"ĐÓNG GÓI DỰ ÁN — profile: {args.profile}")

    files, info = collect_files(args.profile)
    manifest = build_manifest(args.profile, files, info)

    print(f"  Số file        : {manifest['num_files']:,}")
    print(f"  Tổng dung lượng: {manifest['total_mb']:,.1f} MB (trước khi nén)")
    print(f"  Winner         : {info.get('winner') or 'chưa có'}")
    print(f"  Checkpoint     : {info.get('checkpoints')}")
    print(f"  Data đã xử lý  : {info.get('processed_data')}")
    print("\n  Theo thư mục:")
    for top, stats in manifest["by_top_level"].items():
        print(f"    {top:<16}{stats['files']:>6} file{stats['mb']:>12,.2f} MB")

    if args.dry_run:
        print("\n  --dry-run: không tạo file zip.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_name = f"NLPgrSix7_{args.profile}_{stamp}.zip"
    out = Path(args.output) if args.output else PROJECT_ROOT / default_name
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = PROJECT_ROOT / "PACKAGE_MANIFEST.json"
    write_json(manifest_path, manifest)

    print(f"\n  Đang nén -> {out.name} …")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(manifest_path, "PACKAGE_MANIFEST.json")
        for i, f in enumerate(files, start=1):
            zf.write(f, f.relative_to(PROJECT_ROOT).as_posix())
            if i % 100 == 0:
                print(f"    {i:,}/{len(files):,} file…", flush=True)

    size_mb = out.stat().st_size / 1024 / 1024
    ratio = 100 * size_mb / max(manifest["total_mb"], 0.001)
    print(f"\n  Xong: {out.name}")
    print(f"  Dung lượng zip : {size_mb:,.1f} MB  (nén còn {ratio:.0f}%)")
    print(f"  Manifest       : PACKAGE_MANIFEST.json (đã nằm trong zip)")

    if args.profile == "code":
        print(
            "\n  Lưu ý: gói 'code' KHÔNG có trọng số model. Người nhận đọc được mọi\n"
            "  báo cáo/biểu đồ nhưng phải huấn luyện lại nếu muốn chạy inference."
        )
    elif args.profile == "submission":
        print(
            f"\n  Gói 'submission' có checkpoint winner ({info.get('winner')}), nên chạy\n"
            "  được ngay `python scripts/demo_inference.py` và `streamlit run app/app.py`\n"
            "  sau khi cài requirements."
        )
    return 0

if __name__ == "__main__":
    sys.exit(main())
