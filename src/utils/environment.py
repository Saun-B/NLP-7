from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": Path(sys.executable).name,
    }

    try:
        import numpy

        env["numpy_version"] = numpy.__version__
    except Exception:
        env["numpy_version"] = None

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
        env["cudnn_version"] = _safe(lambda: torch.backends.cudnn.version())
        if env["cuda_available"]:
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
            env["gpu_capability"] = f"{props.major}.{props.minor}"
        else:
            env["gpu_count"] = 0
            env["gpu_name"] = None
            env["gpu_total_memory_gb"] = None
            env["gpu_capability"] = None
    except Exception:
        env["torch_version"] = None
        env["cuda_available"] = False
        env["cuda_runtime_version"] = None
        env["cudnn_version"] = None
        env["gpu_count"] = 0
        env["gpu_name"] = None
        env["gpu_total_memory_gb"] = None
        env["gpu_capability"] = None

    for mod_name, key in (
        ("transformers", "transformers_version"),
        ("tokenizers", "tokenizers_version"),
        ("safetensors", "safetensors_version"),
        ("sklearn", "scikit_learn_version"),
        ("pandas", "pandas_version"),
    ):
        try:
            mod = __import__(mod_name)
            env[key] = getattr(mod, "__version__", None)
        except Exception:
            env[key] = None
    return env

def disable_progress_bars() -> None:
    import os

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from tqdm.auto import tqdm

        tqdm.monitor_interval = 0
    except Exception:
        pass
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except Exception:
        pass

def format_environment(env: Dict[str, Any] | None = None) -> str:
    env = env or collect_environment()
    lines = [
        f"Python version      : {env.get('python_version')}",
        f"Platform            : {env.get('platform')}",
        f"PyTorch version     : {env.get('torch_version')}",
        f"CUDA available      : {env.get('cuda_available')}",
        f"CUDA runtime        : {env.get('cuda_runtime_version')}",
        f"GPU name            : {env.get('gpu_name')}",
        f"GPU memory (GB)     : {env.get('gpu_total_memory_gb')}",
        f"Transformers version: {env.get('transformers_version')}",
        f"NumPy version       : {env.get('numpy_version')}",
    ]
    return "\n".join(lines)

