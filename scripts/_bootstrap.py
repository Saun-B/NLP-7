"""Make ``src`` importable when a script is run directly.

Every script starts with ``import _bootstrap`` so that
``python scripts/foo.py`` works from any working directory without requiring
``pip install -e .`` or a PYTHONPATH tweak.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import configure_stdout_utf8

configure_stdout_utf8()
