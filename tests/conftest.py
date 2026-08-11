from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.constants import PROCESSED_FILES


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def processed_available() -> bool:
    """True when the data pipeline has already produced the processed splits."""
    return all(Path(p).exists() for p in PROCESSED_FILES.values())


@pytest.fixture
def sample_tokens():
    return ["hôm", "nay", "trời", "đẹp", "bạn", "khỏe", "không"]

@pytest.fixture
def sample_labels():
    return ["O", "O", "O", "PERIOD", "O", "O", "QUESTION"]
