from __future__ import annotations

from pathlib import Path
from typing import Dict, List

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_REPO_DIR: Path = RAW_DIR / "JointCapPunc"
RAW_DATA_DIR: Path = RAW_REPO_DIR / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUT_DATA_DIR: Path = OUTPUTS_DIR / "data"
CHECKPOINT_DIR: Path = OUTPUTS_DIR / "checkpoints"
EXPERIMENT_DIR: Path = OUTPUTS_DIR / "experiments"
SMOKE_DIR: Path = OUTPUTS_DIR / "smoke"

TRAIN_FILE: Path = PROCESSED_DIR / "train.jsonl"
VALIDATION_FILE: Path = PROCESSED_DIR / "validation.jsonl"
TEST_FILE: Path = PROCESSED_DIR / "test.jsonl"
ALL_LABELED_FILE: Path = PROCESSED_DIR / "all_labeled.jsonl"

PROCESSED_FILES: Dict[str, Path] = {
    "train": TRAIN_FILE,
    "validation": VALIDATION_FILE,
    "test": TEST_FILE,
}

DATASET_NAME: str = "JointCapPunc"
DATASET_REPO_URL: str = "https://github.com/ductho9799/JointCapPunc.git"
DATASET_COMMIT: str = "ee258cae0e95e64245428d59ebbb030280fcebec"
DATASET_LICENSE: str = "Apache-2.0"
DATASET_LICENSE_FILE: str = "JOINTCAPPUNC_LICENSE.txt"

SOURCE_SPLIT_FILES: Dict[str, str] = {
    "train": "train.txt",
    "validation": "dev.txt",
    "test": "test.txt",
}
SPLITS: List[str] = ["train", "validation", "test"]

LABEL2ID: Dict[str, int] = {
    "O": 0,
    "COMMA": 1,
    "PERIOD": 2,
    "QUESTION": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
LABELS: List[str] = [k for k, _ in sorted(LABEL2ID.items(), key=lambda kv: kv[1])]
NUM_LABELS: int = len(LABELS)

PUNCTUATION_LABELS: List[str] = ["COMMA", "PERIOD", "QUESTION"]
PUNCTUATION_LABEL_IDS: List[int] = [LABEL2ID[x] for x in PUNCTUATION_LABELS]

LABEL_TO_SYMBOL: Dict[str, str] = {
    "O": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
}

SENTENCE_END_LABELS: List[str] = ["PERIOD", "QUESTION"]

RAW_LABEL_MAP: Dict[str, str] = {
    "O": "O",
    "COMMA": "COMMA",
    "PERIOD": "PERIOD",
    "QMARK": "QUESTION",
}
RAW_LABELS: List[str] = sorted(RAW_LABEL_MAP.keys())

RAW_CAPITALIZATION_LABELS: List[str] = ["0", "1", "2"]

MAX_WORDS_PER_EXAMPLE: int = 150
PHOBERT_MAX_LENGTH: int = 192
IGNORE_INDEX: int = -100

PHOBERT_MODEL_NAME: str = "vinai/phobert-base-v2"
PHOBERT_REVISION: str = "fb76b7e1f77fa19bc4870e2ad956876f7c81c53f"

PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
PAD_ID: int = 0
UNK_ID: int = 1
SEED: int = 42

EXPERIMENT_IDS: List[str] = ["E1", "E2", "E3", "E4"]

EXPERIMENT_CONFIG_FILES: Dict[str, str] = {
    "E1": "e1_bilstm.yaml",
    "E2": "e2_phobert_none.yaml",
    "E3": "e3_phobert_inverse.yaml",
    "E4": "e4_phobert_sqrt_inverse.yaml",
}

MONITOR_METRIC: str = "punctuation_macro_f1"
