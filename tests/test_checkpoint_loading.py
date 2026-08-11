from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.constants import CHECKPOINT_DIR, EXPERIMENT_IDS, LABEL2ID, LABELS
from src.training.checkpointing import CheckpointManager

REQUIRED_METADATA_KEYS = [
    "experiment_id",
    "model_type",
    "model_name",
    "model_revision",
    "label2id",
    "id2label",
    "monitor_metric",
    "best_epoch",
    "best_score",
    "seed",
    "training_config",
    "data_hashes",
    "python_version",
    "torch_version",
    "transformers_version",
    "cuda_version",
    "gpu_name",
]

AVAILABLE = [e for e in EXPERIMENT_IDS if CheckpointManager.has_checkpoint(CHECKPOINT_DIR / e)]

pytestmark = pytest.mark.integration

def require(experiment_id: str):
    if experiment_id not in AVAILABLE:
        pytest.skip(f"{experiment_id} checkpoint not present — run the training notebook")
    return CHECKPOINT_DIR / experiment_id

@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_metadata_is_complete(experiment_id):
    ckpt = require(experiment_id)
    meta = CheckpointManager.read_metadata(ckpt)
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    assert not missing, f"{experiment_id}: metadata missing {missing}"
    assert meta["experiment_id"] == experiment_id
    assert meta["seed"] == 42
    assert meta["monitor_metric"] == "punctuation_macro_f1"

@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_label_mapping_matches_the_project(experiment_id):
    ckpt = require(experiment_id)
    meta = CheckpointManager.read_metadata(ckpt)
    assert dict(meta["label2id"]) == dict(LABEL2ID)

@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_data_hashes_recorded(experiment_id):
    ckpt = require(experiment_id)
    meta = CheckpointManager.read_metadata(ckpt)
    for split in ("train", "validation"):
        assert meta["data_hashes"][split]["content_sha256"]

def test_all_experiments_share_the_same_training_data():
    if len(AVAILABLE) < 2:
        pytest.skip("need at least two checkpoints")
    hashes = {
        e: CheckpointManager.read_metadata(CHECKPOINT_DIR / e)["data_hashes"]["train"][
            "content_sha256"
        ]
        for e in AVAILABLE
    }
    assert len(set(hashes.values())) == 1, f"experiments trained on different data: {hashes}"

@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_checkpoint_loads_and_predicts(experiment_id):
    ckpt = require(experiment_id)
    from src.models.factory import load_model_from_checkpoint

    model, meta = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
    assert not model.training, "model must come back in eval mode"
    assert meta["model_type"] in ("bilstm", "phobert")

    from src.inference.predictor import PunctuationRestorationPredictor

    predictor = PunctuationRestorationPredictor.from_checkpoint(
        ckpt, device=torch.device("cpu")
    )
    words = "bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài".split()
    labels = predictor.predict_labels(words)

    assert len(labels) == len(words)
    assert all(l in LABELS for l in labels)

@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_checkpoint_restores_text_end_to_end(experiment_id):
    ckpt = require(experiment_id)
    from src.inference.predictor import PunctuationRestorationPredictor

    predictor = PunctuationRestorationPredictor.from_checkpoint(
        ckpt, device=torch.device("cpu")
    )
    result = predictor.restore("hôm nay trời đẹp bạn có muốn đi dạo không")
    assert result.restored_text
    assert result.num_words == 10
    assert result.experiment_id == experiment_id

    stripped = result.restored_text.lower().replace(",", "").replace(".", "").replace("?", "")
    assert stripped.split() == "hôm nay trời đẹp bạn có muốn đi dạo không".split()

def test_phobert_checkpoints_share_the_pinned_revision():
    phobert = [
        e for e in AVAILABLE
        if CheckpointManager.read_metadata(CHECKPOINT_DIR / e)["model_type"] == "phobert"
    ]
    if len(phobert) < 2:
        pytest.skip("need at least two PhoBERT checkpoints")
    revisions = {
        e: CheckpointManager.read_metadata(CHECKPOINT_DIR / e)["model_revision"]
        for e in phobert
    }
    assert len(set(revisions.values())) == 1, f"revisions differ: {revisions}"

def test_bilstm_checkpoint_ships_its_vocabulary():
    for e in AVAILABLE:
        meta = CheckpointManager.read_metadata(CHECKPOINT_DIR / e)
        if meta["model_type"] == "bilstm":
            assert (CHECKPOINT_DIR / e / "vocabulary.json").exists()
            from src.data.dataset import Vocabulary

            vocab = Vocabulary.load(CHECKPOINT_DIR / e / "vocabulary.json")
            assert vocab.source_split == "train"
            assert vocab.itos[0] == "<PAD>" and vocab.itos[1] == "<UNK>"
            return
    pytest.skip("no BiLSTM checkpoint present")

def test_phobert_checkpoint_is_a_loadable_hf_folder():
    for e in AVAILABLE:
        meta = CheckpointManager.read_metadata(CHECKPOINT_DIR / e)
        if meta["model_type"] == "phobert":
            transformers = pytest.importorskip("transformers")
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                CHECKPOINT_DIR / e
            )
            assert model.config.num_labels == 4
            tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT_DIR / e)
            assert tokenizer.tokenize("tiếng việt")
            return
    pytest.skip("no PhoBERT checkpoint present")

def test_locked_winner_checkpoint_loads():
    from src.evaluation.selection import (
        MODEL_SELECTION_PATH,
        load_locked_winner,
        resolve_winner_checkpoint,
    )

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present — run notebook 05")

    selection = load_locked_winner()
    ckpt = resolve_winner_checkpoint(selection)
    assert ckpt.exists()
    assert CheckpointManager.has_checkpoint(ckpt)

    meta = CheckpointManager.read_metadata(ckpt)
    assert meta["experiment_id"] == selection["winner"]
    assert meta["best_score"] == pytest.approx(
        selection["winner_validation_punctuation_macro_f1"], abs=1e-9
    )
