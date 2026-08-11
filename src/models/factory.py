"""Build a model from an experiment config dict.

Keeps the notebooks free of ``if model_type == ...`` branching: each notebook
loads its YAML config and calls :func:`build_model`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch.nn as nn

from src.data.constants import NUM_LABELS, PHOBERT_MAX_LENGTH, PHOBERT_MODEL_NAME, PHOBERT_REVISION
from src.models.bilstm import BiLSTMConfig, BiLSTMTagger
from src.models.phobert import PhoBERTConfig, PhoBERTTokenClassifier
from src.utils.environment import disable_progress_bars
from src.utils.io import read_json

__all__ = ["build_model", "describe_model", "load_model_from_checkpoint"]

MODEL_TYPES = ("bilstm", "phobert")


def build_model(config: Dict[str, Any], *, vocab_size: Optional[int] = None) -> nn.Module:
    """Instantiate the model described by ``config['model']``.

    Parameters
    ----------
    config
        Parsed experiment YAML.
    vocab_size
        Required for ``bilstm`` - the vocabulary is built from the training
        split at runtime, so its size is not knowable from the YAML.
    """
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "")).lower()

    if model_type == "bilstm":
        if vocab_size is None:
            raise ValueError("build_model(bilstm) requires vocab_size=<int>")
        return BiLSTMTagger(
            BiLSTMConfig(
                vocab_size=vocab_size,
                embedding_dim=int(model_cfg.get("embedding_dim", 128)),
                hidden_size=int(model_cfg.get("hidden_size", 128)),
                num_layers=int(model_cfg.get("num_layers", 1)),
                dropout=float(model_cfg.get("dropout", 0.30)),
                num_labels=int(model_cfg.get("num_labels", NUM_LABELS)),
            )
        )

    if model_type == "phobert":
        return PhoBERTTokenClassifier(
            PhoBERTConfig(
                model_name=str(model_cfg.get("name", PHOBERT_MODEL_NAME)),
                revision=str(model_cfg.get("revision", PHOBERT_REVISION)),
                num_labels=int(model_cfg.get("num_labels", NUM_LABELS)),
                max_length=int(model_cfg.get("max_length", PHOBERT_MAX_LENGTH)),
                dropout=model_cfg.get("dropout"),
            )
        )

    raise ValueError(f"Unknown model type {model_type!r}; expected one of {MODEL_TYPES}")


def load_model_from_checkpoint(
    checkpoint_dir: Union[str, Path], *, device: Optional[Any] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the saved best model from a checkpoint folder.

    Returns ``(model_in_eval_mode, checkpoint_metadata)``. The training
    notebooks use it to show sample predictions from the **best** epoch rather
    than from whatever the last epoch happened to be.
    """
    import torch

    directory = Path(checkpoint_dir)
    meta_path = directory / "checkpoint_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint_metadata.json in {directory}")
    meta = read_json(meta_path)

    model_type = meta.get("model_type")
    if model_type == "phobert":
        disable_progress_bars()
        model = PhoBERTTokenClassifier.from_pretrained_dir(directory)
    elif model_type == "bilstm":
        weights = directory / "model.pt"
        if not weights.exists():
            raise FileNotFoundError(f"No model.pt in {directory}")
        try:
            payload = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(weights, map_location="cpu", weights_only=False)
        cfg = payload.get("model_config", {})
        model = BiLSTMTagger(
            BiLSTMConfig(
                vocab_size=int(cfg["vocab_size"]),
                embedding_dim=int(cfg.get("embedding_dim", 128)),
                hidden_size=int(cfg.get("hidden_size_per_direction", 128)),
                num_layers=int(cfg.get("num_layers", 1)),
                dropout=float(cfg.get("dropout", 0.30)),
                num_labels=int(cfg.get("num_labels", NUM_LABELS)),
            )
        )
        model.load_state_dict(payload["state_dict"])
    else:
        raise ValueError(f"Unknown model_type {model_type!r} in {meta_path}")

    if device is not None:
        model.to(device)
    model.eval()
    return model, meta


def describe_model(model: nn.Module) -> Dict[str, Any]:
    """Parameter counts + the model's own config, for ``config.json``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    desc: Dict[str, Any] = {
        "class": type(model).__name__,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "total_parameters_millions": round(total / 1e6, 3),
    }
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        desc["model_config"] = cfg.to_dict()
    return desc


