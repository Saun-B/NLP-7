from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.data.constants import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    PHOBERT_MAX_LENGTH,
    PHOBERT_MODEL_NAME,
    PHOBERT_REVISION,
)
from src.utils.environment import disable_progress_bars
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["PhoBERTConfig", "PhoBERTTokenClassifier", "load_phobert_tokenizer"]

@dataclass
class PhoBERTConfig:
    model_name: str = PHOBERT_MODEL_NAME
    revision: str = PHOBERT_REVISION
    num_labels: int = NUM_LABELS
    max_length: int = PHOBERT_MAX_LENGTH
    dropout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": "RobertaForTokenClassification",
            "model_name": self.model_name,
            "revision": self.revision,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "dropout_override": self.dropout,
        }

def load_phobert_tokenizer(
    model_name: str = PHOBERT_MODEL_NAME, revision: str = PHOBERT_REVISION
):
    disable_progress_bars()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, revision=revision)

class PhoBERTTokenClassifier(nn.Module):
    def __init__(self, config: PhoBERTConfig):
        super().__init__()
        disable_progress_bars()
        from transformers import AutoConfig, AutoModelForTokenClassification

        self.config = config

        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            revision=config.revision,
            num_labels=config.num_labels,
            id2label={int(k): v for k, v in ID2LABEL.items()},
            label2id=dict(LABEL2ID),
        )
        if config.dropout is not None:
            hf_config.hidden_dropout_prob = config.dropout
            hf_config.attention_probs_dropout_prob = config.dropout
        if getattr(hf_config, "classifier_dropout", None) is None:
            hf_config.classifier_dropout = hf_config.hidden_dropout_prob

        self.backbone = AutoModelForTokenClassification.from_pretrained(
            config.model_name, revision=config.revision, config=hf_config
        )
        logger.info(
            "Loaded %s @ %s (%.1fM parameters, num_labels=%d)",
            config.model_name,
            config.revision[:12],
            self.num_parameters() / 1e6,
            config.num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

    def save_pretrained(self, save_directory) -> None:
        """Save in HF format so inference code can ``from_pretrained`` the folder."""
        self.backbone.save_pretrained(save_directory)

    @classmethod
    def from_pretrained_dir(cls, directory, *, config: Optional[PhoBERTConfig] = None):
        """Rebuild a wrapper around a locally saved HF checkpoint."""
        disable_progress_bars()
        from transformers import AutoModelForTokenClassification

        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = config or PhoBERTConfig()
        obj.backbone = AutoModelForTokenClassification.from_pretrained(directory)
        return obj

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def gradient_checkpointing_enable(self) -> None:
        self.backbone.gradient_checkpointing_enable()
