from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.constants import NUM_LABELS, PAD_ID

__all__ = ["BiLSTMConfig", "BiLSTMTagger"]

@dataclass
class BiLSTMConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.30
    num_labels: int = NUM_LABELS
    padding_idx: int = PAD_ID

    def to_dict(self) -> dict:
        return {
            "architecture": "BiLSTM",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size_per_direction": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_labels": self.num_labels,
            "padding_idx": self.padding_idx,
            "classifier_input_dim": 2 * self.hidden_size,
        }

class BiLSTMTagger(nn.Module):
    def __init__(self, config: BiLSTMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> torch.Tensor:
        """Return logits ``(batch, seq_len, num_labels)``."""
        embedded = self.embedding(input_ids)

        if lengths is None and attention_mask is not None:
            lengths = attention_mask.sum(dim=1)

        if lengths is not None:
            lengths_cpu = lengths.detach().to("cpu", torch.int64).clamp(min=1)
            packed = pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            hidden, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            hidden, _ = self.lstm(embedded)

        return self.classifier(self.dropout(hidden))

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )
