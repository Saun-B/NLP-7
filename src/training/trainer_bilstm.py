from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.constants import CHECKPOINT_DIR, EXPERIMENT_DIR, PAD_ID, SEED
from src.data.dataset import BiLSTMDataset, Vocabulary, collate_bilstm
from src.data.schema import Example
from src.models.factory import build_model, describe_model
from src.training.artifacts import ExperimentArtifactWriter
from src.training.checkpointing import CheckpointManager
from src.training.losses import build_loss, describe_loss, get_class_weight_vector
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.seed import make_generator, seed_worker, set_seed
from src.training.trainer_base import TrainerBase
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["BiLSTMTrainer", "build_bilstm_trainer", "build_bilstm_dataloaders"]

class BiLSTMTrainer(TrainerBase):
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = 0
        iterator = self._progress(
            self.train_loader, desc=f"{self.experiment_id} epoch {epoch} train"
        )

        for step, batch in enumerate(iterator, start=1):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            lengths = batch["lengths"]

            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask, lengths=lengths
            )
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            loss.backward()
            self._clip_gradients()
            self._optimizer_step()

            total_loss += float(loss.detach())
            num_batches += 1

            if self.log_every and step % self.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d — loss %.4f",
                    epoch,
                    step,
                    len(self.train_loader),
                    total_loss / num_batches,
                )
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        return total_loss / max(1, num_batches)

def build_bilstm_dataloaders(
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    vocab: Vocabulary,
    config: Dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders for E1."""
    training_cfg = config.get("training", {})
    seed = int(training_cfg.get("seed", SEED))
    num_workers = int(training_cfg.get("num_workers", 0))

    train_ds = BiLSTMDataset(train_examples, vocab)
    val_ds = BiLSTMDataset(validation_examples, vocab)

    def collate(batch):
        return collate_bilstm(batch, pad_id=PAD_ID)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("train_batch_size", 128)),
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg.get("eval_batch_size", 256)),
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader

def build_bilstm_trainer(
    config: Dict[str, Any],
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    *,
    vocab: Optional[Vocabulary] = None,
    data_hashes: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Any] = None,
    experiment_base_dir: Optional[Any] = None,
) -> tuple[BiLSTMTrainer, Vocabulary]:
    training_cfg = config.get("training", {})
    experiment_id = config.get("experiment", {}).get("id", "E1")
    seed = int(training_cfg.get("seed", SEED))
    set_seed(seed, deterministic=bool(training_cfg.get("deterministic", False)))

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vocab is None:
        vocab = Vocabulary.build(
            train_examples,
            min_freq=int(config.get("data", {}).get("vocab_min_freq", 1)),
            max_size=config.get("data", {}).get("vocab_max_size"),
            source_split="train",
        )

    train_loader, val_loader = build_bilstm_dataloaders(
        train_examples, validation_examples, vocab, config
    )

    model = build_model(config, vocab_size=len(vocab))
    weight_mode = str(config.get("loss", {}).get("weight_mode", "none"))
    weight_vector = get_class_weight_vector(weight_mode) if weight_mode != "none" else None
    loss_fn = build_loss(weight_mode, device=device)
    optimizer = build_optimizer(model, config)
    scheduler, step_mode = build_scheduler(optimizer, config, num_training_steps=None)

    artifacts = ExperimentArtifactWriter(
        experiment_id, base_dir=experiment_base_dir or EXPERIMENT_DIR
    )
    artifacts.write_environment()
    if data_hashes:
        artifacts.write_data_hashes(data_hashes)
    artifacts.write_config(
        config,
        extra={
            "model": describe_model(model),
            "loss": describe_loss(weight_mode, weight_vector),
            "vocabulary": {
                "size": len(vocab),
                "min_freq": vocab.min_freq,
                "built_from": vocab.source_split,
            },
            "num_train_examples": len(train_examples),
            "num_validation_examples": len(validation_examples),
        },
    )

    checkpoints = CheckpointManager(
        experiment_id,
        checkpoint_dir or (CHECKPOINT_DIR / experiment_id),
        model_type="bilstm",
        model_name="BiLSTM(emb=128, hidden=128x2)",
        model_revision=None,
        seed=seed,
        training_config=config,
        data_hashes=data_hashes or {},
        vocabulary=vocab,
    )

    trainer = BiLSTMTrainer(
        experiment_id=experiment_id,
        model=model,
        config=config,
        train_loader=train_loader,
        validation_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_mode=step_mode,
        device=device,
        checkpoint_manager=checkpoints,
        artifacts=artifacts,
        use_amp=False,
    )
    return trainer, vocab
