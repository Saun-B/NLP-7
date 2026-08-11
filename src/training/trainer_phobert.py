from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.constants import CHECKPOINT_DIR, EXPERIMENT_DIR, PHOBERT_MAX_LENGTH, SEED
from src.data.dataset import Example, PhoBERTDataset, PhoBERTEncoder, collate_phobert
from src.models.factory import build_model, describe_model
from src.models.phobert import load_phobert_tokenizer
from src.training.artifacts import ExperimentArtifactWriter
from src.training.checkpointing import CheckpointManager
from src.training.losses import build_loss, describe_loss, get_class_weight_vector
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.seed import make_generator, seed_worker, set_seed
from src.training.trainer_base import TrainerBase
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["PhoBERTTrainer", "build_phobert_trainer", "build_phobert_dataloaders"]

class PhoBERTTrainer(TrainerBase):
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum = self.gradient_accumulation_steps
        total_loss = 0.0
        num_micro_batches = 0
        pending = False

        iterator = self._progress(
            self.train_loader, desc=f"{self.experiment_id} epoch {epoch} train"
        )

        for step, batch in enumerate(iterator, start=1):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(
                    logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1)
                )

            if not torch.isfinite(loss):
                logger.warning("epoch %d step %d: non-finite loss, skipping batch", epoch, step)
                continue

            scaled = loss / accum
            if self.use_amp:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            pending = True

            total_loss += float(loss.detach())
            num_micro_batches += 1

            if step % accum == 0:
                self._clip_gradients()
                self._optimizer_step()
                pending = False

            if self.log_every and step % self.log_every == 0:
                mean = total_loss / max(1, num_micro_batches)
                logger.info(
                    "epoch %d step %d/%d — loss %.4f (lr=%.2e)",
                    epoch,
                    step,
                    len(self.train_loader),
                    mean,
                    self.optimizer.param_groups[0]["lr"],
                )
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss=f"{mean:.4f}")

        if pending:
            self._clip_gradients()
            self._optimizer_step()

        return total_loss / max(1, num_micro_batches)

def build_phobert_dataloaders(
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    tokenizer,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Encode both splits into subword windows and wrap them in dataloaders."""
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    seed = int(training_cfg.get("seed", SEED))
    max_length = int(model_cfg.get("max_length", PHOBERT_MAX_LENGTH))
    num_workers = int(training_cfg.get("num_workers", 0))

    encoder = PhoBERTEncoder(tokenizer, max_length=max_length)
    train_ds = PhoBERTDataset(train_examples, encoder)
    val_ds = PhoBERTDataset(validation_examples, encoder)

    pad_id = encoder.pad_id

    def collate(batch):
        return collate_phobert(batch, pad_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("train_batch_size", 4)),
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
        batch_size=int(training_cfg.get("eval_batch_size", 8)),
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    encoding_stats = {
        "train": train_ds.stats.to_dict(),
        "validation": val_ds.stats.to_dict(),
        "max_length": max_length,
        "tokenizer_cache_size": encoder.cache_size,
    }
    return train_loader, val_loader, encoding_stats

def build_phobert_trainer(
    config: Dict[str, Any],
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    *,
    tokenizer=None,
    data_hashes: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Any] = None,
    experiment_base_dir: Optional[Any] = None,
) -> Tuple[PhoBERTTrainer, Any]:
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    experiment_id = config.get("experiment", {}).get("id", "E2")
    seed = int(training_cfg.get("seed", SEED))
    set_seed(seed, deterministic=bool(training_cfg.get("deterministic", False)))

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = load_phobert_tokenizer(
            model_cfg.get("name", "vinai/phobert-base-v2"),
            model_cfg.get("revision"),
        )

    train_loader, val_loader, encoding_stats = build_phobert_dataloaders(
        train_examples, validation_examples, tokenizer, config
    )

    model = build_model(config)

    weight_mode = str(config.get("loss", {}).get("weight_mode", "none"))
    weight_vector = get_class_weight_vector(weight_mode)
    loss_fn = build_loss(weight_mode, device=device)

    optimizer = build_optimizer(model, config)

    accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * int(training_cfg.get("epochs", 1))
    scheduler, step_mode = build_scheduler(optimizer, config, num_training_steps=total_steps)

    use_amp = bool(training_cfg.get("fp16", True)) and device.type == "cuda"
    if training_cfg.get("fp16", True) and device.type != "cuda":
        logger.info("fp16 requested but no CUDA device — falling back to fp32.")

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
            "encoding": encoding_stats,
            "num_train_examples": len(train_examples),
            "num_validation_examples": len(validation_examples),
            "optimizer_steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "effective_batch_size": int(training_cfg.get("train_batch_size", 4)) * accum,
            "fp16_enabled": use_amp,
        },
    )

    checkpoints = CheckpointManager(
        experiment_id,
        checkpoint_dir or (CHECKPOINT_DIR / experiment_id),
        model_type="phobert",
        model_name=str(model_cfg.get("name", "vinai/phobert-base-v2")),
        model_revision=model_cfg.get("revision"),
        seed=seed,
        training_config=config,
        data_hashes=data_hashes or {},
        tokenizer=tokenizer,
    )

    trainer = PhoBERTTrainer(
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
        use_amp=use_amp,
    )
    return trainer, tokenizer
