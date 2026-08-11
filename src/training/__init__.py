"""Training: seeding, losses, optimizer/scheduler, trainers, checkpoints."""

from src.training.artifacts import (
    EpochRecord,
    ExperimentArtifactWriter,
    STATUS_COMPLETED,
    STATUS_NOT_RUN,
)
from src.training.checkpointing import CheckpointManager, CheckpointMetadata
from src.training.losses import build_loss, get_class_weight_vector, load_class_weights
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.seed import make_generator, seed_worker, set_seed
from src.training.trainer_base import TrainerBase, TrainingResult
from src.training.trainer_bilstm import BiLSTMTrainer, build_bilstm_trainer
from src.training.trainer_phobert import PhoBERTTrainer, build_phobert_trainer

__all__ = [
    "set_seed",
    "seed_worker",
    "make_generator",
    "build_loss",
    "load_class_weights",
    "get_class_weight_vector",
    "build_optimizer",
    "build_scheduler",
    "TrainerBase",
    "TrainingResult",
    "BiLSTMTrainer",
    "build_bilstm_trainer",
    "PhoBERTTrainer",
    "build_phobert_trainer",
    "CheckpointManager",
    "CheckpointMetadata",
    "ExperimentArtifactWriter",
    "EpochRecord",
    "STATUS_NOT_RUN",
    "STATUS_COMPLETED",
]
