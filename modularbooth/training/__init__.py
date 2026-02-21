"""Training pipeline for ModularBooth.

Provides the main training loop, learning rate schedulers with warmup,
and callback-driven logging, checkpointing, and validation.
"""

from modularbooth.training.trainer import ModularBoothTrainer, ModelWrapper
from modularbooth.training.scheduler import build_scheduler, WarmupScheduler
from modularbooth.training.callbacks import (
    TrainingCallback,
    CheckpointCallback,
    LoggingCallback,
    ValidationCallback,
)

__all__ = [
    "ModularBoothTrainer",
    "ModelWrapper",
    "build_scheduler",
    "WarmupScheduler",
    "TrainingCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "ValidationCallback",
]
