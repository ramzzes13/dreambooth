"""Learning rate schedulers with warmup support for ModularBooth training.

Provides a ``WarmupScheduler`` that wraps any standard PyTorch LR scheduler
and prepends a linear warmup phase, as well as a ``build_scheduler`` factory
that creates the appropriate scheduler from an OmegaConf config.

Supported schedule types:
    * ``"cosine"`` -- Cosine annealing (``CosineAnnealingLR``) with warmup.
    * ``"linear"`` -- Linear decay to zero with warmup.
    * ``"constant"`` -- Constant learning rate with warmup.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from omegaconf import DictConfig


class WarmupScheduler(_LRScheduler):
    """Wraps an arbitrary base scheduler and adds a linear warmup phase.

    During the first ``warmup_steps`` steps the learning rate ramps linearly
    from zero to the base scheduler's initial LR.  After warmup, the base
    scheduler takes over.

    Args:
        optimizer: Wrapped optimizer.
        base_scheduler: The underlying scheduler that controls the LR after
            warmup completes.  It should be constructed with the same
            optimizer and must **not** have been stepped yet.
        warmup_steps: Number of linear-warmup steps.  If 0, the warmup
            phase is skipped entirely and the base scheduler is used from
            the start.
        last_epoch: The index of the last epoch.  Passed through to the
            ``_LRScheduler`` base class.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {warmup_steps}"
            )
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps

        # Store base LRs *before* calling super().__init__ which will call
        # get_lr() and step() for the first time.
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Compute the learning rate for the current step.

        Returns:
            List of learning rates, one per parameter group.
        """
        if self.warmup_steps == 0 or self.last_epoch >= self.warmup_steps:
            return self.base_scheduler.get_lr()  # type: ignore[return-value]

        # Linear warmup: scale = current_step / warmup_steps
        warmup_factor = self.last_epoch / max(1, self.warmup_steps)
        return [base_lr * warmup_factor for base_lr in self._base_lrs]

    def step(self, epoch: int | None = None) -> None:
        """Advance the scheduler by one step.

        During warmup the base scheduler is not stepped.  After warmup
        completes, each call forwards the step to the base scheduler.

        Args:
            epoch: Optional explicit epoch index (passed to base class).
        """
        # Always call our own step to update last_epoch.
        super().step(epoch)

        # Only step the base scheduler once warmup is complete.
        if self.last_epoch >= self.warmup_steps:
            self.base_scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        state = super().state_dict()
        state["base_scheduler"] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint."""
        base_state = state_dict.pop("base_scheduler", None)
        super().load_state_dict(state_dict)
        if base_state is not None:
            self.base_scheduler.load_state_dict(base_state)


def _build_cosine_scheduler(
    optimizer: Optimizer,
    num_steps: int,
    warmup_steps: int,
) -> _LRScheduler:
    """Build a cosine annealing scheduler with linear warmup.

    After warmup, the learning rate decays following a cosine curve from
    the initial LR to zero over the remaining ``num_steps - warmup_steps``
    steps.

    Args:
        optimizer: The optimizer whose LR is scheduled.
        num_steps: Total number of training steps.
        warmup_steps: Number of linear-warmup steps.

    Returns:
        A ``WarmupScheduler`` wrapping ``CosineAnnealingLR``.
    """
    remaining_steps = max(1, num_steps - warmup_steps)
    base = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_steps
    )
    return WarmupScheduler(optimizer, base, warmup_steps=warmup_steps)


def _build_linear_scheduler(
    optimizer: Optimizer,
    num_steps: int,
    warmup_steps: int,
) -> _LRScheduler:
    """Build a linear decay scheduler with linear warmup.

    After warmup the learning rate decays linearly from the initial LR to
    zero over the remaining training steps.

    Args:
        optimizer: The optimizer whose LR is scheduled.
        num_steps: Total number of training steps.
        warmup_steps: Number of linear-warmup steps.

    Returns:
        A ``WarmupScheduler`` wrapping a ``LambdaLR``.
    """
    remaining_steps = max(1, num_steps - warmup_steps)

    def _linear_decay(current_step: int) -> float:
        """Return multiplicative factor for LambdaLR."""
        return max(0.0, 1.0 - current_step / remaining_steps)

    base = LambdaLR(optimizer, lr_lambda=_linear_decay)
    return WarmupScheduler(optimizer, base, warmup_steps=warmup_steps)


def _build_constant_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
) -> _LRScheduler:
    """Build a constant-LR scheduler with linear warmup.

    After warmup the learning rate stays at the initial value for the
    remainder of training.

    Args:
        optimizer: The optimizer whose LR is scheduled.
        warmup_steps: Number of linear-warmup steps.

    Returns:
        A ``WarmupScheduler`` wrapping a ``LambdaLR`` that always returns 1.
    """
    base = LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return WarmupScheduler(optimizer, base, warmup_steps=warmup_steps)


_SCHEDULER_BUILDERS = {
    "cosine": _build_cosine_scheduler,
    "linear": _build_linear_scheduler,
    "constant": _build_constant_scheduler,
}


def build_scheduler(
    optimizer: Optimizer,
    config: DictConfig,
) -> _LRScheduler:
    """Create a learning rate scheduler from an OmegaConf config.

    Reads the following config keys:

    * ``config.training.scheduler`` (str, default ``"cosine"``):
      Schedule type -- one of ``"cosine"``, ``"linear"``, ``"constant"``.
    * ``config.training.warmup_steps`` (int, default 0): Number of warmup
      steps for all schedule types.
    * ``config.training.num_steps`` (int): Total training steps (required
      for ``"cosine"`` and ``"linear"``).

    Args:
        optimizer: The optimizer to schedule.
        config: Top-level OmegaConf config (must contain a ``training``
            sub-key).

    Returns:
        The constructed LR scheduler.

    Raises:
        ValueError: If the scheduler type is not recognised.
    """
    schedule_type: str = getattr(config.training, "scheduler", "cosine")
    warmup_steps: int = getattr(config.training, "warmup_steps", 0)
    num_steps: int = config.training.num_steps

    if schedule_type not in _SCHEDULER_BUILDERS:
        raise ValueError(
            f"Unknown scheduler type '{schedule_type}'. "
            f"Supported types: {sorted(_SCHEDULER_BUILDERS.keys())}"
        )

    if schedule_type == "constant":
        return _build_constant_scheduler(optimizer, warmup_steps)
    else:
        return _SCHEDULER_BUILDERS[schedule_type](
            optimizer, num_steps, warmup_steps
        )
