"""Training callbacks for ModularBooth.

Provides a base callback interface and concrete implementations for:
    * **Logging** -- periodic console / W&B logging of loss, LR, grad norm.
    * **Checkpointing** -- periodic LoRA checkpoint saving.
    * **Validation** -- periodic image generation for visual monitoring.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from modularbooth.training.trainer import ModularBoothTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base callback
# ---------------------------------------------------------------------------

class TrainingCallback:
    """Base class for training callbacks.

    Subclasses override one or more of the hook methods below.  The trainer
    calls these hooks at the appropriate points in the training loop.
    """

    def on_train_begin(self, trainer: "ModularBoothTrainer") -> None:
        """Called once at the start of training, before the first step.

        Args:
            trainer: The trainer instance.
        """

    def on_step_end(
        self,
        trainer: "ModularBoothTrainer",
        global_step: int,
        logs: dict[str, Any],
    ) -> None:
        """Called at the end of every training step.

        Args:
            trainer: The trainer instance.
            global_step: Current 1-based global step index.
            logs: Dictionary of scalar metrics produced by the step (loss
                components, learning rate, gradient norm, etc.).
        """

    def on_train_end(
        self,
        trainer: "ModularBoothTrainer",
        logs: dict[str, Any],
    ) -> None:
        """Called once at the end of training after the last step.

        Args:
            trainer: The trainer instance.
            logs: Summary metrics for the full training run.
        """


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LoggingCallback(TrainingCallback):
    """Logs training metrics to the console and optionally to Weights & Biases.

    Metrics logged every ``log_every`` steps include all loss components
    (diffusion loss, PPL, CCD), the current learning rate, and the
    global gradient norm.

    Args:
        log_every: Logging frequency in steps.
        use_wandb: If ``True``, additionally log scalars to W&B.  The
            caller is responsible for calling ``wandb.init()`` before
            training begins.
    """

    def __init__(self, log_every: int = 10, use_wandb: bool = False) -> None:
        super().__init__()
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        self.log_every = log_every
        self.use_wandb = use_wandb
        self._step_start_time: float = 0.0
        self._train_start_time: float = 0.0

    # -- hooks ---------------------------------------------------------------

    def on_train_begin(self, trainer: "ModularBoothTrainer") -> None:
        """Record the start time of training."""
        self._train_start_time = time.monotonic()
        self._step_start_time = self._train_start_time
        logger.info(
            "Training started -- %d steps, batch_size=%d, grad_accum=%d",
            trainer.config.training.num_steps,
            trainer.config.training.batch_size,
            trainer.config.training.gradient_accumulation,
        )

    def on_step_end(
        self,
        trainer: "ModularBoothTrainer",
        global_step: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics to console and W&B at the configured frequency."""
        if global_step % self.log_every != 0:
            return

        now = time.monotonic()
        elapsed = now - self._step_start_time
        steps_per_sec = self.log_every / max(elapsed, 1e-9)
        self._step_start_time = now

        # Build a tidy log line.
        parts: list[str] = [f"step={global_step}"]
        for key in sorted(logs.keys()):
            value = logs[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.5f}")
            else:
                parts.append(f"{key}={value}")
        parts.append(f"steps/s={steps_per_sec:.2f}")
        logger.info("  ".join(parts))

        # W&B logging.
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import-untyped]

                wandb_logs = {**logs, "steps_per_sec": steps_per_sec}
                wandb.log(wandb_logs, step=global_step)
            except ImportError:
                logger.warning(
                    "use_wandb=True but wandb is not installed. "
                    "Skipping W&B logging."
                )
                self.use_wandb = False  # Disable further attempts.

    def on_train_end(
        self,
        trainer: "ModularBoothTrainer",
        logs: dict[str, Any],
    ) -> None:
        """Log training summary."""
        total_time = time.monotonic() - self._train_start_time
        logger.info(
            "Training complete -- total time: %.1fs, final metrics: %s",
            total_time,
            {k: f"{v:.5f}" if isinstance(v, float) else v for k, v in logs.items()},
        )


# ---------------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------------

class CheckpointCallback(TrainingCallback):
    """Saves LoRA checkpoints at a regular interval.

    Checkpoints are written to ``{output_dir}/step_{global_step}/`` and
    contain the LoRA weights, optimizer state, and the current step number
    so training can be resumed.

    Args:
        save_every: Checkpoint frequency in steps.
        output_dir: Root directory for checkpoints.  A sub-directory is
            created for each checkpoint.
        max_checkpoints: Maximum number of checkpoints to keep on disk.
            Older checkpoints are deleted when the limit is exceeded.
            Set to ``0`` or ``None`` for unlimited retention.
    """

    def __init__(
        self,
        save_every: int = 200,
        output_dir: str | Path = "./outputs/checkpoints",
        max_checkpoints: int = 0,
    ) -> None:
        super().__init__()
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self._saved_paths: list[Path] = []

    def on_train_begin(self, trainer: "ModularBoothTrainer") -> None:
        """Ensure the output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        trainer: "ModularBoothTrainer",
        global_step: int,
        logs: dict[str, Any],
    ) -> None:
        """Save a checkpoint if the current step is a checkpoint step."""
        if global_step % self.save_every != 0:
            return

        ckpt_dir = self.output_dir / f"step_{global_step}"
        trainer.save_checkpoint(ckpt_dir, global_step)
        self._saved_paths.append(ckpt_dir)
        logger.info("Checkpoint saved: %s", ckpt_dir)

        self._maybe_cleanup()

    def on_train_end(
        self,
        trainer: "ModularBoothTrainer",
        logs: dict[str, Any],
    ) -> None:
        """Save a final checkpoint at the end of training."""
        num_steps = trainer.config.training.num_steps
        final_dir = self.output_dir / f"step_{num_steps}_final"
        trainer.save_checkpoint(final_dir, num_steps)
        self._saved_paths.append(final_dir)
        logger.info("Final checkpoint saved: %s", final_dir)

    # -- internal ------------------------------------------------------------

    def _maybe_cleanup(self) -> None:
        """Remove old checkpoints when ``max_checkpoints`` is exceeded."""
        if self.max_checkpoints <= 0:
            return
        while len(self._saved_paths) > self.max_checkpoints:
            old_path = self._saved_paths.pop(0)
            if old_path.exists():
                import shutil

                shutil.rmtree(old_path)
                logger.info("Removed old checkpoint: %s", old_path)


# ---------------------------------------------------------------------------
# Validation callback
# ---------------------------------------------------------------------------

class ValidationCallback(TrainingCallback):
    """Generates validation images at a regular interval for visual monitoring.

    Uses the current LoRA weights to run inference on a set of validation
    prompts.  Generated images are saved to disk and optionally logged to
    W&B.

    Args:
        validate_every: Validation frequency in steps.
        prompts: List of text prompts to generate images for.
        output_dir: Directory where validation images are saved.
        num_inference_steps: Number of diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Optional RNG seed for reproducible validation images.
        use_wandb: If ``True``, log generated images to W&B.
    """

    def __init__(
        self,
        validate_every: int = 100,
        prompts: list[str] | None = None,
        output_dir: str | Path = "./outputs/images",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        use_wandb: bool = False,
    ) -> None:
        super().__init__()
        if validate_every < 1:
            raise ValueError(f"validate_every must be >= 1, got {validate_every}")
        self.validate_every = validate_every
        self.prompts = prompts or ["a [V] object"]
        self.output_dir = Path(output_dir)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.use_wandb = use_wandb

    def on_train_begin(self, trainer: "ModularBoothTrainer") -> None:
        """Ensure the output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        trainer: "ModularBoothTrainer",
        global_step: int,
        logs: dict[str, Any],
    ) -> None:
        """Generate and save validation images at the configured interval."""
        if global_step % self.validate_every != 0:
            return

        logger.info("Running validation at step %d ...", global_step)

        images = self._generate_images(trainer, global_step)

        if images is not None and self.use_wandb:
            self._log_to_wandb(images, global_step)

    # -- internal ------------------------------------------------------------

    @torch.no_grad()
    def _generate_images(
        self,
        trainer: "ModularBoothTrainer",
        global_step: int,
    ) -> list[Any] | None:
        """Generate validation images using the current model state.

        The LoRA modules are kept active so the generated images reflect
        the latest training state.  The model is set to eval mode for
        generation and restored to train mode afterwards.

        Args:
            trainer: The trainer instance, providing access to the model
                wrapper and device.
            global_step: Current step (used for naming saved images).

        Returns:
            List of PIL images, or ``None`` if generation failed.
        """
        model_wrapper = trainer.model_wrapper

        # Attempt to use the underlying pipeline for end-to-end generation.
        pipeline = getattr(model_wrapper, "pipeline", None)
        if pipeline is None:
            logger.warning(
                "Model wrapper has no 'pipeline' attribute; "
                "skipping validation image generation."
            )
            return None

        # Set eval mode on the LoRA parameters.
        trainer.lora.eval()

        try:
            all_images: list[Any] = []
            generator = (
                torch.Generator(device=trainer.device).manual_seed(self.seed)
                if self.seed is not None
                else None
            )

            for prompt_idx, prompt in enumerate(self.prompts):
                output = pipeline(
                    prompt=prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                )
                for img_idx, image in enumerate(output.images):
                    save_path = (
                        self.output_dir
                        / f"step_{global_step}_prompt_{prompt_idx}_img_{img_idx}.png"
                    )
                    image.save(save_path)
                    all_images.append(image)
                    logger.info("Saved validation image: %s", save_path)

            return all_images

        except Exception:
            logger.exception(
                "Validation image generation failed at step %d", global_step
            )
            return None

        finally:
            # Restore training mode.
            trainer.lora.train()

    def _log_to_wandb(self, images: list[Any], global_step: int) -> None:
        """Log generated images to W&B.

        Args:
            images: List of PIL images.
            global_step: Current step for the W&B log entry.
        """
        try:
            import wandb  # type: ignore[import-untyped]

            wandb_images = [
                wandb.Image(img, caption=f"step {global_step}, prompt {i}")
                for i, img in enumerate(images)
            ]
            wandb.log({"validation_images": wandb_images}, step=global_step)
        except ImportError:
            logger.warning(
                "use_wandb=True but wandb is not installed. "
                "Skipping W&B image logging."
            )
            self.use_wandb = False
