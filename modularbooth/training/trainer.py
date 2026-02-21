"""Main training loop for ModularBooth multi-subject personalization.

This module implements :class:`ModularBoothTrainer`, which orchestrates the
full training pipeline for blockwise LoRA fine-tuning on Diffusion
Transformer (DiT) backbones such as FLUX.1-dev and Stable Diffusion 3.

The trainer handles:
    * Mixed-precision forward / backward passes.
    * Three loss components: diffusion denoising, prior-preservation (PPL),
      and contrastive context disentanglement (CCD).
    * Gradient accumulation and gradient clipping.
    * Callback-driven logging, checkpointing, and validation.
    * Checkpoint save / resume.

Diffuser-model interactions are isolated behind the :class:`ModelWrapper`
helper so that the trainer can be tested without loading a real 12B
parameter pipeline.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig

from modularbooth.training.callbacks import TrainingCallback
from modularbooth.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper -- thin abstraction over HuggingFace diffusers pipeline
# ---------------------------------------------------------------------------


class ModelWrapper:
    """Thin wrapper around a HuggingFace diffusers pipeline.

    Centralises all model interactions (VAE encoding, text encoding, noise
    prediction) so the rest of the trainer is pipeline-agnostic and easy to
    mock in tests.

    The wrapper auto-detects whether the pipeline exposes ``pipeline.unet``
    (UNet-based architectures) or ``pipeline.transformer`` (DiT-based
    architectures such as FLUX and SD3) and routes calls accordingly.

    Args:
        pipeline: A loaded ``diffusers`` pipeline instance (e.g.
            ``FluxPipeline`` or ``StableDiffusion3Pipeline``).
        device: Torch device string (e.g. ``"cuda"``).
        dtype: Torch dtype for inference (e.g. ``torch.bfloat16``).
    """

    def __init__(
        self,
        pipeline: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype

        # Resolve the denoising backbone.
        if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            self._denoiser = pipeline.transformer
        elif hasattr(pipeline, "unet") and pipeline.unet is not None:
            self._denoiser = pipeline.unet
        else:
            raise ValueError(
                "Pipeline has neither 'transformer' nor 'unet'. "
                "Cannot determine the denoising backbone."
            )

        # Resolve the noise scheduler.
        if hasattr(pipeline, "scheduler"):
            self._noise_scheduler = pipeline.scheduler
        else:
            raise ValueError("Pipeline has no 'scheduler' attribute.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def denoiser(self) -> nn.Module:
        """The underlying denoising model (transformer or UNet)."""
        return self._denoiser

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a pixel-space image into the VAE latent space.

        Args:
            image: Batch of images normalised to ``[-1, 1]``, shape
                ``(B, 3, H, W)``.

        Returns:
            Latent tensor of shape ``(B, C, h, w)`` scaled by the VAE
            scaling factor.
        """
        with torch.no_grad():
            image = image.to(device=self.device, dtype=self.dtype)
            latent_dist = self.pipeline.vae.encode(image).latent_dist
            latents = latent_dist.sample()

            # Apply the VAE's scaling factor.
            scaling_factor = getattr(
                self.pipeline.vae.config, "scaling_factor", 0.18215
            )
            latents = latents * scaling_factor

        return latents

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run the denoising backbone to predict noise.

        Args:
            noisy_latents: Noised latent tensor, shape ``(B, C, h, w)``.
            timestep: Integer timestep tensor, shape ``(B,)``.
            encoder_hidden_states: Text encoder outputs, shape
                ``(B, seq_len, hidden_dim)``.

        Returns:
            Predicted noise tensor with the same shape as *noisy_latents*.
        """
        noisy_latents = noisy_latents.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(
            device=self.device, dtype=self.dtype
        )

        model_output = self._denoiser(
            noisy_latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        # diffusers models return a dataclass; extract the sample tensor.
        if hasattr(model_output, "sample"):
            return model_output.sample
        return model_output

    def encode_text(self, prompt: str | list[str]) -> torch.Tensor:
        """Encode text prompt(s) into conditioning embeddings.

        Uses the pipeline's tokenizer and text encoder(s).  For multi-
        encoder pipelines (FLUX, SD3) this returns the concatenated /
        pooled output matching what the denoising backbone expects.

        Args:
            prompt: A single string or list of strings.

        Returns:
            Encoder hidden states tensor of shape
            ``(B, seq_len, hidden_dim)``.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # Most diffusers pipelines expose an encode_prompt helper.
        if hasattr(self.pipeline, "encode_prompt"):
            result = self.pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            # encode_prompt may return a tuple; take the first element.
            if isinstance(result, tuple):
                return result[0]
            return result

        # Fallback: manual tokenize + encode for single-encoder pipelines.
        tokenizer = self.pipeline.tokenizer
        text_encoder = self.pipeline.text_encoder
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        with torch.no_grad():
            encoder_output = text_encoder(input_ids)
        if hasattr(encoder_output, "last_hidden_state"):
            return encoder_output.last_hidden_state
        return encoder_output[0]

    def get_noise_scheduler(self) -> Any:
        """Return the pipeline's noise scheduler.

        Returns:
            The ``diffusers`` noise scheduler instance attached to the
            pipeline.
        """
        return self._noise_scheduler


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


class ModularBoothTrainer:
    """Main training loop for ModularBooth subject encoding.

    Trains blockwise LoRA modules on a frozen DiT backbone using three
    loss objectives:

    1. **Diffusion denoising loss** on subject images.
    2. **Prior-preservation loss (PPL)** on class images to prevent
       language drift.
    3. **Contrastive context disentanglement loss (CCD)** on intermediate
       DiT features to separate identity from background.

    Args:
        config: OmegaConf ``DictConfig`` containing the full experiment
            configuration (see ``configs/default.yaml``).
        model: A HuggingFace diffusers pipeline (or any object compatible
            with :class:`ModelWrapper`).  Alternatively, a pre-constructed
            ``ModelWrapper`` instance.
        lora: A ``BlockwiseLoRA`` module whose parameters will be
            optimized.
        dataset: A :class:`~modularbooth.data.dataset.DreamBoothDataset`
            providing interleaved subject and class samples.
        loss_fn: A ``ModularBoothLoss`` callable that combines diffusion,
            PPL, and CCD loss components.
        device: Torch device string (e.g. ``"cuda"``).
        callbacks: Optional list of :class:`TrainingCallback` instances.
    """

    def __init__(
        self,
        config: DictConfig,
        model: Any,
        lora: nn.Module,
        dataset: Dataset,
        loss_fn: nn.Module,
        device: str = "cuda",
        callbacks: list[TrainingCallback] | None = None,
    ) -> None:
        self.config = config
        self.lora = lora
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.device = device
        self.callbacks = callbacks or []

        # Wrap the pipeline if a raw diffusers pipeline was passed.
        if isinstance(model, ModelWrapper):
            self.model_wrapper = model
        else:
            dtype_str = getattr(config.model, "dtype", "bfloat16")
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(dtype_str, torch.bfloat16)
            self.model_wrapper = ModelWrapper(model, device=device, dtype=dtype)

        # ---- Optimizer -------------------------------------------------------
        # Only optimise LoRA parameters; the backbone stays frozen.
        lora_params = list(self.lora.parameters())
        if not lora_params:
            raise ValueError(
                "LoRA module has no parameters. Ensure LoRA layers have been "
                "injected into the model before constructing the trainer."
            )

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.training.learning_rate,
            weight_decay=getattr(config.training, "weight_decay", 0.01),
        )

        # ---- LR scheduler ---------------------------------------------------
        self.scheduler = build_scheduler(self.optimizer, config)

        # ---- Data loader -----------------------------------------------------
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,  # Keep it simple; images are small and few.
            drop_last=True,
            pin_memory=(device != "cpu"),
        )

        # ---- Mixed precision -------------------------------------------------
        mp_setting = getattr(config.training, "mixed_precision", "no")
        self._amp_enabled = mp_setting in ("bf16", "fp16")
        self._amp_dtype: torch.dtype = (
            torch.bfloat16 if mp_setting == "bf16" else torch.float16
        )
        # GradScaler is only needed for fp16; bf16 does not require it.
        self._use_grad_scaler = mp_setting == "fp16"
        self.scaler = GradScaler(enabled=self._use_grad_scaler)

        # ---- Training hyper-parameters (cached for convenience) ---------------
        self._gradient_accumulation: int = getattr(
            config.training, "gradient_accumulation", 1
        )
        self._max_grad_norm: float = getattr(config.training, "max_grad_norm", 1.0)
        self._num_steps: int = config.training.num_steps

        # ---- CCD settings ----------------------------------------------------
        self._ccd_enabled: bool = getattr(config, "ccd", {}).get("enabled", False)
        self._ccd_warmup: int = getattr(config, "ccd", {}).get("warmup_steps", 0)

        # ---- Feature extraction hook handle -----------------------------------
        self._hook_features: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary of final aggregated metrics (averaged losses, etc.).
        """
        # Freeze backbone; only LoRA is trainable.
        self.model_wrapper.denoiser.requires_grad_(False)
        self.lora.train()

        # Seed RNG for reproducibility.
        seed = getattr(self.config.training, "seed", None)
        if seed is not None:
            torch.manual_seed(seed)

        # Notify callbacks.
        for cb in self.callbacks:
            cb.on_train_begin(self)

        # Infinite data iterator (cycles the dataloader).
        data_iter = self._infinite_dataloader()

        # Running metric accumulators.
        running_metrics: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        global_step = 0
        self.optimizer.zero_grad()

        while global_step < self._num_steps:
            # ----- Gradient accumulation inner loop -----------------------
            accumulated_logs: dict[str, float] = {}
            for _accum_idx in range(self._gradient_accumulation):
                batch = next(data_iter)
                step_logs = self._training_step(batch, global_step)

                # Accumulate logs (average later).
                for key, val in step_logs.items():
                    if isinstance(val, (int, float)):
                        accumulated_logs[key] = (
                            accumulated_logs.get(key, 0.0) + val
                        )

            # Average over accumulation steps.
            for key in accumulated_logs:
                accumulated_logs[key] /= self._gradient_accumulation

            # ----- Gradient clipping --------------------------------------
            if self._use_grad_scaler:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.lora.parameters(), self._max_grad_norm
            )
            accumulated_logs["grad_norm"] = grad_norm.item()

            # ----- Optimizer + scheduler step -----------------------------
            if self._use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            global_step += 1

            # Current LR.
            accumulated_logs["lr"] = self.optimizer.param_groups[0]["lr"]

            # Update running averages.
            for key, val in accumulated_logs.items():
                if isinstance(val, (int, float)):
                    running_metrics[key] = running_metrics.get(key, 0.0) + val
                    metric_counts[key] = metric_counts.get(key, 0) + 1

            # ----- Callbacks -------------------------------------------
            for cb in self.callbacks:
                cb.on_step_end(self, global_step, accumulated_logs)

        # ---- End of training -------------------------------------------------
        final_metrics: dict[str, Any] = {}
        for key in running_metrics:
            count = metric_counts.get(key, 1)
            final_metrics[key] = running_metrics[key] / max(count, 1)
        final_metrics["total_steps"] = global_step

        for cb in self.callbacks:
            cb.on_train_end(self, final_metrics)

        return final_metrics

    def save_checkpoint(self, path: str | Path, global_step: int) -> None:
        """Save a training checkpoint.

        The checkpoint contains:
            * ``lora_state_dict`` -- LoRA module weights.
            * ``optimizer_state_dict`` -- Optimizer state.
            * ``scheduler_state_dict`` -- LR scheduler state.
            * ``scaler_state_dict`` -- Mixed-precision grad scaler state.
            * ``global_step`` -- The step at which the checkpoint was taken.
            * ``config`` -- The training configuration (for reproducibility).

        Args:
            path: Directory in which to save the checkpoint files.
            global_step: The current training step.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "lora_state_dict": self.lora.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": global_step,
            "config": dict(self.config),
        }
        torch.save(checkpoint, path / "checkpoint.pt")
        logger.info("Saved checkpoint at step %d to %s", global_step, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a training checkpoint and restore all state.

        Args:
            path: Directory containing a ``checkpoint.pt`` file.

        Returns:
            The ``global_step`` at which the checkpoint was saved, so the
            caller can resume the training loop from the correct step.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        ckpt_file = path / "checkpoint.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        checkpoint = torch.load(ckpt_file, map_location=self.device)

        self.lora.load_state_dict(checkpoint["lora_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        global_step: int = checkpoint["global_step"]
        logger.info("Restored checkpoint from step %d (%s)", global_step, path)
        return global_step

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------

    def _training_step(
        self, batch: dict[str, Any], global_step: int
    ) -> dict[str, float]:
        """Execute a single training step (forward + backward).

        This method:
            1. Encodes subject and class images to latents.
            2. Samples random timesteps and adds noise.
            3. Predicts noise via the DiT backbone (with LoRA active).
            4. Computes diffusion denoising loss on subject images.
            5. Computes prior-preservation loss on class images.
            6. Optionally computes CCD loss on intermediate features.
            7. Runs the backward pass (scaled by gradient accumulation).

        Args:
            batch: A dict from the dataloader with keys ``pixel_values``,
                ``input_ids``, ``is_class_image``, and optionally
                ``augmented_pixel_values``.
            global_step: Current (0-based) global step for warmup checks.

        Returns:
            Dictionary of scalar loss values for this step.
        """
        logs: dict[str, float] = {}

        pixel_values: torch.Tensor = batch["pixel_values"].to(self.device)
        captions: list[str] = batch["input_ids"]  # strings from dataset
        is_class: torch.Tensor = batch["is_class_image"]  # bool tensor

        batch_size = pixel_values.shape[0]

        # ---- AMP context ----------------------------------------------
        amp_ctx = (
            autocast(device_type=self.device.split(":")[0], dtype=self._amp_dtype)
            if self._amp_enabled
            else nullcontext()
        )

        with amp_ctx:
            # ---- Encode images to latents --------------------------------
            latents = self.model_wrapper.encode_image(pixel_values)

            # ---- Encode text prompts ------------------------------------
            encoder_hidden_states = self.model_wrapper.encode_text(captions)

            # ---- Sample timesteps and add noise --------------------------
            noise_scheduler = self.model_wrapper.get_noise_scheduler()
            noise = torch.randn_like(latents)
            num_train_timesteps = getattr(
                noise_scheduler.config, "num_train_timesteps", 1000
            )
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ---- (Optional) register feature-extraction hook for CCD -----
            ccd_active = (
                self._ccd_enabled and global_step >= self._ccd_warmup
            )
            if ccd_active:
                self._register_feature_hook()

            # ---- Predict noise -------------------------------------------
            noise_pred = self.model_wrapper.predict_noise(
                noisy_latents, timesteps, encoder_hidden_states
            )

            # ---- Compute losses ------------------------------------------
            # Build boolean masks for subject vs class samples.
            is_class_bool = is_class.bool().to(self.device)
            is_subject_bool = ~is_class_bool

            total_loss = torch.tensor(0.0, device=self.device)

            # 1. Diffusion denoising loss on subject images.
            if is_subject_bool.any():
                subject_loss = F.mse_loss(
                    noise_pred[is_subject_bool],
                    noise[is_subject_bool],
                )
                total_loss = total_loss + subject_loss
                logs["loss_diffusion"] = subject_loss.item()

            # 2. Prior-preservation loss on class images.
            ppl_enabled = getattr(self.config, "prior_preservation", {}).get(
                "enabled", False
            )
            if ppl_enabled and is_class_bool.any():
                lambda_ppl: float = getattr(
                    self.config, "prior_preservation", {}
                ).get("lambda_ppl", 1.0)
                ppl_loss = F.mse_loss(
                    noise_pred[is_class_bool],
                    noise[is_class_bool],
                )
                total_loss = total_loss + lambda_ppl * ppl_loss
                logs["loss_ppl"] = ppl_loss.item()

            # 3. CCD loss on intermediate features.
            if ccd_active:
                ccd_loss_val = self._compute_ccd_loss(
                    batch, latents, timesteps, encoder_hidden_states
                )
                if ccd_loss_val is not None:
                    lambda_ccd: float = getattr(
                        self.config, "ccd", {}
                    ).get("lambda_ccd", 0.3)
                    total_loss = total_loss + lambda_ccd * ccd_loss_val
                    logs["loss_ccd"] = ccd_loss_val.item()

                # Clean up hooks.
                self._remove_feature_hooks()

            logs["loss_total"] = total_loss.item()

        # ---- Backward pass (scaled for gradient accumulation) ----------
        scaled_loss = total_loss / self._gradient_accumulation
        if self._use_grad_scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return logs

    # ------------------------------------------------------------------
    # Feature extraction for CCD
    # ------------------------------------------------------------------

    def _extract_intermediate_features(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Extract intermediate features from a specified DiT layer via hooks.

        Registers a forward hook on the target layer, runs a forward pass,
        and returns the captured activation tensor.

        Args:
            model: The denoising backbone (transformer / UNet).
            latents: Noised latent tensor, shape ``(B, C, h, w)``.
            timestep: Timestep tensor, shape ``(B,)``.
            encoder_hidden_states: Text conditioning tensor.

        Returns:
            Intermediate feature tensor captured by the hook.  Shape
            depends on the target layer (typically ``(B, hidden_dim, h, w)``
            or ``(B, seq_len, hidden_dim)``).

        Raises:
            RuntimeError: If the hook did not capture any features (e.g.
                the target layer name is incorrect).
        """
        self._hook_features.clear()

        target_layer = self._resolve_feature_layer(model)
        handle = target_layer.register_forward_hook(self._feature_hook_fn)

        try:
            with torch.no_grad():
                _ = self.model_wrapper.predict_noise(
                    latents, timestep, encoder_hidden_states
                )
        finally:
            handle.remove()

        if "features" not in self._hook_features:
            raise RuntimeError(
                "Feature extraction hook did not capture any activations. "
                "Check the 'ccd.feature_layer' config setting."
            )
        return self._hook_features["features"]

    def _feature_hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        """Forward hook that caches the layer output."""
        if isinstance(output, torch.Tensor):
            self._hook_features["features"] = output
        elif isinstance(output, tuple) and len(output) > 0:
            self._hook_features["features"] = output[0]
        else:
            self._hook_features["features"] = output

    def _register_feature_hook(self) -> None:
        """Register a persistent feature extraction hook on the target layer."""
        self._hook_features.clear()
        denoiser = self.model_wrapper.denoiser
        target_layer = self._resolve_feature_layer(denoiser)
        handle = target_layer.register_forward_hook(self._feature_hook_fn)
        self._hook_handles.append(handle)

    def _remove_feature_hooks(self) -> None:
        """Remove all registered feature extraction hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _resolve_feature_layer(self, model: nn.Module) -> nn.Module:
        """Resolve the config-specified feature layer to an nn.Module.

        Supports:
            * ``"middle"`` -- selects the middle block of the backbone.
            * A dot-separated attribute path (e.g.
              ``"transformer_blocks.9"``).

        Args:
            model: The denoising backbone.

        Returns:
            The target ``nn.Module`` layer.

        Raises:
            ValueError: If the layer specification cannot be resolved.
        """
        feature_layer_spec: str = getattr(
            self.config, "ccd", {}
        ).get("feature_layer", "middle")

        if feature_layer_spec == "middle":
            # Heuristic: pick the middle transformer block.
            # Try common attribute names across diffusers architectures.
            for attr_name in (
                "transformer_blocks",
                "joint_transformer_blocks",
                "single_transformer_blocks",
                "blocks",
                "mid_block",
            ):
                blocks = getattr(model, attr_name, None)
                if blocks is not None and isinstance(blocks, nn.ModuleList):
                    mid_idx = len(blocks) // 2
                    return blocks[mid_idx]
                if blocks is not None and isinstance(blocks, nn.Module):
                    return blocks

            # Fallback: return the model itself (hook on top-level forward).
            logger.warning(
                "Could not resolve 'middle' feature layer; "
                "hooking on the top-level denoiser module."
            )
            return model

        # Dot-separated attribute path.
        current: Any = model
        for part in feature_layer_spec.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part, None)
                if current is None:
                    raise ValueError(
                        f"Cannot resolve feature layer path "
                        f"'{feature_layer_spec}': attribute '{part}' not found."
                    )
        if not isinstance(current, nn.Module):
            raise ValueError(
                f"Resolved feature layer is not an nn.Module: "
                f"got {type(current)}"
            )
        return current

    def _compute_ccd_loss(
        self,
        batch: dict[str, Any],
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute the CCD contrastive loss using intermediate features.

        Requires that subject samples in the batch contain
        ``augmented_pixel_values`` (background-swapped variants of the
        subject image).  If no augmented images are available, returns
        ``None``.

        The method:
            1. Extracts features captured by the hook from the main forward
               pass (subject = anchor).
            2. Encodes augmented variants and runs forward passes to
               extract positive features.
            3. Uses class-image features as negatives.
            4. Delegates to ``self.loss_fn`` for the actual InfoNCE
               computation.

        Args:
            batch: The current training batch.
            latents: Clean latents for the batch.
            timesteps: Sampled timesteps for the batch.
            encoder_hidden_states: Text-conditioning tensors.

        Returns:
            Scalar CCD loss, or ``None`` if augmented images are not
            available in this batch.
        """
        # Check if augmented images exist in this batch.
        if "augmented_pixel_values" not in batch:
            return None

        augmented_pixel_values = batch["augmented_pixel_values"]

        # We need at least one augmented variant to form a positive pair.
        # augmented_pixel_values is a list of tensors (one per augmented variant)
        # batched from the dataloader, so its structure depends on collation.
        if augmented_pixel_values is None:
            return None

        is_class_bool = batch["is_class_image"].bool().to(self.device)
        is_subject_bool = ~is_class_bool

        if not is_subject_bool.any():
            return None

        # Subject features from the main forward pass (captured by hook).
        if "features" not in self._hook_features:
            return None
        all_features = self._hook_features["features"]

        # Select subject features as anchors.
        # Handle both (B, C, H, W) and (B, seq_len, hidden_dim) shapes.
        subject_features = all_features[is_subject_bool]
        if subject_features.ndim == 4:
            # Global average pool: (B_s, C, H, W) -> (B_s, C)
            subject_features = subject_features.mean(dim=(2, 3))
        elif subject_features.ndim == 3:
            # Mean pool over sequence: (B_s, seq_len, D) -> (B_s, D)
            subject_features = subject_features.mean(dim=1)

        # Extract positive features from augmented images.
        # For simplicity, use the first augmented variant per subject.
        if isinstance(augmented_pixel_values, (list, tuple)):
            # Each element is a tensor for one augmented variant across the batch.
            aug_images = augmented_pixel_values[0]
        elif isinstance(augmented_pixel_values, torch.Tensor):
            # Shape might be (B, num_aug, C, H, W); take first variant.
            if augmented_pixel_values.ndim == 5:
                aug_images = augmented_pixel_values[:, 0]
            else:
                aug_images = augmented_pixel_values
        else:
            return None

        aug_images = aug_images.to(self.device)
        # Only keep subject-sample augmented images.
        if aug_images.shape[0] == all_features.shape[0]:
            aug_images = aug_images[is_subject_bool]

        aug_latents = self.model_wrapper.encode_image(aug_images)
        noise_scheduler = self.model_wrapper.get_noise_scheduler()
        aug_noise = torch.randn_like(aug_latents)
        # Use same timesteps for the subject samples.
        subject_timesteps = timesteps[is_subject_bool]
        aug_noisy = noise_scheduler.add_noise(
            aug_latents, aug_noise, subject_timesteps
        )
        subject_enc_states = encoder_hidden_states[is_subject_bool]

        aug_features = self._extract_intermediate_features(
            self.model_wrapper.denoiser,
            aug_noisy,
            subject_timesteps,
            subject_enc_states,
        )
        if aug_features.ndim == 4:
            positive_features = aug_features.mean(dim=(2, 3))
        elif aug_features.ndim == 3:
            positive_features = aug_features.mean(dim=1)
        else:
            positive_features = aug_features

        # Negative features from class images.
        if is_class_bool.any():
            negative_features = all_features[is_class_bool]
            if negative_features.ndim == 4:
                negative_features = negative_features.mean(dim=(2, 3))
            elif negative_features.ndim == 3:
                negative_features = negative_features.mean(dim=1)
            # Reshape to (B_subject, N_neg, D).
            n_neg = negative_features.shape[0]
            dim = negative_features.shape[-1]
            n_subject = subject_features.shape[0]
            # Broadcast: repeat negatives for each subject sample.
            negative_features = (
                negative_features.unsqueeze(0)
                .expand(n_subject, n_neg, dim)
                .contiguous()
            )
        else:
            # No class images in this batch; use zero negatives.
            dim = subject_features.shape[-1]
            n_subject = subject_features.shape[0]
            negative_features = torch.zeros(
                n_subject, 1, dim, device=self.device
            )

        # Delegate to loss_fn (expected to be CCDLoss or ModularBoothLoss
        # with a .ccd_loss attribute).
        ccd_loss_module = self.loss_fn
        if hasattr(ccd_loss_module, "ccd_loss"):
            ccd_loss_module = ccd_loss_module.ccd_loss

        ccd_loss = ccd_loss_module(
            subject_features, positive_features, negative_features
        )
        return ccd_loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _infinite_dataloader(self):
        """Yield batches from the dataloader indefinitely (cycling).

        Yields:
            Batch dicts from the underlying dataloader, cycling when the
            dataset is exhausted.
        """
        while True:
            for batch in self.dataloader:
                yield batch
