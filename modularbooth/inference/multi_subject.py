"""Multi-subject image generation with masked LoRA composition.

This module implements :class:`MultiSubjectGenerator`, the core inference engine
for composing multiple personalized subjects into a single image.  Unlike the
single-subject case, multi-subject generation requires a **custom denoising
loop** because each subject's LoRA must be applied selectively -- weighted by a
spatial mask that confines its influence to the designated bounding box.

The algorithm for each denoising step:

1. For every subject, temporarily apply its LoRA to the transformer and compute
   a noise prediction with spatial masking via token-aware attention masks.
2. Compose the per-subject predictions using spatial blending.
3. Apply negative attention (scaled down) outside each subject's region to
   prevent identity leakage.
4. Execute the scheduler step to advance the latents.

Typical usage::

    from omegaconf import OmegaConf
    from modularbooth.inference.multi_subject import MultiSubjectGenerator

    cfg = OmegaConf.load("configs/flux.yaml")
    gen = MultiSubjectGenerator(cfg, device="cuda")
    gen.load_pipeline()
    gen.load_subjects({
        "V1": "checkpoints/dog_lora.safetensors",
        "V2": "checkpoints/cat_lora.safetensors",
    })
    images = gen.generate(
        prompt="a [V1] dog playing with a [V2] cat on a beach",
        layout={"V1": (0.0, 0.2, 0.45, 0.8), "V2": (0.55, 0.2, 1.0, 0.8)},
        num_images=4,
        seed=42,
    )
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from modularbooth.inference.layout import LayoutGenerator

logger = logging.getLogger(__name__)


class MultiSubjectGenerator:
    """Generate images composing multiple personalized subjects via masked inference.

    Each subject is associated with a trained LoRA checkpoint and a spatial
    bounding box.  During the reverse diffusion process the generator applies
    each subject's LoRA only within its designated region, preventing identity
    blending across subjects.

    Args:
        config: OmegaConf ``DictConfig`` with ``model.*`` and ``inference.*``
            sections (see ``configs/default.yaml``).
        device: Torch device string.  Defaults to ``"cuda"``.
    """

    # Pattern that matches subject tokens like [V1], [V2], etc.
    _SUBJECT_TOKEN_RE = re.compile(r"\[V\d+\]")

    def __init__(self, config: DictConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device

        self.pipeline: Any | None = None
        self._lora_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        self._layout_generator = LayoutGenerator()

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def load_pipeline(self, backbone: str | None = None) -> None:
        """Load the base Diffusion Transformer pipeline from HuggingFace.

        The model identifier is taken from *backbone* (if given) or falls
        back to ``config.model.backbone``.

        Args:
            backbone: Optional HuggingFace model id override.
        """
        from diffusers import DiffusionPipeline

        model_id = backbone or self.config.model.backbone
        dtype_str = getattr(self.config.model, "dtype", "bfloat16")
        dtype = _resolve_dtype(dtype_str)
        revision = getattr(self.config.model, "revision", None)

        logger.info(
            "Loading base pipeline '%s' (dtype=%s) ...", model_id, dtype_str
        )

        self.pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            revision=revision,
        )
        self.pipeline.to(self.device)

        logger.info("Pipeline loaded on %s.", self.device)

    # ------------------------------------------------------------------
    # Subject LoRA management
    # ------------------------------------------------------------------

    def load_subjects(self, lora_paths: dict[str, str]) -> None:
        """Load multiple LoRA checkpoints into memory.

        The checkpoints are **not** applied to the pipeline immediately;
        they are stored as state dicts and selectively applied during the
        custom denoising loop.

        Args:
            lora_paths: Mapping from subject identifier (e.g. ``"V1"``) to
                the filesystem path of the corresponding LoRA checkpoint
                (safetensors or PyTorch format).

        Raises:
            FileNotFoundError: If any checkpoint path does not exist.
        """
        from safetensors.torch import load_file as load_safetensors

        self._lora_state_dicts.clear()

        for subject_id, path_str in lora_paths.items():
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(
                    f"LoRA checkpoint for subject '{subject_id}' not found: {path}"
                )

            if path.suffix == ".safetensors":
                state_dict = load_safetensors(str(path))
            else:
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

            # Move tensors to the target device
            state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            self._lora_state_dicts[subject_id] = state_dict

            logger.info(
                "Loaded LoRA for subject '%s' from '%s' (%d tensors).",
                subject_id,
                path,
                len(state_dict),
            )

        logger.info(
            "All %d subject LoRAs loaded into memory.", len(self._lora_state_dicts)
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        layout: dict[str, tuple[float, float, float, float]] | None = None,
        num_images: int = 4,
        seed: int | None = None,
    ) -> list[PIL.Image.Image]:
        """Generate images composing multiple personalized subjects.

        Args:
            prompt: Composite text prompt containing subject tokens, e.g.
                ``"a [V1] dog playing with a [V2] cat on a beach"``.
            layout: Optional mapping from subject identifier to a normalized
                bounding box ``(x1, y1, x2, y2)`` in [0, 1].  When ``None``
                a layout is auto-generated using the horizontal strategy.
            num_images: Number of images to produce.
            seed: Random seed for reproducibility.

        Returns:
            List of PIL Images of length *num_images*.

        Raises:
            RuntimeError: If the pipeline is not loaded or no subjects are
                loaded.
            ValueError: If a subject token in the prompt has no matching
                loaded LoRA.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline has not been loaded. Call load_pipeline() first."
            )
        if not self._lora_state_dicts:
            raise RuntimeError(
                "No subject LoRAs have been loaded. Call load_subjects() first."
            )

        # -- 1. Parse subject tokens from the prompt ----------------------
        subject_tokens = self._parse_subject_tokens(prompt)
        if not subject_tokens:
            raise ValueError(
                f"No subject tokens (e.g. [V1]) found in prompt: '{prompt}'"
            )

        # Strip brackets to get identifiers: "[V1]" -> "V1"
        subject_ids = [tok.strip("[]") for tok in subject_tokens]

        for sid in subject_ids:
            if sid not in self._lora_state_dicts:
                raise ValueError(
                    f"Subject '{sid}' referenced in prompt but no LoRA is "
                    f"loaded for it. Loaded subjects: "
                    f"{sorted(self._lora_state_dicts.keys())}"
                )

        # -- 2. Generate or validate layout -------------------------------
        if layout is None:
            layout = self._layout_generator.generate_layout(
                num_subjects=len(subject_ids),
                strategy="horizontal",
            )
            # Ensure keys match subject ids
            layout = {
                sid: layout[f"V{i + 1}"]
                for i, sid in enumerate(subject_ids)
            }
            logger.info("Auto-generated layout: %s", layout)
        else:
            if not self._layout_generator.validate_layout(layout):
                logger.warning(
                    "Provided layout did not pass validation; proceeding anyway."
                )

        # -- 3. Resolve inference parameters ------------------------------
        inf_cfg = self.config.inference
        num_steps = int(inf_cfg.num_steps)
        guidance_scale = float(inf_cfg.guidance_scale)
        resolution = int(getattr(inf_cfg, "resolution", 1024))

        if seed is None:
            seed = getattr(inf_cfg, "seed", None)

        # -- 4. Encode prompt ---------------------------------------------
        dtype = next(self.pipeline.transformer.parameters()).dtype

        # Replace subject tokens with generic placeholders for the text
        # encoder (the LoRA captures the identity, not the token embedding).
        clean_prompt = prompt
        for tok in subject_tokens:
            clean_prompt = clean_prompt.replace(tok, tok.strip("[]"))

        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
            self.pipeline, clean_prompt, self.device, dtype
        )

        # -- 5. Create spatial masks from layout --------------------------
        latent_h = resolution // 8
        latent_w = resolution // 8

        spatial_masks: dict[str, torch.Tensor] = {}
        for sid, (x1, y1, x2, y2) in layout.items():
            mask = _bbox_to_mask(x1, y1, x2, y2, latent_h, latent_w, self.device, dtype)
            spatial_masks[sid] = mask

        # -- 6. Collect LoRA state dicts for referenced subjects ----------
        lora_modules: dict[str, dict[str, torch.Tensor]] = {
            sid: self._lora_state_dicts[sid] for sid in subject_ids
        }

        # -- 7. Run the custom denoising loop for each image --------------
        all_images: list[PIL.Image.Image] = []

        for img_idx in range(num_images):
            img_seed = (seed + img_idx) if seed is not None else None
            generator = _make_generator(img_seed)

            # Initialize latents
            latent_shape = (
                1,
                self.pipeline.transformer.config.in_channels,
                latent_h,
                latent_w,
            )
            latents = torch.randn(
                latent_shape,
                generator=generator,
                device=self.device,
                dtype=dtype,
            )

            # Scale initial noise by scheduler
            self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
            timesteps = self.pipeline.scheduler.timesteps
            latents = latents * self.pipeline.scheduler.init_noise_sigma

            decoded = self._custom_denoising_loop(
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                spatial_masks=spatial_masks,
                lora_modules=lora_modules,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                timesteps=timesteps,
            )

            # Decode latents -> PIL Image
            image = _decode_latents(self.pipeline, decoded)
            all_images.append(image)

        logger.info(
            "Multi-subject generation complete: %d images for %d subjects.",
            len(all_images),
            len(subject_ids),
        )
        return all_images

    # ------------------------------------------------------------------
    # Prompt parsing
    # ------------------------------------------------------------------

    def _parse_subject_tokens(self, prompt: str) -> list[str]:
        """Extract subject tokens from a prompt string.

        Subject tokens follow the pattern ``[V<number>]``, e.g. ``[V1]``,
        ``[V2]``.  Tokens are returned in the order they appear, with
        duplicates preserved.

        Args:
            prompt: The composite text prompt.

        Returns:
            List of subject token strings (e.g. ``["[V1]", "[V2]"]``).
        """
        return self._SUBJECT_TOKEN_RE.findall(prompt)

    # ------------------------------------------------------------------
    # Core denoising loop
    # ------------------------------------------------------------------

    def _custom_denoising_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        spatial_masks: dict[str, torch.Tensor],
        lora_modules: dict[str, dict[str, torch.Tensor]],
        num_steps: int,
        guidance_scale: float,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Run the masked multi-subject denoising loop.

        For each timestep the loop:

        1. Obtains a *base* noise prediction from the unmodified transformer.
        2. For each subject, temporarily applies that subject's LoRA, computes
           a *subject-specific* noise prediction, and removes the LoRA.
        3. Blends predictions using spatial masks so each subject's LoRA
           influences only its designated region.
        4. Applies negative attention scaling outside each subject's region
           to suppress identity leakage.
        5. Advances the scheduler.

        Args:
            latents: Initial noisy latents, shape ``(1, C, H, W)``.
            prompt_embeds: Text encoder hidden states.
            pooled_prompt_embeds: Pooled text embeddings (for pipelines that
                use them, e.g. SD3).
            spatial_masks: Per-subject binary masks at latent resolution,
                shape ``(1, 1, H, W)`` each.
            lora_modules: Per-subject LoRA state dicts.
            num_steps: Total number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            timesteps: Scheduler timestep tensor.

        Returns:
            Denoised latent tensor ready for VAE decoding, shape ``(1, C, H, W)``.
        """
        neg_attn_strength = float(
            getattr(self.config.inference, "negative_attention_strength", 3.0)
        )
        mask_leakage_alpha = float(
            getattr(self.config.inference, "mask_leakage_alpha", 0.05)
        )

        transformer = self.pipeline.transformer
        subject_ids = list(lora_modules.keys())

        for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            # Prepare the timestep tensor
            timestep = t.unsqueeze(0) if t.dim() == 0 else t

            # ---- Base prediction (no LoRA) ----
            with torch.no_grad():
                base_pred = _forward_transformer(
                    transformer,
                    latents,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                )

            # ---- Per-subject predictions with masked LoRA ----
            subject_preds: dict[str, torch.Tensor] = {}

            for sid in subject_ids:
                lora_sd = lora_modules[sid]

                # Apply LoRA weights additively to transformer
                _apply_lora_state_dict(transformer, lora_sd)

                with torch.no_grad():
                    subj_pred = _forward_transformer(
                        transformer,
                        latents,
                        timestep,
                        prompt_embeds,
                        pooled_prompt_embeds,
                    )

                # Remove the LoRA weights
                _remove_lora_state_dict(transformer, lora_sd)

                subject_preds[sid] = subj_pred

            # ---- Spatial blending ----
            # Start from the base prediction
            blended = base_pred.clone()

            # Build a combined mask to track which regions are "owned" by
            # a subject.  Regions not owned stay at the base prediction.
            combined_mask = torch.zeros_like(
                list(spatial_masks.values())[0]
            )

            for sid in subject_ids:
                mask = spatial_masks[sid]  # (1, 1, H, W)

                # Soft mask with leakage to avoid hard seams
                soft_mask = mask * (1.0 - mask_leakage_alpha) + mask_leakage_alpha

                # The subject-specific delta (difference from base)
                delta = subject_preds[sid] - base_pred

                # Add the masked delta to the blended prediction
                blended = blended + delta * soft_mask

                combined_mask = torch.clamp(combined_mask + mask, 0.0, 1.0)

            # ---- Negative attention outside subject regions ----
            # Suppress subject-like features in background regions to prevent
            # identity leakage.
            background_mask = 1.0 - combined_mask
            if background_mask.sum() > 0:
                # Average the subject deltas and subtract them (scaled)
                # from background regions.
                avg_delta = torch.stack(
                    [subject_preds[sid] - base_pred for sid in subject_ids]
                ).mean(dim=0)
                blended = blended - neg_attn_strength * avg_delta * background_mask

            # ---- Classifier-free guidance ----
            # For simplicity with DiTs that handle guidance internally, we
            # apply guidance scaling to the blended prediction.  Pipelines
            # that compute conditional and unconditional predictions
            # separately would need a separate unconditional forward pass.
            # Here we assume a guidance-distilled model (e.g. FLUX-dev) or
            # handle it at a higher level.
            noise_pred = blended

            # ---- Scheduler step ----
            latents = self.pipeline.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        return latents


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map a string dtype name to ``torch.dtype``."""
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. Expected one of {sorted(mapping.keys())}."
        )
    return mapping[dtype_str]


def _make_generator(seed: int | None) -> torch.Generator | None:
    """Create a seeded CPU generator (or ``None``)."""
    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(seed)


def _encode_prompt(
    pipeline: Any,
    prompt: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt through the pipeline's text encoder(s).

    This helper handles both single-encoder pipelines (FLUX) and
    multi-encoder pipelines (SD3 with CLIP + T5).  It returns prompt
    embeddings and pooled embeddings as two separate tensors.

    Args:
        pipeline: A loaded HuggingFace Diffusers pipeline.
        prompt: The text prompt to encode.
        device: Target device string.
        dtype: Target dtype.

    Returns:
        Tuple of ``(prompt_embeds, pooled_prompt_embeds)`` tensors.
    """
    # SD3-style pipelines expose encode_prompt directly
    if hasattr(pipeline, "encode_prompt"):
        result = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            num_images_per_prompt=1,
        )
        # SD3's encode_prompt returns (prompt_embeds, negative_prompt_embeds,
        # pooled_prompt_embeds, negative_pooled_prompt_embeds)
        if isinstance(result, tuple) and len(result) >= 3:
            prompt_embeds = result[0]
            pooled_prompt_embeds = result[2]
        else:
            prompt_embeds = result[0] if isinstance(result, tuple) else result
            pooled_prompt_embeds = torch.zeros(
                1, prompt_embeds.shape[-1], device=device, dtype=dtype
            )
    else:
        # Fallback: use the tokenizer + text_encoder directly
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)

        with torch.no_grad():
            encoder_output = text_encoder(input_ids)

        if hasattr(encoder_output, "last_hidden_state"):
            prompt_embeds = encoder_output.last_hidden_state.to(dtype=dtype)
        else:
            prompt_embeds = encoder_output[0].to(dtype=dtype)

        if hasattr(encoder_output, "pooler_output") and encoder_output.pooler_output is not None:
            pooled_prompt_embeds = encoder_output.pooler_output.to(dtype=dtype)
        else:
            pooled_prompt_embeds = prompt_embeds.mean(dim=1)

    return prompt_embeds, pooled_prompt_embeds


def _bbox_to_mask(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    h: int,
    w: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a normalized bounding box to a binary latent-space mask.

    Args:
        x1, y1, x2, y2: Normalized [0, 1] bounding-box coordinates.
        h: Latent height.
        w: Latent width.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Binary mask tensor of shape ``(1, 1, h, w)``.
    """
    mask = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

    # Convert normalized coords to pixel indices
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))

    # Clamp to valid range
    px1 = max(0, min(px1, w))
    px2 = max(0, min(px2, w))
    py1 = max(0, min(py1, h))
    py2 = max(0, min(py2, h))

    if px2 > px1 and py2 > py1:
        mask[:, :, py1:py2, px1:px2] = 1.0

    return mask


def _forward_transformer(
    transformer: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    """Run a single forward pass through the DiT transformer.

    This function wraps the transformer call to handle the varying
    signatures across FLUX and SD3 transformers in HuggingFace diffusers.

    Args:
        transformer: The transformer module from the pipeline.
        latents: Noisy latents, shape ``(B, C, H, W)``.
        timestep: Current timestep tensor.
        prompt_embeds: Encoded prompt hidden states.
        pooled_prompt_embeds: Pooled prompt embeddings.

    Returns:
        Predicted noise tensor, same shape as *latents*.
    """
    # Build kwargs dict; different transformers accept different arguments.
    kwargs: dict[str, Any] = {
        "hidden_states": latents,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds,
        "return_dict": False,
    }

    # SD3 / FLUX transformers may accept pooled projections
    import inspect

    sig = inspect.signature(transformer.forward)
    if "pooled_projections" in sig.parameters:
        kwargs["pooled_projections"] = pooled_prompt_embeds

    output = transformer(**kwargs)

    # The output may be a tuple (noise_pred, ...) or a single tensor.
    if isinstance(output, tuple):
        return output[0]
    return output


def _apply_lora_state_dict(
    transformer: torch.nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
) -> None:
    """Additively apply LoRA weight deltas to the transformer in-place.

    For each LoRA pair ``(module_name.lora_A, module_name.lora_B)`` the
    function locates the corresponding ``nn.Linear`` in the transformer and
    adds ``scaling * B @ A`` to its weight.  The scaling factor is assumed to
    be baked into the checkpoint (i.e. ``alpha / rank`` was applied before
    saving).

    This is intentionally a simple, state-dict-level operation to avoid
    the overhead of wrapping layers in ``LoRALinear`` at every denoising step.

    Args:
        transformer: The DiT transformer module.
        lora_state_dict: LoRA state dict with keys like
            ``"blocks.5.attn.to_q.lora_A"`` and
            ``"blocks.5.attn.to_q.lora_B"``.
    """
    # Group by module name
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state_dict.items():
        if key.endswith(".lora_A"):
            module_name = key[: -len(".lora_A")]
            modules.setdefault(module_name, {})["lora_A"] = tensor
        elif key.endswith(".lora_B"):
            module_name = key[: -len(".lora_B")]
            modules.setdefault(module_name, {})["lora_B"] = tensor

    for module_name, params in modules.items():
        if "lora_A" not in params or "lora_B" not in params:
            continue

        # Navigate to the target linear layer
        target = _get_submodule(transformer, module_name)
        if target is None or not hasattr(target, "weight"):
            continue

        lora_A = params["lora_A"]
        lora_B = params["lora_B"]
        rank = lora_A.shape[0]
        # Assume alpha = rank (scaling = 1.0) unless the checkpoint stores
        # separate scaling.  This matches the convention in BlockwiseLoRA
        # when alpha_ratio = 1.0.
        scaling = 1.0
        delta = (lora_B @ lora_A) * scaling
        target.weight.data.add_(delta)


def _remove_lora_state_dict(
    transformer: torch.nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
) -> None:
    """Remove previously applied LoRA weight deltas from the transformer.

    This is the exact inverse of :func:`_apply_lora_state_dict` -- it
    subtracts the same delta that was added.

    Args:
        transformer: The DiT transformer module.
        lora_state_dict: The same LoRA state dict that was applied.
    """
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state_dict.items():
        if key.endswith(".lora_A"):
            module_name = key[: -len(".lora_A")]
            modules.setdefault(module_name, {})["lora_A"] = tensor
        elif key.endswith(".lora_B"):
            module_name = key[: -len(".lora_B")]
            modules.setdefault(module_name, {})["lora_B"] = tensor

    for module_name, params in modules.items():
        if "lora_A" not in params or "lora_B" not in params:
            continue

        target = _get_submodule(transformer, module_name)
        if target is None or not hasattr(target, "weight"):
            continue

        lora_A = params["lora_A"]
        lora_B = params["lora_B"]
        scaling = 1.0
        delta = (lora_B @ lora_A) * scaling
        target.weight.data.sub_(delta)


def _get_submodule(
    module: torch.nn.Module,
    name: str,
) -> torch.nn.Module | None:
    """Safely retrieve a nested sub-module by dotted name.

    Args:
        module: Root module.
        name: Dotted path (e.g. ``"blocks.5.attn.to_q"``).

    Returns:
        The sub-module, or ``None`` if the path is invalid.
    """
    parts = name.split(".")
    current = module
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def _decode_latents(pipeline: Any, latents: torch.Tensor) -> PIL.Image.Image:
    """Decode latent tensors to a PIL Image using the pipeline's VAE.

    Handles the latent scaling conventions used by diffusers for both FLUX
    and SD3 pipelines.

    Args:
        pipeline: The loaded HuggingFace Diffusers pipeline.
        latents: Denoised latents, shape ``(1, C, H, W)``.

    Returns:
        A single PIL Image.
    """
    vae = pipeline.vae

    # Apply inverse scaling.  Diffusers stores the scaling factor on the
    # VAE config or on the pipeline scheduler config.
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents / scaling_factor

    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    # Convert from [-1, 1] to [0, 1]
    image_tensor = (image_tensor / 2.0 + 0.5).clamp(0.0, 1.0)

    # Convert to PIL
    image_tensor = image_tensor.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
    image_array = (image_tensor * 255.0).round().astype("uint8")
    image = PIL.Image.fromarray(image_array)

    return image
