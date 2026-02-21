"""Single-subject image generation with personalized LoRA.

This module provides :class:`SingleSubjectGenerator`, a high-level interface
for generating images that contain a single personalized subject.  It wraps a
HuggingFace Diffusers pipeline (FLUX or SD3), loads a trained LoRA checkpoint,
and exposes convenient ``generate`` / ``generate_batch`` methods.

Typical usage::

    from omegaconf import OmegaConf
    from modularbooth.inference.single_subject import SingleSubjectGenerator

    cfg = OmegaConf.load("configs/flux.yaml")
    gen = SingleSubjectGenerator(cfg, device="cuda")
    gen.load_pipeline()
    gen.load_subject("outputs/checkpoints/lora.safetensors", token="[V]", class_noun="dog")
    images = gen.generate("a [V] dog on a beach", num_images=4, seed=42)
"""

from __future__ import annotations

import logging
from typing import Any

import PIL.Image
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SingleSubjectGenerator:
    """Generate images containing a single personalized subject.

    The generator wraps a HuggingFace Diffusers ``DiffusionPipeline`` and
    applies a trained LoRA checkpoint to inject subject identity.

    Args:
        config: OmegaConf ``DictConfig`` with at least ``model.*`` and
            ``inference.*`` sections (see ``configs/default.yaml``).
        device: Torch device string.  Defaults to ``"cuda"``.
    """

    def __init__(self, config: DictConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device

        self.pipeline: Any | None = None
        self._lora_loaded: bool = False
        self._subject_token: str | None = None
        self._class_noun: str | None = None

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def load_pipeline(self, backbone: str | None = None) -> None:
        """Load the base Diffusion Transformer pipeline from HuggingFace.

        The backbone model identifier is resolved in the following order:

        1. The *backbone* argument (if provided).
        2. ``config.model.backbone``.

        The pipeline is moved to *self.device* with the dtype specified in
        ``config.model.dtype``.

        Args:
            backbone: Optional HuggingFace model identifier that overrides
                the config value (e.g. ``"black-forest-labs/FLUX.1-dev"``).
        """
        from diffusers import DiffusionPipeline

        model_id = backbone or self.config.model.backbone
        dtype_str = getattr(self.config.model, "dtype", "bfloat16")
        dtype = _resolve_dtype(dtype_str)
        revision = getattr(self.config.model, "revision", None)

        logger.info("Loading pipeline '%s' (dtype=%s) ...", model_id, dtype_str)

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

    def load_subject(
        self,
        lora_path: str,
        token: str = "[V]",
        class_noun: str = "object",
    ) -> None:
        """Load a trained LoRA checkpoint and apply it to the pipeline.

        If a previous subject LoRA is loaded it is unloaded first so that
        adapters do not stack.

        Args:
            lora_path: Path to a safetensors or PyTorch LoRA checkpoint
                compatible with the ``diffusers`` PEFT integration.
            token: The rare-token identifier used during training (e.g.
                ``"[V]"``).
            class_noun: Natural-language class noun (e.g. ``"dog"``).
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline has not been loaded. Call load_pipeline() first."
            )

        # Unload any existing adapter before loading a new one.
        if self._lora_loaded:
            self.unload_subject()

        logger.info("Loading subject LoRA from '%s' (token=%s) ...", lora_path, token)

        self.pipeline.load_lora_weights(lora_path)

        self._lora_loaded = True
        self._subject_token = token
        self._class_noun = class_noun

        logger.info("Subject LoRA applied successfully.")

    def unload_subject(self) -> None:
        """Remove the current LoRA adapter from the pipeline.

        After calling this method the pipeline reverts to the base model
        weights and is ready for a different subject to be loaded.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline has not been loaded. Call load_pipeline() first."
            )

        if not self._lora_loaded:
            logger.warning("No LoRA is currently loaded; nothing to unload.")
            return

        self.pipeline.unload_lora_weights()

        self._lora_loaded = False
        self._subject_token = None
        self._class_noun = None

        logger.info("Subject LoRA removed from pipeline.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        num_images: int = 4,
        seed: int | None = None,
        guidance_scale: float | None = None,
        num_steps: int | None = None,
    ) -> list[PIL.Image.Image]:
        """Generate images for a single prompt.

        Parameters that are left as ``None`` are resolved from the
        ``inference`` section of the config.

        Args:
            prompt: Text prompt (should contain the subject token, e.g.
                ``"a [V] dog on a beach"``).
            num_images: Number of images to produce.
            seed: Random seed for reproducibility.  ``None`` uses the config
                default; if that is also ``None`` a non-deterministic seed is
                used.
            guidance_scale: Classifier-free guidance scale.
            num_steps: Number of denoising steps.

        Returns:
            List of PIL Images of length *num_images*.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline has not been loaded. Call load_pipeline() first."
            )

        # Resolve defaults from config
        inf_cfg = self.config.inference
        guidance_scale = guidance_scale if guidance_scale is not None else inf_cfg.guidance_scale
        num_steps = num_steps if num_steps is not None else inf_cfg.num_steps
        resolution = getattr(inf_cfg, "resolution", 1024)

        if seed is None:
            seed = getattr(inf_cfg, "seed", None)

        # Build generator for reproducibility
        generator = _make_generator(seed, self.device)

        logger.info(
            "Generating %d image(s) | steps=%d | cfg=%.1f | seed=%s",
            num_images,
            num_steps,
            guidance_scale,
            seed,
        )

        result = self.pipeline(
            prompt=prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution,
            generator=generator,
        )

        images: list[PIL.Image.Image] = list(result.images)
        return images

    def generate_batch(
        self,
        prompts: list[str],
        num_images_per_prompt: int = 4,
        seed: int | None = None,
    ) -> dict[str, list[PIL.Image.Image]]:
        """Generate images for multiple prompts.

        Each prompt is processed independently so that per-prompt seed
        incrementing produces reproducible results.

        Args:
            prompts: List of text prompts.
            num_images_per_prompt: Number of images to generate per prompt.
            seed: Base random seed.  For the *i*-th prompt the effective seed
                is ``seed + i`` (when *seed* is not ``None``).

        Returns:
            Dictionary mapping each prompt string to its list of generated
            PIL Images.
        """
        results: dict[str, list[PIL.Image.Image]] = {}

        for idx, prompt in enumerate(prompts):
            prompt_seed = (seed + idx) if seed is not None else None
            images = self.generate(
                prompt=prompt,
                num_images=num_images_per_prompt,
                seed=prompt_seed,
            )
            results[prompt] = images

        logger.info(
            "Batch generation complete: %d prompts, %d total images.",
            len(prompts),
            sum(len(v) for v in results.values()),
        )
        return results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map a string dtype name to the corresponding ``torch.dtype``.

    Args:
        dtype_str: One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Returns:
        The matching ``torch.dtype``.

    Raises:
        ValueError: If the string is not recognised.
    """
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


def _make_generator(
    seed: int | None,
    device: str,
) -> torch.Generator | None:
    """Create a seeded ``torch.Generator`` or return ``None`` for non-deterministic sampling.

    Args:
        seed: Random seed, or ``None`` for non-deterministic.
        device: Torch device string.

    Returns:
        A seeded generator, or ``None``.
    """
    if seed is None:
        return None
    # Generators on CUDA must be created on "cpu" and then the pipeline
    # handles device placement internally in diffusers.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return gen
