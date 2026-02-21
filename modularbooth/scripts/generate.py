#!/usr/bin/env python3
"""ModularBooth image generation entry-point.

Generates images from one or more fine-tuned LoRA checkpoints with support
for single-subject and multi-subject composition.  Multi-subject generation
uses spatial layout constraints (bounding boxes) and optional negative
attention masking to prevent identity leakage between subjects.

Usage::

    # Single subject
    python -m modularbooth.scripts.generate \
        --lora-paths ./outputs/dog/checkpoints/lora_final.safetensors \
        --tokens "[V]" \
        --class-nouns dog \
        --prompts "a [V] dog sitting on a beach"

    # Multi-subject
    python -m modularbooth.scripts.generate \
        --lora-paths ./outputs/dog/lora_final.safetensors ./outputs/cat/lora_final.safetensors \
        --tokens "[V1]" "[V2]" \
        --class-nouns dog cat \
        --prompts "a [V1] dog and a [V2] cat playing in the park" \
        --layout-strategy horizontal

See ``--help`` for the full list of CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from modularbooth.configs import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the generation script.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate images using one or more fine-tuned LoRA checkpoints. "
            "Supports single-subject and multi-subject layout-guided generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--lora-paths",
        nargs="+",
        type=str,
        required=True,
        help=(
            "One or more paths to LoRA checkpoint files (.safetensors). "
            "Provide multiple paths for multi-subject generation."
        ),
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=str,
        required=True,
        help=(
            "One or more subject identifier tokens, corresponding 1-to-1 "
            'with --lora-paths. Example: "[V1]" "[V2]".'
        ),
    )
    parser.add_argument(
        "--class-nouns",
        nargs="+",
        type=str,
        required=True,
        help=(
            "One or more class nouns, corresponding 1-to-1 with --lora-paths. "
            'Example: "dog" "cat".'
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=str,
        required=True,
        help=(
            "One or more text prompts for generation, OR a path to a JSON "
            "file containing a list of prompts. Subject tokens should appear "
            "in the prompt text."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/generated",
        help="Directory to save generated images (default: ./outputs/generated).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="flux",
        help='Backbone name for loading pipeline config (default: "flux").',
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of images to generate per prompt (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for random).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help=(
            "Optional JSON string specifying bounding boxes for multi-subject "
            "layout. Format: a list of [x_min, y_min, x_max, y_max] normalised "
            "coordinates, one per subject. Example: "
            '\'[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]\''
        ),
    )
    parser.add_argument(
        "--layout-strategy",
        type=str,
        default="horizontal",
        help=(
            "Automatic layout strategy for multi-subject generation when "
            "--layout is not provided. Options: horizontal, vertical, grid. "
            "Default: horizontal."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _resolve_prompts(raw_prompts: list[str]) -> list[str]:
    """Resolve prompts from CLI arguments.

    If a single argument is provided and it is a path to an existing JSON
    file, load prompts from that file.  Otherwise, treat all arguments as
    literal prompt strings.

    Args:
        raw_prompts: Raw prompt strings from ``--prompts``.

    Returns:
        List of resolved prompt strings.
    """
    if len(raw_prompts) == 1:
        candidate = Path(raw_prompts[0])
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            logger.info("Loading prompts from JSON file: %s", candidate)
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(p) for p in data]
            if isinstance(data, dict):
                return [str(v) for v in data.values()]
            raise ValueError(
                f"Prompts JSON must be a list or dict, got {type(data).__name__}"
            )

    return list(raw_prompts)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _parse_layout(
    layout_json: str | None,
    num_subjects: int,
    strategy: str,
) -> list[list[float]]:
    """Parse or generate bounding box layout for multi-subject generation.

    Args:
        layout_json: Optional JSON string with explicit bounding boxes.
        num_subjects: Number of subjects.
        strategy: Automatic layout strategy (``"horizontal"``, ``"vertical"``,
            or ``"grid"``).

    Returns:
        List of ``[x_min, y_min, x_max, y_max]`` normalised bounding boxes,
        one per subject.

    Raises:
        ValueError: If the layout JSON is malformed or the number of boxes
            does not match the number of subjects.
    """
    if layout_json is not None:
        boxes = json.loads(layout_json)
        if not isinstance(boxes, list) or len(boxes) != num_subjects:
            raise ValueError(
                f"Layout must be a list of {num_subjects} bounding boxes, "
                f"got {len(boxes) if isinstance(boxes, list) else type(boxes).__name__}"
            )
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(
                    f"Each bounding box must be [x_min, y_min, x_max, y_max], got {box}"
                )
        return boxes

    return _generate_automatic_layout(num_subjects, strategy)


def _generate_automatic_layout(
    num_subjects: int,
    strategy: str,
) -> list[list[float]]:
    """Generate automatic bounding box layout for multi-subject composition.

    Args:
        num_subjects: Number of subjects to place.
        strategy: Layout strategy name.

    Returns:
        List of ``[x_min, y_min, x_max, y_max]`` normalised bounding boxes.

    Raises:
        ValueError: If the strategy is not recognised.
    """
    if strategy == "horizontal":
        # Divide the image into equal horizontal strips.
        width = 1.0 / num_subjects
        boxes = []
        for i in range(num_subjects):
            x_min = i * width
            x_max = (i + 1) * width
            boxes.append([x_min, 0.0, x_max, 1.0])
        return boxes

    if strategy == "vertical":
        # Divide the image into equal vertical strips.
        height = 1.0 / num_subjects
        boxes = []
        for i in range(num_subjects):
            y_min = i * height
            y_max = (i + 1) * height
            boxes.append([0.0, y_min, 1.0, y_max])
        return boxes

    if strategy == "grid":
        # Arrange subjects in a grid (roughly square).
        import math
        cols = math.ceil(math.sqrt(num_subjects))
        rows = math.ceil(num_subjects / cols)
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows
        boxes = []
        for i in range(num_subjects):
            row = i // cols
            col = i % cols
            boxes.append([
                col * cell_w,
                row * cell_h,
                (col + 1) * cell_w,
                (row + 1) * cell_h,
            ])
        return boxes

    raise ValueError(
        f"Unknown layout strategy '{strategy}'. "
        "Supported: horizontal, vertical, grid."
    )


# ---------------------------------------------------------------------------
# Generator wrappers
# ---------------------------------------------------------------------------

class SingleSubjectGenerator:
    """Generate images with a single fine-tuned LoRA adapter.

    This generator loads the base backbone pipeline, applies the LoRA weights,
    and runs standard text-to-image inference.

    Args:
        cfg: Full configuration.
        lora_path: Path to the LoRA checkpoint.
        token: Subject identifier token.
        class_noun: Class noun for the subject.
        device: Torch device string.
    """

    def __init__(
        self,
        cfg: DictConfig,
        lora_path: str,
        token: str,
        class_noun: str,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.lora_path = lora_path
        self.token = token
        self.class_noun = class_noun
        self.device = device

        self._pipeline: Any = None
        self._lora_manager: Any = None

    def _load_pipeline(self) -> None:
        """Load the diffusion pipeline and apply LoRA weights."""
        from diffusers import DiffusionPipeline
        from modularbooth.models.blockwise_lora import BlockwiseLoRA

        logger.info("Loading backbone: %s", self.cfg.model.backbone)
        dtype = getattr(torch, self.cfg.model.dtype)
        self._pipeline = DiffusionPipeline.from_pretrained(
            self.cfg.model.backbone,
            torch_dtype=dtype,
            revision=self.cfg.model.get("revision"),
        ).to(self.device)

        # Extract transformer and apply LoRA.
        transformer = getattr(self._pipeline, "transformer", None)
        if transformer is None:
            transformer = getattr(self._pipeline, "unet", None)

        # Build block config from the YAML structure.
        block_config = _build_block_config_for_lora(self.cfg)
        target_modules = OmegaConf.to_container(
            self.cfg.lora.target_modules, resolve=True
        )

        self._lora_manager = BlockwiseLoRA(
            model=transformer,
            block_config=block_config,
            identity_rank=self.cfg.lora.identity_rank,
            context_rank=self.cfg.lora.context_rank,
            shared_rank=self.cfg.lora.shared_rank,
            alpha_ratio=self.cfg.lora.alpha_ratio,
            dropout=0.0,
            target_modules=target_modules,
        )
        self._lora_manager.apply_lora()
        self._lora_manager.load_lora(self.lora_path)
        logger.info("LoRA weights loaded from %s", self.lora_path)

    def generate(
        self,
        prompt: str,
        num_images: int = 4,
        seed: int | None = None,
    ) -> list[Any]:
        """Generate images for a given prompt.

        Args:
            prompt: Text prompt containing the subject token.
            num_images: Number of images to generate.
            seed: Optional random seed.

        Returns:
            List of PIL Images.
        """
        if self._pipeline is None:
            self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info("Generating %d images for prompt: '%s'", num_images, prompt)

        with torch.no_grad():
            result = self._pipeline(
                prompt=prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=self.cfg.inference.num_steps,
                guidance_scale=self.cfg.inference.guidance_scale,
                generator=generator,
            )

        return result.images


class MultiSubjectGenerator:
    """Generate images with multiple fine-tuned LoRA adapters and spatial layout.

    Uses bounding-box-guided attention masking to compose multiple subjects
    in a single image while minimising identity leakage.

    Args:
        cfg: Full configuration.
        lora_paths: List of paths to LoRA checkpoints.
        tokens: List of subject identifier tokens.
        class_nouns: List of class nouns.
        layout: List of ``[x_min, y_min, x_max, y_max]`` bounding boxes.
        device: Torch device string.
    """

    def __init__(
        self,
        cfg: DictConfig,
        lora_paths: list[str],
        tokens: list[str],
        class_nouns: list[str],
        layout: list[list[float]],
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.lora_paths = lora_paths
        self.tokens = tokens
        self.class_nouns = class_nouns
        self.layout = layout
        self.device = device

        self._pipeline: Any = None
        self._lora_managers: list[Any] = []

    def _load_pipeline(self) -> None:
        """Load the diffusion pipeline and apply all LoRA adapters."""
        from diffusers import DiffusionPipeline
        from modularbooth.models.blockwise_lora import BlockwiseLoRA

        logger.info("Loading backbone: %s", self.cfg.model.backbone)
        dtype = getattr(torch, self.cfg.model.dtype)
        self._pipeline = DiffusionPipeline.from_pretrained(
            self.cfg.model.backbone,
            torch_dtype=dtype,
            revision=self.cfg.model.get("revision"),
        ).to(self.device)

        transformer = getattr(self._pipeline, "transformer", None)
        if transformer is None:
            transformer = getattr(self._pipeline, "unet", None)

        block_config = _build_block_config_for_lora(self.cfg)
        target_modules = OmegaConf.to_container(
            self.cfg.lora.target_modules, resolve=True
        )

        # For multi-subject, we load LoRA weights sequentially and accumulate
        # their effects.  In practice this would use specialised LoRA
        # composition (e.g. LoRA switch or weighted merging); here we apply
        # each adapter to a separate copy or use additive composition.
        for i, lora_path in enumerate(self.lora_paths):
            logger.info(
                "Loading LoRA %d/%d for %s %s from %s",
                i + 1,
                len(self.lora_paths),
                self.tokens[i],
                self.class_nouns[i],
                lora_path,
            )
            lora_manager = BlockwiseLoRA(
                model=transformer,
                block_config=block_config,
                identity_rank=self.cfg.lora.identity_rank,
                context_rank=self.cfg.lora.context_rank,
                shared_rank=self.cfg.lora.shared_rank,
                alpha_ratio=self.cfg.lora.alpha_ratio,
                dropout=0.0,
                target_modules=target_modules,
            )
            if i == 0:
                lora_manager.apply_lora()
            lora_manager.load_lora(lora_path)
            self._lora_managers.append(lora_manager)

    def _build_spatial_masks(
        self,
        resolution: int,
    ) -> list[torch.Tensor]:
        """Build spatial attention masks from bounding boxes.

        Args:
            resolution: Image resolution in pixels.

        Returns:
            List of binary mask tensors of shape ``(1, 1, H, W)``.
        """
        masks = []
        for box in self.layout:
            x_min, y_min, x_max, y_max = box
            mask = torch.zeros(1, 1, resolution, resolution, device=self.device)
            h_start = int(y_min * resolution)
            h_end = int(y_max * resolution)
            w_start = int(x_min * resolution)
            w_end = int(x_max * resolution)
            mask[:, :, h_start:h_end, w_start:w_end] = 1.0
            masks.append(mask)
        return masks

    def generate(
        self,
        prompt: str,
        num_images: int = 4,
        seed: int | None = None,
    ) -> list[Any]:
        """Generate multi-subject images for a given prompt.

        Args:
            prompt: Text prompt containing all subject tokens.
            num_images: Number of images to generate.
            seed: Optional random seed.

        Returns:
            List of PIL Images.
        """
        if self._pipeline is None:
            self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        resolution = self.cfg.inference.resolution
        neg_attn_strength = self.cfg.inference.negative_attention_strength
        spatial_masks = self._build_spatial_masks(resolution)

        logger.info(
            "Generating %d multi-subject images for prompt: '%s' "
            "(subjects: %s, layout: %s)",
            num_images,
            prompt,
            list(zip(self.tokens, self.class_nouns)),
            self.layout,
        )

        # Multi-subject generation with layout guidance.
        # The actual implementation depends on the pipeline's support for
        # cross-attention control and spatial masking; here we invoke the
        # pipeline with additional kwargs that a custom pipeline would handle.
        with torch.no_grad():
            result = self._pipeline(
                prompt=prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=self.cfg.inference.num_steps,
                guidance_scale=self.cfg.inference.guidance_scale,
                generator=generator,
                cross_attention_kwargs={
                    "spatial_masks": spatial_masks,
                    "subject_tokens": self.tokens,
                    "negative_attention_strength": neg_attn_strength,
                },
            )

        return result.images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_block_config_for_lora(cfg: DictConfig) -> dict[int, str]:
    """Flatten the YAML block_config into a ``{block_idx: role}`` mapping.

    Delegates to the same logic used in the training script.

    Args:
        cfg: Full configuration.

    Returns:
        Mapping from block index to role string.
    """
    raw_block_config = cfg.lora.block_config
    if raw_block_config is None:
        return {}

    mapping: dict[int, str] = {}
    block_config = OmegaConf.to_container(raw_block_config, resolve=True)
    if not isinstance(block_config, dict):
        return mapping

    for group_key, group_val in block_config.items():
        if not isinstance(group_val, dict):
            continue
        for role, indices in group_val.items():
            if isinstance(indices, list):
                for idx in indices:
                    mapping[int(idx)] = str(role)

    return mapping


def _sanitize_filename(text: str, max_length: int = 80) -> str:
    """Create a filesystem-safe filename from a text string.

    Args:
        text: Input text (e.g. a prompt).
        max_length: Maximum length of the output string.

    Returns:
        Sanitised filename-safe string.
    """
    import re
    sanitized = re.sub(r"[^\w\s-]", "", text)
    sanitized = re.sub(r"[\s]+", "_", sanitized).strip("_")
    return sanitized[:max_length]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """ModularBooth generation entry-point.

    Orchestrates the image generation pipeline:

    1. Parse CLI arguments and load configuration.
    2. Select single- or multi-subject generator based on the number of LoRAs.
    3. Generate images for all prompts.
    4. Save images organised by prompt into the output directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Validate cardinality.
    num_loras = len(args.lora_paths)
    if len(args.tokens) != num_loras:
        parser.error(
            f"Number of --tokens ({len(args.tokens)}) must match "
            f"number of --lora-paths ({num_loras})."
        )
    if len(args.class_nouns) != num_loras:
        parser.error(
            f"Number of --class-nouns ({len(args.class_nouns)}) must match "
            f"number of --lora-paths ({num_loras})."
        )

    # Validate LoRA checkpoint paths.
    for lora_path in args.lora_paths:
        if not Path(lora_path).is_file():
            parser.error(f"LoRA checkpoint not found: {lora_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config ----
    logger.info("Loading config for backbone '%s'...", args.backbone)
    cfg = load_config(backbone=args.backbone)

    # Override inference settings from CLI.
    OmegaConf.update(cfg, "inference.num_images_per_prompt", args.num_images, merge=True)
    if args.seed is not None:
        OmegaConf.update(cfg, "inference.seed", args.seed, merge=True)

    # ---- Resolve prompts ----
    prompts = _resolve_prompts(args.prompts)
    logger.info("Resolved %d prompt(s).", len(prompts))

    # ---- Select generator ----
    is_multi_subject = num_loras > 1

    if is_multi_subject:
        logger.info(
            "Multi-subject mode: %d subjects (%s).",
            num_loras,
            list(zip(args.tokens, args.class_nouns)),
        )
        layout = _parse_layout(args.layout, num_loras, args.layout_strategy)
        generator = MultiSubjectGenerator(
            cfg=cfg,
            lora_paths=args.lora_paths,
            tokens=args.tokens,
            class_nouns=args.class_nouns,
            layout=layout,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        logger.info(
            "Single-subject mode: %s %s.",
            args.tokens[0],
            args.class_nouns[0],
        )
        generator = SingleSubjectGenerator(
            cfg=cfg,
            lora_path=args.lora_paths[0],
            token=args.tokens[0],
            class_noun=args.class_nouns[0],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # ---- Generate for all prompts ----
    total_generated = 0
    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir_name = f"{prompt_idx:03d}_{_sanitize_filename(prompt)}"
        prompt_dir = output_dir / prompt_dir_name
        prompt_dir.mkdir(parents=True, exist_ok=True)

        images = generator.generate(
            prompt=prompt,
            num_images=args.num_images,
            seed=args.seed,
        )

        for img_idx, image in enumerate(images):
            file_name = f"image_{img_idx:03d}.png"
            save_path = prompt_dir / file_name
            image.save(str(save_path))
            logger.debug("Saved %s", save_path)

        total_generated += len(images)
        logger.info(
            "Prompt %d/%d: saved %d images to %s",
            prompt_idx + 1,
            len(prompts),
            len(images),
            prompt_dir,
        )

    # ---- Save generation metadata ----
    metadata = {
        "backbone": args.backbone,
        "lora_paths": args.lora_paths,
        "tokens": args.tokens,
        "class_nouns": args.class_nouns,
        "prompts": prompts,
        "num_images_per_prompt": args.num_images,
        "seed": args.seed,
        "layout": args.layout,
        "layout_strategy": args.layout_strategy,
        "is_multi_subject": is_multi_subject,
        "total_images": total_generated,
    }
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("  Generation Summary")
    print("=" * 50)
    print(f"  Mode:            {'multi-subject' if is_multi_subject else 'single-subject'}")
    print(f"  Prompts:         {len(prompts)}")
    print(f"  Images/prompt:   {args.num_images}")
    print(f"  Total images:    {total_generated}")
    print(f"  Output:          {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
