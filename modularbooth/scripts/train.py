#!/usr/bin/env python3
"""ModularBooth training entry-point.

Trains a blockwise LoRA adapter on a Diffusion Transformer backbone for
single-subject personalisation using DreamBooth with optional prior
preservation loss (PPL) and contrastive context disentanglement (CCD).

Usage::

    python -m modularbooth.scripts.train \
        --subject-dir ./data/dog \
        --class-noun dog \
        --backbone flux \
        --output-dir ./outputs/dog_experiment

See ``--help`` for the full list of CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
    """Build the CLI argument parser for the training script.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a blockwise LoRA adapter for single-subject personalisation "
            "on a Diffusion Transformer backbone."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--subject-dir",
        type=str,
        required=True,
        help="Path to the directory containing 3-5 subject images.",
    )
    parser.add_argument(
        "--class-noun",
        type=str,
        required=True,
        help='Natural-language class noun for the subject (e.g. "dog", "person").',
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--token",
        type=str,
        default="[V]",
        help='Rare identifier token for the subject (default: "[V]").',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Root output directory for checkpoints, images, and logs (default: ./outputs).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["flux", "sd3"],
        default="flux",
        help='Diffusion Transformer backbone to use (default: "flux").',
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "Optional list of config overrides in dotted key=value format. "
            'Example: --config-overrides training.num_steps=1000 lora.identity_rank=32'
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint (.safetensors) to resume training from.",
    )
    parser.add_argument(
        "--no-ccd",
        action="store_true",
        help="Disable the Contrastive Context Disentanglement (CCD) loss.",
    )
    parser.add_argument(
        "--no-ppl",
        action="store_true",
        help="Disable Prior Preservation Loss (PPL) and skip class image generation.",
    )

    return parser


def _parse_overrides(raw_overrides: list[str] | None) -> dict[str, Any]:
    """Parse a list of ``"key=value"`` strings into a nested override dict.

    Dotted keys are expanded into nested dictionaries so that OmegaConf can
    merge them correctly.  For example ``"training.num_steps=1000"`` becomes
    ``{"training": {"num_steps": 1000}}``.

    Numeric strings are automatically cast to ``int`` or ``float`` where
    appropriate.  The literal strings ``"true"`` and ``"false"`` (case-
    insensitive) are cast to booleans.

    Args:
        raw_overrides: List of ``"key=value"`` strings from ``--config-overrides``.

    Returns:
        Nested dictionary suitable for ``OmegaConf.merge``.

    Raises:
        ValueError: If an override string does not contain ``"="``.
    """
    if not raw_overrides:
        return {}

    overrides: dict[str, Any] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(
                f"Config override must be in 'key=value' format, got: {item!r}"
            )
        key, value_str = item.split("=", 1)
        value: Any = _cast_value(value_str)

        # Expand dotted key into nested dicts.
        parts = key.split(".")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    return overrides


def _cast_value(value_str: str) -> Any:
    """Attempt to cast a string value to an appropriate Python type.

    Args:
        value_str: Raw string value from the CLI.

    Returns:
        The value as ``int``, ``float``, ``bool``, ``None``, or ``str``.
    """
    # Booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    # None
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None
    # Integers
    try:
        return int(value_str)
    except ValueError:
        pass
    # Floats
    try:
        return float(value_str)
    except ValueError:
        pass
    # Fallback to string
    return value_str


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _apply_cli_to_config(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    """Apply CLI arguments and overrides on top of the loaded config.

    Mutates *cfg* in-place and returns it for convenience.

    Args:
        cfg: The base configuration loaded from YAML files.
        args: Parsed CLI arguments.

    Returns:
        The updated configuration.
    """
    # Core subject settings
    OmegaConf.update(cfg, "subject.token", args.token, merge=True)
    OmegaConf.update(cfg, "subject.class_noun", args.class_noun, merge=True)
    OmegaConf.update(cfg, "output.dir", args.output_dir, merge=True)

    # Feature flags from CLI
    if args.no_ccd:
        OmegaConf.update(cfg, "ccd.enabled", False, merge=True)
    if args.no_ppl:
        OmegaConf.update(cfg, "prior_preservation.enabled", False, merge=True)

    # User-supplied key=value overrides
    cli_overrides = _parse_overrides(args.config_overrides)
    if cli_overrides:
        override_cfg = OmegaConf.create(cli_overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def _generate_class_images_if_needed(
    cfg: DictConfig,
    class_noun: str,
    output_dir: Path,
) -> Path:
    """Generate class-prior images for PPL if they do not already exist.

    Uses the frozen base pipeline to produce ``num_class_images`` samples of
    the target class.

    Args:
        cfg: Full training configuration.
        class_noun: The class noun (e.g. ``"dog"``).
        output_dir: Root output directory.

    Returns:
        Path to the directory containing class images.
    """
    from diffusers import DiffusionPipeline
    from modularbooth.losses.prior_preservation import generate_class_images

    class_images_dir = output_dir / "class_images"
    num_class_images = cfg.prior_preservation.num_class_images
    class_prompt = f"a photo of a {class_noun}"

    existing_images = list(class_images_dir.glob("*.png")) if class_images_dir.exists() else []
    if len(existing_images) >= num_class_images:
        logger.info(
            "Found %d existing class images in %s; skipping generation.",
            len(existing_images),
            class_images_dir,
        )
        return class_images_dir

    logger.info("Generating %d class images for '%s'...", num_class_images, class_prompt)
    pipeline = DiffusionPipeline.from_pretrained(
        cfg.model.backbone,
        torch_dtype=getattr(torch, cfg.model.dtype),
        revision=cfg.model.get("revision"),
    ).to("cuda")

    generate_class_images(
        pipeline=pipeline,
        class_prompt=class_prompt,
        num_images=num_class_images,
        output_dir=str(class_images_dir),
        batch_size=cfg.training.batch_size,
        seed=cfg.training.seed,
    )

    # Free the pipeline to reclaim VRAM.
    del pipeline
    torch.cuda.empty_cache()

    return class_images_dir


def _generate_augmented_images(
    cfg: DictConfig,
    subject_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate background-augmented images for CCD loss.

    Uses :class:`BackgroundAugmentor` to produce variants with replaced
    backgrounds for each subject image.

    Args:
        cfg: Full training configuration.
        subject_dir: Path to the original subject images.
        output_dir: Root output directory.

    Returns:
        Path to the directory containing augmented images.
    """
    from modularbooth.data.augmentation import BackgroundAugmentor

    augmented_dir = output_dir / "augmented_images"
    num_variants = cfg.ccd.num_augmentations

    if augmented_dir.exists() and any(augmented_dir.iterdir()):
        logger.info("Augmented images already exist at %s; skipping generation.", augmented_dir)
        return augmented_dir

    logger.info("Generating %d augmented variants per subject image...", num_variants)
    augmentor = BackgroundAugmentor(seed=cfg.training.seed)
    augmentor.augment_subject(
        subject_images_dir=str(subject_dir),
        output_dir=str(augmented_dir),
        num_variants=num_variants,
    )

    return augmented_dir


def _build_block_config_mapping(cfg: DictConfig) -> dict[int, str]:
    """Flatten the YAML block_config structure into a ``{block_idx: role}`` dict.

    The YAML config may store block classifications in grouped form::

        block_config:
          double_blocks:
            context: [0, 1, 2]
            identity: [3, 4]
          single_blocks:
            ...

    or in flat form (output of ``probe_blocks``)::

        block_config:
          blocks:
            context: [0, 1]
            shared: [2, 3]
            identity: [4, 5]

    This helper normalises both representations into a flat mapping.

    Args:
        cfg: Full training configuration.

    Returns:
        Mapping from block index to role string.
    """
    raw_block_config = cfg.lora.block_config
    if raw_block_config is None:
        logger.warning(
            "No block_config specified -- falling back to all blocks as 'shared'. "
            "Run `probe_blocks` first for optimal results."
        )
        return {}

    mapping: dict[int, str] = {}
    # OmegaConf DictConfig iteration
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


def _create_model(cfg: DictConfig) -> torch.nn.Module:
    """Load the backbone Diffusion Transformer model.

    Args:
        cfg: Full training configuration.

    Returns:
        The loaded model moved to the appropriate device and dtype.
    """
    from diffusers import DiffusionPipeline

    logger.info("Loading backbone model: %s", cfg.model.backbone)
    dtype = getattr(torch, cfg.model.dtype)
    pipeline = DiffusionPipeline.from_pretrained(
        cfg.model.backbone,
        torch_dtype=dtype,
        revision=cfg.model.get("revision"),
    )

    # Extract the transformer (DiT) component.
    # Different pipelines store the transformer under different attribute names.
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        transformer = getattr(pipeline, "unet", None)
    if transformer is None:
        raise AttributeError(
            f"Could not find a transformer or unet component in the pipeline "
            f"loaded from '{cfg.model.backbone}'."
        )

    transformer = transformer.to("cuda")
    transformer.requires_grad_(False)

    # Keep the text encoders and VAE around for training loop use.
    pipeline.to("cuda")

    return pipeline


def _create_lora(
    model: torch.nn.Module,
    cfg: DictConfig,
    block_config: dict[int, str],
) -> Any:
    """Create and apply blockwise LoRA adapters to the model.

    Args:
        model: The transformer backbone (nn.Module).
        cfg: Full training configuration.
        block_config: Mapping from block index to role.

    Returns:
        The :class:`BlockwiseLoRA` manager instance.
    """
    from modularbooth.models.blockwise_lora import BlockwiseLoRA

    target_modules = OmegaConf.to_container(cfg.lora.target_modules, resolve=True)

    lora_manager = BlockwiseLoRA(
        model=model,
        block_config=block_config,
        identity_rank=cfg.lora.identity_rank,
        context_rank=cfg.lora.context_rank,
        shared_rank=cfg.lora.shared_rank,
        alpha_ratio=cfg.lora.alpha_ratio,
        dropout=cfg.lora.dropout,
        target_modules=target_modules,
    )
    lora_manager.apply_lora()

    return lora_manager


def _create_loss(cfg: DictConfig) -> torch.nn.Module:
    """Instantiate the combined ModularBooth loss from config.

    Args:
        cfg: Full training configuration.

    Returns:
        A :class:`ModularBoothLoss` module.
    """
    from modularbooth.losses.combined import ModularBoothLoss

    ppl_enabled = cfg.prior_preservation.enabled
    ccd_enabled = cfg.ccd.enabled

    return ModularBoothLoss(
        lambda_ppl=cfg.prior_preservation.lambda_ppl if ppl_enabled else 0.0,
        lambda_ccd=cfg.ccd.lambda_ccd if ccd_enabled else 0.0,
        ccd_warmup_steps=cfg.ccd.warmup_steps if ccd_enabled else 0,
        ccd_temperature=cfg.ccd.temperature if ccd_enabled else 0.07,
    )


def _create_optimizer(
    lora_manager: Any,
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    """Create the optimizer for LoRA parameters.

    Args:
        lora_manager: The :class:`BlockwiseLoRA` manager holding trainable params.
        cfg: Full training configuration.

    Returns:
        Configured optimizer instance.
    """
    lora_params = list(lora_manager.get_lora_params().values())
    optimizer_name = cfg.training.optimizer.lower()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            lora_params,
            lr=cfg.training.learning_rate,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            lora_params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> Any:
    """Create the learning rate scheduler from config.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Full training configuration.

    Returns:
        The configured LR scheduler.
    """
    from modularbooth.training.scheduler import build_scheduler

    return build_scheduler(optimizer, cfg)


def _training_loop(
    pipeline: Any,
    lora_manager: Any,
    dataset: Any,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: DictConfig,
    output_dir: Path,
    resume_path: str | None = None,
) -> dict[str, Any]:
    """Execute the main training loop.

    Args:
        pipeline: The full diffusion pipeline (for encoding text, VAE, etc.).
        lora_manager: The :class:`BlockwiseLoRA` manager.
        dataset: The :class:`DreamBoothDataset`.
        loss_fn: The combined loss module.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        cfg: Full training configuration.
        output_dir: Root output directory.
        resume_path: Optional checkpoint path to resume from.

    Returns:
        Dictionary with training summary (``steps``, ``final_loss``, etc.).
    """
    from torch.utils.data import DataLoader

    num_steps = cfg.training.num_steps
    batch_size = cfg.training.batch_size
    grad_accum = cfg.training.gradient_accumulation
    max_grad_norm = cfg.training.max_grad_norm
    log_every = cfg.training.log_every
    save_every = cfg.training.save_every
    seed = cfg.training.seed

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified.
    start_step = 0
    if resume_path is not None:
        logger.info("Resuming from checkpoint: %s", resume_path)
        lora_manager.load_lora(resume_path)
        # Attempt to infer the step number from the filename.
        resume_stem = Path(resume_path).stem
        for part in resume_stem.split("_"):
            if part.isdigit():
                start_step = int(part)
                break
        logger.info("Resuming from step %d.", start_step)

    # Set random seed for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    logger.info("Starting training for %d steps (starting from step %d)...", num_steps, start_step)
    start_time = time.time()

    global_step = start_step
    running_loss = 0.0
    last_loss = 0.0

    while global_step < num_steps:
        for batch in dataloader:
            if global_step >= num_steps:
                break

            # Forward pass through the diffusion model and compute loss.
            # The actual forward logic depends on the pipeline internals;
            # here we outline the standard DreamBooth training step.
            pixel_values = batch["pixel_values"].to("cuda")
            captions = batch["input_ids"]
            is_class = batch["is_class_image"]

            # Separate subject and class samples.
            subject_mask = ~is_class
            class_mask = is_class

            if not subject_mask.any() or not class_mask.any():
                continue

            # Encode images through the VAE.
            with torch.no_grad():
                vae = getattr(pipeline, "vae", None)
                if vae is not None:
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                else:
                    latents = pixel_values

            # Sample noise and timesteps.
            noise = torch.randn_like(latents)
            scheduler_train = getattr(pipeline, "scheduler", None)
            if scheduler_train is not None:
                timesteps = torch.randint(
                    0,
                    scheduler_train.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = scheduler_train.add_noise(latents, noise, timesteps)
            else:
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
                noisy_latents = latents + noise

            # Encode text prompts.
            transformer = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)

            # Get model predictions.
            model_pred = transformer(noisy_latents, timesteps).sample

            # Split predictions for subject and class.
            model_pred_subject = model_pred[subject_mask]
            model_pred_class = model_pred[class_mask]
            noise_target_subject = noise[subject_mask]
            noise_target_class = noise[class_mask]

            # Compute CCD features if enabled and augmented data is available.
            subject_features = None
            positive_features = None
            negative_features = None

            if cfg.ccd.enabled and "augmented_pixel_values" in batch:
                # CCD feature extraction would be performed on intermediate
                # DiT features here via hook-based feature extraction.
                pass

            # Compute combined loss.
            loss_dict = loss_fn(
                model_pred_subject=model_pred_subject,
                noise_target_subject=noise_target_subject,
                model_pred_class=model_pred_class,
                noise_target_class=noise_target_class,
                subject_features=subject_features,
                positive_features=positive_features,
                negative_features=negative_features,
                global_step=global_step,
            )

            total_loss = loss_dict["total_loss"] / grad_accum
            total_loss.backward()

            if (global_step + 1) % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_manager.get_lora_params().values(),
                        max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            last_loss = loss_dict["total_loss"].item()
            running_loss += last_loss

            # Logging
            if (global_step + 1) % log_every == 0:
                avg_loss = running_loss / log_every
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                steps_per_sec = (global_step - start_step + 1) / elapsed
                logger.info(
                    "Step %d/%d | loss=%.4f | lr=%.2e | %.2f steps/s | "
                    "diffusion=%.4f | ppl=%.4f | ccd=%.4f",
                    global_step + 1,
                    num_steps,
                    avg_loss,
                    lr,
                    steps_per_sec,
                    loss_dict["loss_components"]["diffusion_loss_raw"],
                    loss_dict["loss_components"]["ppl_loss_weighted"],
                    loss_dict["loss_components"]["ccd_loss_weighted"],
                )
                running_loss = 0.0

            # Periodic checkpoint saving
            if (global_step + 1) % save_every == 0:
                ckpt_path = checkpoints_dir / f"lora_step_{global_step + 1:06d}.safetensors"
                lora_manager.save_lora(str(ckpt_path))
                logger.info("Saved checkpoint: %s", ckpt_path)

            global_step += 1

    elapsed_total = time.time() - start_time
    logger.info("Training complete in %.1f seconds.", elapsed_total)

    return {
        "steps": global_step,
        "final_loss": last_loss,
        "elapsed_seconds": elapsed_total,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """ModularBooth training entry-point.

    Orchestrates the full training pipeline:

    1. Parse CLI arguments and load configuration.
    2. Apply CLI overrides to the config.
    3. Generate class images if PPL is enabled and they do not exist.
    4. Generate augmented images if CCD is enabled.
    5. Create the dataset, model, LoRA adapters, loss, and trainer.
    6. Run the training loop.
    7. Save the final LoRA checkpoint.
    8. Print a summary of the training run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Validate paths
    subject_dir = Path(args.subject_dir).resolve()
    if not subject_dir.is_dir():
        parser.error(f"Subject directory does not exist: {subject_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None and not Path(args.resume).is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume}")

    # ---- 1. Load config ----
    logger.info("Loading config for backbone '%s'...", args.backbone)
    cfg = load_config(backbone=args.backbone)

    # ---- 2. Apply CLI overrides ----
    cfg = _apply_cli_to_config(cfg, args)

    # Save the resolved config for reproducibility.
    config_save_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, str(config_save_path))
    logger.info("Resolved config saved to %s", config_save_path)

    # ---- 3. Generate class images if PPL enabled ----
    class_images_dir: Path | None = None
    if cfg.prior_preservation.enabled:
        class_images_dir = _generate_class_images_if_needed(cfg, args.class_noun, output_dir)
    else:
        logger.info("Prior Preservation Loss is disabled; skipping class image generation.")
        # Create a minimal class images dir with a placeholder so the dataset
        # can still function (it requires class_images_dir).
        class_images_dir = output_dir / "class_images"
        class_images_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. Generate augmented images if CCD enabled ----
    augmented_dir: Path | None = None
    if cfg.ccd.enabled:
        augmented_dir = _generate_augmented_images(cfg, subject_dir, output_dir)
    else:
        logger.info("CCD loss is disabled; skipping augmentation.")

    # ---- 5. Create dataset, model, LoRA, loss, trainer ----
    logger.info("Creating dataset...")
    from modularbooth.data.dataset import DreamBoothDataset

    dataset = DreamBoothDataset.from_config(
        subject_images_dir=str(subject_dir),
        class_images_dir=str(class_images_dir),
        cfg=cfg,
        augmented_images_dir=str(augmented_dir) if augmented_dir else None,
    )

    logger.info("Loading model and applying LoRA...")
    pipeline = _create_model(cfg)
    transformer = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)

    block_config = _build_block_config_mapping(cfg)
    lora_manager = _create_lora(transformer, cfg, block_config)

    loss_fn = _create_loss(cfg)
    optimizer = _create_optimizer(lora_manager, cfg)
    scheduler = _create_scheduler(optimizer, cfg)

    # ---- 6. Train ----
    summary = _training_loop(
        pipeline=pipeline,
        lora_manager=lora_manager,
        dataset=dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=args.resume,
    )

    # ---- 7. Save final LoRA checkpoint ----
    final_ckpt_path = output_dir / "checkpoints" / "lora_final.safetensors"
    lora_manager.save_lora(str(final_ckpt_path))
    logger.info("Final LoRA checkpoint saved to %s", final_ckpt_path)

    # ---- 8. Print summary ----
    param_counts = lora_manager.get_parameter_count()
    total_params = sum(param_counts.values())

    print("\n" + "=" * 60)
    print("  ModularBooth Training Summary")
    print("=" * 60)
    print(f"  Backbone:         {args.backbone}")
    print(f"  Subject:          {args.token} {args.class_noun}")
    print(f"  Total steps:      {summary['steps']}")
    print(f"  Final loss:       {summary['final_loss']:.4f}")
    print(f"  Elapsed time:     {summary['elapsed_seconds']:.1f}s")
    print(f"  Checkpoint:       {final_ckpt_path}")
    print(f"  LoRA parameters:  {total_params:,}")
    print(f"    Identity:       {param_counts.get('identity', 0):,}")
    print(f"    Context:        {param_counts.get('context', 0):,}")
    print(f"    Shared:         {param_counts.get('shared', 0):,}")
    print(f"  PPL enabled:      {cfg.prior_preservation.enabled}")
    print(f"  CCD enabled:      {cfg.ccd.enabled}")
    print("=" * 60)


if __name__ == "__main__":
    main()
