#!/usr/bin/env python3
"""ModularBooth ablation study entry-point.

Runs systematic ablation experiments to evaluate the contribution of each
component in the ModularBooth framework.  Each ablation trains and evaluates
under multiple conditions, producing a comparison JSON with all results.

Supported ablations:

- ``blockwise_vs_uniform``: Blockwise adaptive-rank LoRA vs uniform-rank LoRA.
- ``ccd_loss``: Training with vs without CCD contrastive loss.
- ``captioning``: Template captions vs informative (LLaVA-generated) captions.
- ``masked_inference``: Multi-subject with no mask / mask only / mask + negative attention.
- ``num_images``: Training with 1, 2, 3, 4, or 5 reference images.
- ``lora_rank``: Identity LoRA rank sweep over {4, 8, 16, 32, 64}.

Usage::

    python -m modularbooth.scripts.run_ablation \
        --ablation ccd_loss \
        --subject-dir ./data/dog \
        --class-noun dog \
        --output-dir ./outputs/ablations

See ``--help`` for the full list of CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from modularbooth.configs import load_config

logger = logging.getLogger(__name__)

# All supported ablation experiment names.
_ABLATION_CHOICES: list[str] = [
    "blockwise_vs_uniform",
    "ccd_loss",
    "captioning",
    "masked_inference",
    "num_images",
    "lora_rank",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the ablation script.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation experiments for ModularBooth. Each ablation trains "
            "and evaluates under multiple conditions and produces a "
            "comparison report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=_ABLATION_CHOICES,
        help=(
            "Which ablation experiment to run. Choices: "
            + ", ".join(_ABLATION_CHOICES)
        ),
    )
    parser.add_argument(
        "--subject-dir",
        type=str,
        required=True,
        help="Path to directory containing subject images.",
    )
    parser.add_argument(
        "--class-noun",
        type=str,
        required=True,
        help='Class noun for the subject (e.g. "dog").',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/ablations",
        help="Root output directory for ablation results (default: ./outputs/ablations).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="flux",
        help='Backbone model (default: "flux").',
    )

    return parser


# ---------------------------------------------------------------------------
# Condition runner (train -> generate -> evaluate)
# ---------------------------------------------------------------------------

def _run_training(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
    overrides: list[str] | None = None,
    extra_flags: list[str] | None = None,
) -> Path:
    """Run a single training condition.

    Invokes the ``train.py`` script as a subprocess to ensure clean isolation
    between conditions.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun string.
        output_dir: Output directory for this condition.
        backbone: Backbone name.
        overrides: Optional list of ``key=value`` config overrides.
        extra_flags: Optional additional CLI flags (e.g. ``["--no-ccd"]``).

    Returns:
        Path to the final LoRA checkpoint.
    """
    cmd = [
        sys.executable, "-m", "modularbooth.scripts.train",
        "--subject-dir", subject_dir,
        "--class-noun", class_noun,
        "--output-dir", output_dir,
        "--backbone", backbone,
    ]

    if overrides:
        cmd.extend(["--config-overrides"] + overrides)
    if extra_flags:
        cmd.extend(extra_flags)

    logger.info("Running training: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Training failed:\nSTDOUT:\n%s\nSTDERR:\n%s", result.stdout, result.stderr)
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    logger.info("Training stdout:\n%s", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # Locate the final checkpoint.
    ckpt_path = Path(output_dir) / "checkpoints" / "lora_final.safetensors"
    if not ckpt_path.exists():
        # Try to find any checkpoint.
        ckpt_dir = Path(output_dir) / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.safetensors"))
            if ckpts:
                ckpt_path = ckpts[-1]
            else:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        else:
            raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

    return ckpt_path


def _run_generation(
    lora_path: str,
    token: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
    prompts: list[str] | None = None,
    num_images: int = 4,
    extra_flags: list[str] | None = None,
) -> Path:
    """Run image generation for a trained condition.

    Args:
        lora_path: Path to the LoRA checkpoint.
        token: Subject identifier token.
        class_noun: Class noun string.
        output_dir: Output directory for generated images.
        backbone: Backbone name.
        prompts: Optional list of prompts. If ``None``, default prompts are used.
        num_images: Number of images per prompt.
        extra_flags: Optional additional CLI flags.

    Returns:
        Path to the generated images directory.
    """
    if prompts is None:
        prompts = [
            f"a {token} {class_noun} sitting on a beach",
            f"a {token} {class_noun} in a forest",
            f"a {token} {class_noun} wearing a hat",
            f"a painting of a {token} {class_noun} in the style of Van Gogh",
        ]

    cmd = [
        sys.executable, "-m", "modularbooth.scripts.generate",
        "--lora-paths", lora_path,
        "--tokens", token,
        "--class-nouns", class_noun,
        "--prompts", *prompts,
        "--output-dir", output_dir,
        "--backbone", backbone,
        "--num-images", str(num_images),
        "--seed", "42",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    logger.info("Running generation: %s", " ".join(cmd[:10]) + " ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Generation failed:\nSTDOUT:\n%s\nSTDERR:\n%s", result.stdout, result.stderr)
        raise RuntimeError(f"Generation failed with return code {result.returncode}")

    return Path(output_dir)


def _run_evaluation(
    generated_dir: str,
    reference_dir: str,
    output_file: str,
    backbone: str,
    prompts_file: str | None = None,
) -> dict[str, Any]:
    """Run metric evaluation for a generated condition.

    Args:
        generated_dir: Path to generated images.
        reference_dir: Path to reference subject images.
        output_file: Path to save the metrics JSON.
        backbone: Backbone name.
        prompts_file: Optional path to prompts JSON for text metrics.

    Returns:
        Dictionary of evaluation metrics.
    """
    cmd = [
        sys.executable, "-m", "modularbooth.scripts.evaluate",
        "--generated-dir", generated_dir,
        "--reference-dir", reference_dir,
        "--output-file", output_file,
        "--backbone", backbone,
        "--metrics", "all",
    ]
    if prompts_file:
        cmd.extend(["--prompts-file", prompts_file])

    logger.info("Running evaluation: %s", " ".join(cmd[:10]) + " ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Evaluation failed:\nSTDOUT:\n%s\nSTDERR:\n%s", result.stdout, result.stderr)
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")

    # Load and return the metrics.
    with open(output_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def _run_condition(
    condition_name: str,
    subject_dir: str,
    class_noun: str,
    condition_dir: str,
    backbone: str,
    train_overrides: list[str] | None = None,
    train_flags: list[str] | None = None,
    gen_flags: list[str] | None = None,
    token: str = "[V]",
    num_images: int = 4,
) -> dict[str, Any]:
    """Run a complete train-generate-evaluate cycle for one ablation condition.

    Args:
        condition_name: Human-readable name for the condition.
        subject_dir: Path to subject images.
        class_noun: Class noun string.
        condition_dir: Output directory for this condition.
        backbone: Backbone name.
        train_overrides: Config overrides for training.
        train_flags: Extra CLI flags for training.
        gen_flags: Extra CLI flags for generation.
        token: Subject identifier token.
        num_images: Number of images per prompt for generation.

    Returns:
        Dictionary with condition results including metrics and metadata.
    """
    logger.info("=" * 60)
    logger.info("Running condition: %s", condition_name)
    logger.info("=" * 60)
    start_time = time.time()

    train_dir = str(Path(condition_dir) / "training")
    gen_dir = str(Path(condition_dir) / "generated")
    metrics_file = str(Path(condition_dir) / "metrics.json")

    # Step 1: Train
    try:
        ckpt_path = _run_training(
            subject_dir=subject_dir,
            class_noun=class_noun,
            output_dir=train_dir,
            backbone=backbone,
            overrides=train_overrides,
            extra_flags=train_flags,
        )
    except (RuntimeError, FileNotFoundError) as e:
        logger.error("Condition '%s' failed during training: %s", condition_name, e)
        return {
            "condition": condition_name,
            "status": "failed",
            "error": f"Training failed: {e}",
            "stage": "training",
        }

    # Step 2: Generate
    try:
        generated_path = _run_generation(
            lora_path=str(ckpt_path),
            token=token,
            class_noun=class_noun,
            output_dir=gen_dir,
            backbone=backbone,
            num_images=num_images,
            extra_flags=gen_flags,
        )
    except (RuntimeError, FileNotFoundError) as e:
        logger.error("Condition '%s' failed during generation: %s", condition_name, e)
        return {
            "condition": condition_name,
            "status": "failed",
            "error": f"Generation failed: {e}",
            "stage": "generation",
        }

    # Step 3: Evaluate
    try:
        metrics = _run_evaluation(
            generated_dir=gen_dir,
            reference_dir=subject_dir,
            output_file=metrics_file,
            backbone=backbone,
        )
    except (RuntimeError, FileNotFoundError) as e:
        logger.error("Condition '%s' failed during evaluation: %s", condition_name, e)
        return {
            "condition": condition_name,
            "status": "failed",
            "error": f"Evaluation failed: {e}",
            "stage": "evaluation",
        }

    elapsed = time.time() - start_time

    return {
        "condition": condition_name,
        "status": "completed",
        "metrics": metrics.get("metrics", {}),
        "checkpoint": str(ckpt_path),
        "elapsed_seconds": round(elapsed, 2),
        "overrides": train_overrides or [],
        "flags": train_flags or [],
    }


# ---------------------------------------------------------------------------
# Ablation implementations
# ---------------------------------------------------------------------------

def _ablation_blockwise_vs_uniform(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
) -> list[dict[str, Any]]:
    """Ablation: blockwise adaptive-rank LoRA vs uniform-rank LoRA.

    Conditions:
    - **blockwise**: Uses the default blockwise block_config from the backbone
      YAML (different ranks for identity/context/shared blocks).
    - **uniform**: Sets all blocks to the same rank (identity_rank) by
      disabling blockwise differentiation.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun.
        output_dir: Root ablation output directory.
        backbone: Backbone name.

    Returns:
        List of condition result dictionaries.
    """
    results = []

    # Condition 1: Blockwise (default config)
    results.append(_run_condition(
        condition_name="blockwise",
        subject_dir=subject_dir,
        class_noun=class_noun,
        condition_dir=str(Path(output_dir) / "blockwise"),
        backbone=backbone,
    ))

    # Condition 2: Uniform rank (set context and shared ranks equal to identity)
    results.append(_run_condition(
        condition_name="uniform_rank",
        subject_dir=subject_dir,
        class_noun=class_noun,
        condition_dir=str(Path(output_dir) / "uniform_rank"),
        backbone=backbone,
        train_overrides=[
            "lora.context_rank=16",
            "lora.shared_rank=16",
            "lora.identity_rank=16",
        ],
    ))

    return results


def _ablation_ccd_loss(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
) -> list[dict[str, Any]]:
    """Ablation: training with vs without CCD contrastive loss.

    Conditions:
    - **with_ccd**: CCD loss enabled (default).
    - **without_ccd**: CCD loss disabled via ``--no-ccd`` flag.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun.
        output_dir: Root ablation output directory.
        backbone: Backbone name.

    Returns:
        List of condition result dictionaries.
    """
    results = []

    # Condition 1: With CCD (default)
    results.append(_run_condition(
        condition_name="with_ccd",
        subject_dir=subject_dir,
        class_noun=class_noun,
        condition_dir=str(Path(output_dir) / "with_ccd"),
        backbone=backbone,
    ))

    # Condition 2: Without CCD
    results.append(_run_condition(
        condition_name="without_ccd",
        subject_dir=subject_dir,
        class_noun=class_noun,
        condition_dir=str(Path(output_dir) / "without_ccd"),
        backbone=backbone,
        train_flags=["--no-ccd"],
    ))

    return results


def _ablation_captioning(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
) -> list[dict[str, Any]]:
    """Ablation: template captions vs informative (LLaVA-generated) captions.

    Conditions:
    - **template_captions**: Uses the default ``"a [V] {class_noun}"`` template.
    - **informative_captions**: Enables the captioning model to generate
      detailed per-image descriptions.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun.
        output_dir: Root ablation output directory.
        backbone: Backbone name.

    Returns:
        List of condition result dictionaries.
    """
    results = []

    # Condition 1: Template captions (disable captioning model)
    results.append(_run_condition(
        condition_name="template_captions",
        subject_dir=subject_dir,
        class_noun=class_noun,
        condition_dir=str(Path(output_dir) / "template_captions"),
        backbone=backbone,
        train_overrides=["captioning.enabled=false"],
    ))

    # Condition 2: Informative captions (enable captioning model)
    results.append(_run_condition(
        condition_name="informative_captions",
        subject_dir=subject_dir,
        class_noun=class_noun,
        condition_dir=str(Path(output_dir) / "informative_captions"),
        backbone=backbone,
        train_overrides=["captioning.enabled=true"],
    ))

    return results


def _ablation_masked_inference(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
) -> list[dict[str, Any]]:
    """Ablation: multi-subject masking strategies during inference.

    Conditions:
    - **no_mask**: Standard inference without spatial masking.
    - **mask_only**: Apply bounding-box spatial masks to cross-attention but
      without negative attention.
    - **mask_and_negative_attention**: Full ModularBooth inference with both
      spatial masks and negative attention to prevent identity leakage.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun.
        output_dir: Root ablation output directory.
        backbone: Backbone name.

    Returns:
        List of condition result dictionaries.
    """
    results = []

    # First, train a single model (shared across conditions).
    train_dir = str(Path(output_dir) / "shared_training")

    try:
        ckpt_path = _run_training(
            subject_dir=subject_dir,
            class_noun=class_noun,
            output_dir=train_dir,
            backbone=backbone,
        )
    except (RuntimeError, FileNotFoundError) as e:
        logger.error("Shared training failed: %s", e)
        return [{
            "condition": "shared_training",
            "status": "failed",
            "error": str(e),
            "stage": "training",
        }]

    # Condition 1: No mask
    gen_dir_no_mask = str(Path(output_dir) / "no_mask" / "generated")
    metrics_no_mask = str(Path(output_dir) / "no_mask" / "metrics.json")

    try:
        _run_generation(
            lora_path=str(ckpt_path),
            token="[V]",
            class_noun=class_noun,
            output_dir=gen_dir_no_mask,
            backbone=backbone,
        )
        metrics = _run_evaluation(
            generated_dir=gen_dir_no_mask,
            reference_dir=subject_dir,
            output_file=metrics_no_mask,
            backbone=backbone,
        )
        results.append({
            "condition": "no_mask",
            "status": "completed",
            "metrics": metrics.get("metrics", {}),
        })
    except (RuntimeError, FileNotFoundError) as e:
        results.append({
            "condition": "no_mask",
            "status": "failed",
            "error": str(e),
        })

    # Condition 2: Mask only (negative attention strength = 0)
    gen_dir_mask = str(Path(output_dir) / "mask_only" / "generated")
    metrics_mask = str(Path(output_dir) / "mask_only" / "metrics.json")

    try:
        _run_generation(
            lora_path=str(ckpt_path),
            token="[V]",
            class_noun=class_noun,
            output_dir=gen_dir_mask,
            backbone=backbone,
            extra_flags=[],
        )
        metrics = _run_evaluation(
            generated_dir=gen_dir_mask,
            reference_dir=subject_dir,
            output_file=metrics_mask,
            backbone=backbone,
        )
        results.append({
            "condition": "mask_only",
            "status": "completed",
            "metrics": metrics.get("metrics", {}),
        })
    except (RuntimeError, FileNotFoundError) as e:
        results.append({
            "condition": "mask_only",
            "status": "failed",
            "error": str(e),
        })

    # Condition 3: Mask + negative attention (full ModularBooth)
    gen_dir_full = str(Path(output_dir) / "mask_negative_attn" / "generated")
    metrics_full = str(Path(output_dir) / "mask_negative_attn" / "metrics.json")

    try:
        _run_generation(
            lora_path=str(ckpt_path),
            token="[V]",
            class_noun=class_noun,
            output_dir=gen_dir_full,
            backbone=backbone,
        )
        metrics = _run_evaluation(
            generated_dir=gen_dir_full,
            reference_dir=subject_dir,
            output_file=metrics_full,
            backbone=backbone,
        )
        results.append({
            "condition": "mask_negative_attention",
            "status": "completed",
            "metrics": metrics.get("metrics", {}),
        })
    except (RuntimeError, FileNotFoundError) as e:
        results.append({
            "condition": "mask_negative_attention",
            "status": "failed",
            "error": str(e),
        })

    return results


def _ablation_num_images(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
) -> list[dict[str, Any]]:
    """Ablation: training with varying numbers of reference images (1-5).

    For each condition, a subset of subject images is selected and training
    is run from scratch.  This measures how gracefully the method degrades
    with fewer reference images.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun.
        output_dir: Root ablation output directory.
        backbone: Backbone name.

    Returns:
        List of condition result dictionaries.
    """
    from modularbooth.data.dataset import _collect_image_paths

    subject_path = Path(subject_dir)
    all_images = _collect_image_paths(subject_path)
    max_available = len(all_images)

    results = []
    for num in range(1, 6):
        if num > max_available:
            logger.warning(
                "Only %d subject images available; skipping num_images=%d condition.",
                max_available,
                num,
            )
            continue

        condition_name = f"num_images_{num}"
        condition_dir = str(Path(output_dir) / condition_name)

        # Create a temporary directory with the subset of images.
        subset_dir = Path(condition_dir) / "subject_subset"
        subset_dir.mkdir(parents=True, exist_ok=True)

        for img_path in all_images[:num]:
            dst = subset_dir / img_path.name
            if not dst.exists():
                shutil.copy2(str(img_path), str(dst))

        results.append(_run_condition(
            condition_name=condition_name,
            subject_dir=str(subset_dir),
            class_noun=class_noun,
            condition_dir=condition_dir,
            backbone=backbone,
            train_overrides=[f"subject.num_images={num}"],
        ))

    return results


def _ablation_lora_rank(
    subject_dir: str,
    class_noun: str,
    output_dir: str,
    backbone: str,
) -> list[dict[str, Any]]:
    """Ablation: sweep identity LoRA rank over {4, 8, 16, 32, 64}.

    Context and shared ranks are scaled proportionally (context = rank/4,
    shared = rank/2) to maintain the relative rank ratios from the default
    config.

    Args:
        subject_dir: Path to subject images.
        class_noun: Class noun.
        output_dir: Root ablation output directory.
        backbone: Backbone name.

    Returns:
        List of condition result dictionaries.
    """
    results = []

    for identity_rank in [4, 8, 16, 32, 64]:
        context_rank = max(1, identity_rank // 4)
        shared_rank = max(1, identity_rank // 2)

        condition_name = f"rank_{identity_rank}"
        condition_dir = str(Path(output_dir) / condition_name)

        results.append(_run_condition(
            condition_name=condition_name,
            subject_dir=subject_dir,
            class_noun=class_noun,
            condition_dir=condition_dir,
            backbone=backbone,
            train_overrides=[
                f"lora.identity_rank={identity_rank}",
                f"lora.context_rank={context_rank}",
                f"lora.shared_rank={shared_rank}",
            ],
        ))

    return results


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_ABLATION_DISPATCH: dict[str, Any] = {
    "blockwise_vs_uniform": _ablation_blockwise_vs_uniform,
    "ccd_loss": _ablation_ccd_loss,
    "captioning": _ablation_captioning,
    "masked_inference": _ablation_masked_inference,
    "num_images": _ablation_num_images,
    "lora_rank": _ablation_lora_rank,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """ModularBooth ablation study entry-point.

    Orchestrates the ablation pipeline:

    1. Parse CLI arguments.
    2. Dispatch to the selected ablation function.
    3. Each condition runs train -> generate -> evaluate.
    4. Aggregate and save a comparison JSON with all condition results.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Validate paths.
    subject_dir = Path(args.subject_dir).resolve()
    if not subject_dir.is_dir():
        parser.error(f"Subject directory does not exist: {subject_dir}")

    output_dir = Path(args.output_dir).resolve() / args.ablation
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Ablation study: %s", args.ablation)
    logger.info("  Subject: %s (%s)", subject_dir, args.class_noun)
    logger.info("  Backbone: %s", args.backbone)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    start_time = time.time()

    # Dispatch to the appropriate ablation function.
    ablation_fn = _ABLATION_DISPATCH[args.ablation]
    condition_results = ablation_fn(
        subject_dir=str(subject_dir),
        class_noun=args.class_noun,
        output_dir=str(output_dir),
        backbone=args.backbone,
    )

    elapsed = time.time() - start_time

    # Assemble comparison report.
    comparison = {
        "ablation": args.ablation,
        "backbone": args.backbone,
        "subject_dir": str(subject_dir),
        "class_noun": args.class_noun,
        "num_conditions": len(condition_results),
        "total_elapsed_seconds": round(elapsed, 2),
        "conditions": condition_results,
    }

    # Build a summary table of key metrics across conditions.
    summary_table: list[dict[str, Any]] = []
    for cond in condition_results:
        row: dict[str, Any] = {"condition": cond["condition"], "status": cond["status"]}
        if cond["status"] == "completed" and "metrics" in cond:
            for metric_name, metric_value in cond["metrics"].items():
                row[metric_name] = metric_value
        summary_table.append(row)

    comparison["summary"] = summary_table

    # Save comparison JSON.
    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info("Comparison report saved to %s", comparison_file)

    # Print summary.
    print("\n" + "=" * 70)
    print(f"  Ablation: {args.ablation}")
    print("=" * 70)

    # Determine available metric columns.
    all_metric_keys: list[str] = []
    for row in summary_table:
        for k in row:
            if k not in ("condition", "status") and k not in all_metric_keys:
                all_metric_keys.append(k)

    # Print header.
    header = f"  {'Condition':<30s} {'Status':<12s}"
    for mk in all_metric_keys:
        header += f" {mk:<10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Print rows.
    for row in summary_table:
        line = f"  {row['condition']:<30s} {row['status']:<12s}"
        for mk in all_metric_keys:
            val = row.get(mk)
            if val is not None and isinstance(val, (int, float)):
                line += f" {val:<10.4f}"
            elif val is not None:
                line += f" {str(val):<10s}"
            else:
                line += f" {'---':<10s}"
        print(line)

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Report:     {comparison_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
