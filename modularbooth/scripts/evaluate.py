#!/usr/bin/env python3
"""ModularBooth evaluation entry-point.

Computes quantitative metrics comparing generated images against reference
subject images.  Supports both subject-fidelity metrics (DINO, DINOv2,
CLIP-I, LPIPS) and prompt-fidelity metrics (CLIP-T) as well as composite
scores (CAE, IIS) and VQA-based evaluation.

Usage::

    python -m modularbooth.scripts.evaluate \
        --generated-dir ./outputs/generated \
        --reference-dir ./data/dog \
        --metrics all \
        --output-file metrics.json

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
from PIL import Image
from omegaconf import DictConfig, OmegaConf

from modularbooth.configs import load_config

logger = logging.getLogger(__name__)

# Supported image extensions for directory scanning.
_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# All individually selectable metric names.
_METRIC_CHOICES: list[str] = [
    "dino",
    "dinov2",
    "clip_i",
    "clip_t",
    "lpips",
    "cae",
    "iis",
    "vqa",
    "all",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the evaluation script.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated images against reference subjects using "
            "subject-fidelity and prompt-fidelity metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--generated-dir",
        type=str,
        required=True,
        help="Path to directory containing generated images.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        required=True,
        help="Path to directory containing reference subject images.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file containing the prompts used for "
            "generation. Required for CLIP-T and VQA metrics."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="metrics.json",
        help="Path to save the JSON metrics report (default: metrics.json).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=_METRIC_CHOICES,
        default=["all"],
        help=(
            "Which metrics to compute.  Specify one or more of: "
            f"{', '.join(_METRIC_CHOICES)}.  Default: all."
        ),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="flux",
        help='Backbone name for loading evaluation config (default: "flux").',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Torch device for metric computation (default: "cuda").',
    )

    return parser


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_images_from_dir(directory: Path) -> list[Image.Image]:
    """Load all images from a directory, sorted by filename.

    Args:
        directory: Path to a directory containing image files.

    Returns:
        List of PIL Image objects in RGB mode.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no images.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")

    images = [Image.open(p).convert("RGB") for p in paths]
    logger.info("Loaded %d images from %s", len(images), directory)
    return images


def _load_prompts(prompts_file: str | None) -> list[str] | None:
    """Load prompts from a JSON file if provided.

    The JSON file is expected to contain either a list of strings or a dict
    mapping prompt identifiers to prompt strings.

    Args:
        prompts_file: Path to the prompts JSON file, or ``None``.

    Returns:
        List of prompt strings, or ``None`` if no file was provided.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the JSON structure is not supported.
    """
    if prompts_file is None:
        return None

    path = Path(prompts_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(p) for p in data]
    if isinstance(data, dict):
        return [str(v) for v in data.values()]

    raise ValueError(
        f"Prompts JSON must be a list of strings or a dict, got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _resolve_metrics(selected: list[str]) -> set[str]:
    """Expand the ``"all"`` shortcut and return a deduplicated set of metric names.

    Args:
        selected: List of metric name strings from CLI.

    Returns:
        Set of individual metric names to compute.
    """
    if "all" in selected:
        return {"dino", "dinov2", "clip_i", "clip_t", "lpips", "cae", "iis", "vqa"}
    return set(selected)


def _compute_dino(
    generated: list[Image.Image],
    reference: list[Image.Image],
    device: str,
    model_name: str = "facebook/dino-vits16",
) -> float:
    """Compute the DINO subject-fidelity score.

    Args:
        generated: Generated images.
        reference: Reference subject images.
        device: Torch device string.
        model_name: DINO model identifier.

    Returns:
        Mean pairwise cosine similarity.
    """
    from modularbooth.evaluation.dino_score import DINOScore

    scorer = DINOScore(model_name=model_name, device=device)
    return scorer.compute_score(generated, reference)


def _compute_dinov2(
    generated: list[Image.Image],
    reference: list[Image.Image],
    device: str,
    model_name: str = "facebook/dinov2-vitb14",
) -> float:
    """Compute the DINOv2 subject-fidelity score.

    Args:
        generated: Generated images.
        reference: Reference subject images.
        device: Torch device string.
        model_name: DINOv2 model identifier.

    Returns:
        Mean pairwise cosine similarity.
    """
    from modularbooth.evaluation.dino_score import DINOv2Score

    scorer = DINOv2Score(model_name=model_name, device=device)
    return scorer.compute_score(generated, reference)


def _compute_clip_i(
    generated: list[Image.Image],
    reference: list[Image.Image],
    device: str,
    model_name: str = "openai/clip-vit-large-patch14",
) -> float:
    """Compute CLIP-I (image-image similarity) between generated and reference images.

    Uses CLIP image encoder embeddings and computes mean pairwise cosine
    similarity, analogous to DINO-I but with CLIP features.

    Args:
        generated: Generated images.
        reference: Reference subject images.
        device: Torch device string.
        model_name: CLIP model identifier.

    Returns:
        Mean pairwise cosine similarity in CLIP image space.
    """
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    with torch.no_grad():
        gen_inputs = processor(images=generated, return_tensors="pt", padding=True)
        gen_inputs = {k: v.to(device) for k, v in gen_inputs.items() if isinstance(v, torch.Tensor)}
        gen_embeds = model.get_image_features(**gen_inputs)
        gen_embeds = F.normalize(gen_embeds, dim=-1)

        ref_inputs = processor(images=reference, return_tensors="pt", padding=True)
        ref_inputs = {k: v.to(device) for k, v in ref_inputs.items() if isinstance(v, torch.Tensor)}
        ref_embeds = model.get_image_features(**ref_inputs)
        ref_embeds = F.normalize(ref_embeds, dim=-1)

    similarity_matrix = gen_embeds @ ref_embeds.T
    return similarity_matrix.mean().item()


def _compute_clip_t(
    generated: list[Image.Image],
    prompts: list[str],
    device: str,
    model_name: str = "openai/clip-vit-large-patch14",
) -> float:
    """Compute CLIP-T (text-image alignment) between generated images and prompts.

    Each generated image is compared against its corresponding prompt to
    measure how well the generation follows the text description.

    Args:
        generated: Generated images.
        prompts: Text prompts corresponding to the generated images.
        device: Torch device string.
        model_name: CLIP model identifier.

    Returns:
        Mean cosine similarity between image and text embeddings.
    """
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    # Expand prompts to match generated images if there are more images than prompts.
    if len(prompts) < len(generated):
        expanded_prompts = []
        for i in range(len(generated)):
            expanded_prompts.append(prompts[i % len(prompts)])
        prompts = expanded_prompts

    with torch.no_grad():
        img_inputs = processor(images=generated, return_tensors="pt", padding=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items() if isinstance(v, torch.Tensor)}
        img_embeds = model.get_image_features(**img_inputs)
        img_embeds = F.normalize(img_embeds, dim=-1)

        txt_inputs = processor(text=prompts[:len(generated)], return_tensors="pt", padding=True)
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items() if isinstance(v, torch.Tensor)}
        txt_embeds = model.get_text_features(**txt_inputs)
        txt_embeds = F.normalize(txt_embeds, dim=-1)

    # Compute per-image cosine similarity with corresponding prompt.
    per_image_sim = (img_embeds * txt_embeds).sum(dim=-1)
    return per_image_sim.mean().item()


def _compute_lpips(
    generated: list[Image.Image],
    reference: list[Image.Image],
    device: str,
) -> float:
    """Compute LPIPS perceptual distance between generated and reference images.

    Lower LPIPS values indicate higher perceptual similarity.  We report the
    mean pairwise LPIPS distance across all (generated, reference) pairs.

    Args:
        generated: Generated images.
        reference: Reference subject images.
        device: Torch device string.

    Returns:
        Mean LPIPS distance (lower is better for fidelity).
    """
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    total_dist = 0.0
    num_pairs = 0

    with torch.no_grad():
        for gen_img in generated:
            gen_tensor = transform(gen_img).unsqueeze(0).to(device)
            for ref_img in reference:
                ref_tensor = transform(ref_img).unsqueeze(0).to(device)
                dist = loss_fn(gen_tensor, ref_tensor)
                total_dist += dist.item()
                num_pairs += 1

    return total_dist / max(num_pairs, 1)


def _compute_cae(
    generated: list[Image.Image],
    reference: list[Image.Image],
    device: str,
) -> float:
    """Compute Context Appearance Entanglement (CAE) score.

    CAE measures how well the model preserves subject identity while varying
    context.  It is computed as the ratio of intra-subject DINO similarity
    (among generated images of the same subject in different contexts) to
    inter-subject similarity.  Higher CAE indicates better disentanglement.

    For single-subject evaluation this is approximated by measuring the
    variance of DINO similarity across different generated contexts.

    Args:
        generated: Generated images (ideally from diverse prompts).
        reference: Reference subject images.
        device: Torch device string.

    Returns:
        CAE score (higher is better).
    """
    from modularbooth.evaluation.dino_score import DINOv2Score

    scorer = DINOv2Score(device=device)

    # Intra-identity similarity: how consistent are the generated images?
    if len(generated) > 1:
        gen_embeds = scorer.compute_embeddings(generated)
        intra_sim = (gen_embeds @ gen_embeds.T).fill_diagonal_(0)
        n = len(generated)
        intra_score = intra_sim.sum().item() / max(n * (n - 1), 1)
    else:
        intra_score = 1.0

    # Cross-identity similarity to reference.
    cross_score = scorer.compute_score(generated, reference)

    # CAE: identity preservation relative to context variation.
    # A perfectly disentangled model has high cross_score and high intra_score.
    cae = (intra_score + cross_score) / 2.0
    return cae


def _compute_iis(
    generated: list[Image.Image],
    reference: list[Image.Image],
    device: str,
) -> float:
    """Compute Identity-Isolation Score (IIS).

    IIS combines DINO and CLIP-I similarity to measure how well the generated
    images capture the specific identity of the reference subject versus
    generic class features.  It is the harmonic mean of DINO-I and CLIP-I.

    Args:
        generated: Generated images.
        reference: Reference subject images.
        device: Torch device string.

    Returns:
        IIS score in [0, 1] (higher is better).
    """
    dino_score = _compute_dinov2(generated, reference, device)
    clip_i_score = _compute_clip_i(generated, reference, device)

    # Harmonic mean.
    if dino_score + clip_i_score > 0:
        iis = 2.0 * dino_score * clip_i_score / (dino_score + clip_i_score)
    else:
        iis = 0.0

    return iis


def _compute_vqa(
    generated: list[Image.Image],
    prompts: list[str],
    device: str,
) -> float:
    """Compute VQA-based prompt alignment score.

    Uses a Visual Question Answering model to assess whether the generated
    images match the intent of the text prompts.  For each generated image
    and its corresponding prompt, we ask "Does this image show [prompt]?"
    and measure the confidence of the "yes" answer.

    Args:
        generated: Generated images.
        prompts: Text prompts corresponding to the generated images.
        device: Torch device string.

    Returns:
        Mean VQA confidence score in [0, 1].
    """
    from transformers import pipeline as hf_pipeline

    vqa_pipe = hf_pipeline(
        "visual-question-answering",
        model="dandelin/vilt-b32-finetuned-vqa",
        device=0 if device == "cuda" else -1,
    )

    # Expand prompts to match number of generated images.
    if len(prompts) < len(generated):
        expanded = [prompts[i % len(prompts)] for i in range(len(generated))]
    else:
        expanded = prompts[:len(generated)]

    scores: list[float] = []
    for img, prompt in zip(generated, expanded):
        question = f"Does this image show {prompt}?"
        result = vqa_pipe(image=img, question=question, top_k=1)
        # The pipeline returns a list of dicts with 'score' and 'answer'.
        if result and isinstance(result, list):
            top_answer = result[0]
            if isinstance(top_answer, dict) and top_answer.get("answer", "").lower() == "yes":
                scores.append(top_answer.get("score", 0.0))
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """ModularBooth evaluation entry-point.

    Orchestrates the evaluation pipeline:

    1. Load configuration.
    2. Initialise selected metrics.
    3. Load generated and reference images.
    4. Compute all selected metrics.
    5. Save a JSON report.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Validate paths.
    generated_dir = Path(args.generated_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()

    if not generated_dir.is_dir():
        parser.error(f"Generated images directory does not exist: {generated_dir}")
    if not reference_dir.is_dir():
        parser.error(f"Reference images directory does not exist: {reference_dir}")

    output_file = Path(args.output_file).resolve()
    device = args.device

    # ---- 1. Load config ----
    logger.info("Loading config for backbone '%s'...", args.backbone)
    cfg = load_config(backbone=args.backbone)

    # ---- 2. Determine which metrics to compute ----
    metrics_to_compute = _resolve_metrics(args.metrics)
    logger.info("Metrics to compute: %s", sorted(metrics_to_compute))

    # Check if prompts are needed and available.
    prompts: list[str] | None = None
    text_metrics = {"clip_t", "vqa"}
    needs_prompts = bool(metrics_to_compute & text_metrics)

    if needs_prompts:
        prompts = _load_prompts(args.prompts_file)
        if prompts is None:
            logger.warning(
                "No --prompts-file provided but text-dependent metrics (%s) were "
                "requested. These metrics will be skipped.",
                sorted(metrics_to_compute & text_metrics),
            )
            metrics_to_compute -= text_metrics

    # ---- 3. Load images ----
    logger.info("Loading generated images from %s...", generated_dir)
    generated_images = _load_images_from_dir(generated_dir)

    logger.info("Loading reference images from %s...", reference_dir)
    reference_images = _load_images_from_dir(reference_dir)

    # ---- 4. Compute metrics ----
    results: dict[str, Any] = {
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "num_generated": len(generated_images),
        "num_reference": len(reference_images),
        "metrics": {},
    }
    start_time = time.time()

    eval_config = cfg.get("evaluation", {})

    if "dino" in metrics_to_compute:
        logger.info("Computing DINO score...")
        dino_model = getattr(eval_config, "dino_model", "facebook/dino-vits16")
        score = _compute_dino(generated_images, reference_images, device, dino_model)
        results["metrics"]["dino"] = round(score, 4)
        logger.info("  DINO: %.4f", score)

    if "dinov2" in metrics_to_compute:
        logger.info("Computing DINOv2 score...")
        dinov2_model = getattr(eval_config, "dinov2_model", "facebook/dinov2-vitb14")
        score = _compute_dinov2(generated_images, reference_images, device, dinov2_model)
        results["metrics"]["dinov2"] = round(score, 4)
        logger.info("  DINOv2: %.4f", score)

    if "clip_i" in metrics_to_compute:
        logger.info("Computing CLIP-I score...")
        clip_model = getattr(eval_config, "clip_model", "openai/clip-vit-large-patch14")
        score = _compute_clip_i(generated_images, reference_images, device, clip_model)
        results["metrics"]["clip_i"] = round(score, 4)
        logger.info("  CLIP-I: %.4f", score)

    if "clip_t" in metrics_to_compute and prompts is not None:
        logger.info("Computing CLIP-T score...")
        clip_model = getattr(eval_config, "clip_model", "openai/clip-vit-large-patch14")
        score = _compute_clip_t(generated_images, prompts, device, clip_model)
        results["metrics"]["clip_t"] = round(score, 4)
        logger.info("  CLIP-T: %.4f", score)

    if "lpips" in metrics_to_compute:
        logger.info("Computing LPIPS distance...")
        score = _compute_lpips(generated_images, reference_images, device)
        results["metrics"]["lpips"] = round(score, 4)
        logger.info("  LPIPS: %.4f", score)

    if "cae" in metrics_to_compute:
        logger.info("Computing CAE score...")
        score = _compute_cae(generated_images, reference_images, device)
        results["metrics"]["cae"] = round(score, 4)
        logger.info("  CAE: %.4f", score)

    if "iis" in metrics_to_compute:
        logger.info("Computing IIS score...")
        score = _compute_iis(generated_images, reference_images, device)
        results["metrics"]["iis"] = round(score, 4)
        logger.info("  IIS: %.4f", score)

    if "vqa" in metrics_to_compute and prompts is not None:
        logger.info("Computing VQA score...")
        score = _compute_vqa(generated_images, prompts, device)
        results["metrics"]["vqa"] = round(score, 4)
        logger.info("  VQA: %.4f", score)

    elapsed = time.time() - start_time
    results["evaluation_time_seconds"] = round(elapsed, 2)
    logger.info("Evaluation completed in %.1f seconds.", elapsed)

    # ---- 5. Save JSON report ----
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Metrics report saved to %s", output_file)

    # Print summary to stdout.
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for metric_name, metric_value in results["metrics"].items():
        print(f"  {metric_name:<12s}: {metric_value:.4f}")
    print(f"  {'time':<12s}: {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
