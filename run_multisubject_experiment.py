"""Multi-subject composition experiments for ModularBooth.

This script:
1. Generates cat subject/class images using SD 1.4
2. Trains a cat LoRA (blockwise + CCD)
3. Runs multi-subject composition with dog+cat LoRAs
4. Evaluates composition quality (IIS, DINO, CLIP-T, etc.)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("multisubject")

DEVICE = "cuda"
OUTPUT_BASE = Path("outputs/experiments")
CAT_SUBJECT_DIR = Path("outputs/cat_training/subject_images")
CAT_CLASS_DIR = Path("outputs/cat_training/class_images")
DOG_SUBJECT_DIR = Path("outputs/sd14_training/subject_images")
DOG_CLASS_DIR = Path("outputs/sd14_training/class_images")
DOG_LORA = Path("outputs/experiments/blockwise_ccd/lora_weights.safetensors")
MULTI_OUTPUT = OUTPUT_BASE / "multi_subject"


def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cast_lora_to_fp32(lora):
    for lora_mod in lora._lora_modules.values():
        lora_mod.lora_A.data = lora_mod.lora_A.data.float()
        lora_mod.lora_B.data = lora_mod.lora_B.data.float()
        lora_mod.lora_A.requires_grad_(True)
        lora_mod.lora_B.requires_grad_(True)


# ---------------------------------------------------------------
# Step 1: Generate cat training data
# ---------------------------------------------------------------

def generate_cat_images():
    """Generate cat subject and class images using SD 1.4."""
    if CAT_SUBJECT_DIR.exists() and len(list(CAT_SUBJECT_DIR.glob("*.png"))) >= 4:
        logger.info("Cat subject images already exist, skipping.")
        return

    logger.info("Generating cat subject and class images...")
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    # Generate subject images (sks cat)
    CAT_SUBJECT_DIR.mkdir(parents=True, exist_ok=True)
    subject_prompts = [
        "a portrait photo of a sks cat, detailed fur, studio lighting",
        "a photo of a sks cat sitting in a garden",
        "a photo of a sks cat on a wooden table, close up",
        "a photo of a sks cat looking at camera, natural light",
    ]
    for i, prompt in enumerate(subject_prompts):
        gen = torch.Generator(device=DEVICE).manual_seed(7777 + i)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=gen,
            )
        result.images[0].save(CAT_SUBJECT_DIR / f"subject_{i:02d}.png")
        logger.info("Cat subject %d saved.", i)

    # Generate class images (generic cats)
    CAT_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    class_prompts = [
        "a photo of a cat",
        "a photo of a cat in a house",
        "a photo of a cat outdoors",
        "a photo of a cute cat",
        "a photo of a cat, high quality",
        "a photo of a cat sitting",
        "a photo of a cat sleeping",
        "a photo of a cat playing",
    ]
    for i, prompt in enumerate(class_prompts):
        gen = torch.Generator(device=DEVICE).manual_seed(8888 + i)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=gen,
            )
        result.images[0].save(CAT_CLASS_DIR / f"class_{i:02d}.png")

    logger.info("Generated %d cat subject + %d class images.", 4, 8)

    del pipe
    gpu_cleanup()


# ---------------------------------------------------------------
# Step 2: Train cat LoRA
# ---------------------------------------------------------------

def train_cat_lora() -> Path:
    """Train a blockwise+CCD LoRA for the cat subject. Returns path to weights."""
    cat_lora_path = OUTPUT_BASE / "cat_blockwise_ccd" / "lora_weights.safetensors"
    if cat_lora_path.exists():
        logger.info("Cat LoRA already exists at %s, skipping.", cat_lora_path)
        return cat_lora_path

    logger.info("Training cat LoRA (blockwise + CCD)...")
    gpu_cleanup()

    from diffusers import StableDiffusionPipeline
    from modularbooth.data.dataset import DreamBoothDataset
    from modularbooth.losses.combined import ModularBoothLoss
    from modularbooth.models.blockwise_lora import BlockwiseLoRA
    from modularbooth.training.trainer import ModularBoothTrainer

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    # Create augmented images for CCD
    aug_dir = OUTPUT_BASE / "cat_augmented_images"
    if not aug_dir.exists():
        aug_dir.mkdir(parents=True, exist_ok=True)
        jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,
        )
        for img_path in sorted(CAT_SUBJECT_DIR.glob("*.png")):
            variant_dir = aug_dir / img_path.stem
            variant_dir.mkdir(exist_ok=True)
            img = Image.open(img_path).convert("RGB")
            for v in range(3):
                torch.manual_seed(v * 1000 + hash(img_path.stem) % 10000)
                augmented = jitter(img)
                augmented.save(variant_dir / f"variant_{v:02d}.png")

    # Dataset
    dataset = DreamBoothDataset(
        subject_images_dir=str(CAT_SUBJECT_DIR),
        class_images_dir=str(CAT_CLASS_DIR),
        token="[V]",
        class_noun="cat",
        resolution=512,
        augmented_images_dir=str(aug_dir),
    )

    # Discover block structure
    block_indices = set()
    for mod_name, _ in pipe.unet.named_modules():
        match = re.search(r"transformer_blocks\.(\d+)\.", mod_name)
        if match:
            block_indices.add(int(match.group(1)))
    sorted_blocks = sorted(block_indices)
    n_blocks = len(sorted_blocks)

    blockwise_config = {}
    for i, idx in enumerate(sorted_blocks):
        if i < n_blocks // 3:
            blockwise_config[idx] = "identity"
        elif i >= 2 * n_blocks // 3:
            blockwise_config[idx] = "context"
        else:
            blockwise_config[idx] = "shared"

    lora = BlockwiseLoRA(
        model=pipe.unet,
        block_config=blockwise_config,
        identity_rank=16,
        context_rank=4,
        shared_rank=8,
        alpha_ratio=1.0,
        dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()
    cast_lora_to_fp32(lora)

    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "cat", "num_images": 4},
        "lora": {"identity_rank": 16, "context_rank": 4, "shared_rank": 8},
        "training": {
            "num_steps": 500,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "warmup_steps": 20,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "seed": 42,
            "log_every": 50,
            "save_every": 500,
            "validate_every": 500,
            "scheduler": "cosine",
        },
        "prior_preservation": {
            "enabled": True,
            "num_class_images": 8,
            "lambda_ppl": 1.0,
        },
        "ccd": {
            "enabled": True,
            "lambda_ccd": 0.3,
            "temperature": 0.07,
            "warmup_steps": 100,
            "feature_layer": "middle",
        },
        "inference": {"num_steps": 30, "guidance_scale": 7.5, "resolution": 512},
    })

    loss_fn = ModularBoothLoss(
        lambda_ppl=1.0,
        lambda_ccd=0.3,
        ccd_warmup_steps=100,
    )

    trainer = ModularBoothTrainer(
        config=cfg,
        model=pipe,
        lora=lora,
        dataset=dataset,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    t0 = time.time()
    results = trainer.train()
    train_time = time.time() - t0
    logger.info("Cat training: %.1fs, loss=%.4f", train_time, results.get("loss_total", 0))

    # Check norms
    total_b, n = 0.0, 0
    for lora_mod in lora._lora_modules.values():
        total_b += lora_mod.lora_B.data.norm().item()
        n += 1
    logger.info("Cat LoRA avg |B|=%.6f (%d modules)", total_b / n if n else 0, n)

    # Save
    out_dir = OUTPUT_BASE / "cat_blockwise_ccd"
    out_dir.mkdir(parents=True, exist_ok=True)
    lora.save_lora(str(cat_lora_path))
    logger.info("Cat LoRA saved: %s (%.2f MB)", cat_lora_path, cat_lora_path.stat().st_size / 1e6)

    del trainer, pipe, lora, loss_fn, dataset
    gpu_cleanup()

    return cat_lora_path


# ---------------------------------------------------------------
# Step 3: Multi-subject composition (for SD 1.4 UNet)
# ---------------------------------------------------------------

def multi_subject_generation(
    dog_lora_path: Path,
    cat_lora_path: Path,
    output_dir: Path,
    mode: str = "masked",
) -> list[tuple[str, str]]:
    """Generate multi-subject images with dog + cat.

    Args:
        dog_lora_path: Path to dog LoRA weights.
        cat_lora_path: Path to cat LoRA weights.
        output_dir: Directory for generated images.
        mode: 'masked' (spatial masking) or 'naive' (simple LoRA addition).

    Returns:
        List of (image_path, prompt) tuples.
    """
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file as load_safetensors

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SD 1.4 for multi-subject generation (mode=%s)...", mode)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    # Load LoRA state dicts
    dog_sd = load_safetensors(str(dog_lora_path))
    cat_sd = load_safetensors(str(cat_lora_path))

    # Move to device
    dog_sd = {k: v.to(DEVICE) for k, v in dog_sd.items()}
    cat_sd = {k: v.to(DEVICE) for k, v in cat_sd.items()}

    # Prompts for multi-subject generation
    prompts = [
        "a photo of a dog and a cat sitting together on a beach",
        "a photo of a dog and a cat in a garden with flowers",
        "a painting of a dog and a cat playing in a park",
        "a photo of a dog on the left and a cat on the right in a living room",
        "a photo of a dog and a cat side by side on grass",
        "a photo of a dog and a cat together in a studio portrait",
    ]

    results = []
    for p_idx, prompt in enumerate(prompts):
        for seed_idx in range(2):
            seed = 5000 + p_idx * 100 + seed_idx
            gen = torch.Generator(device=DEVICE).manual_seed(seed)

            if mode == "naive":
                # Naive: add both LoRAs to the UNet simultaneously
                _apply_lora_to_unet(pipe.unet, dog_sd)
                _apply_lora_to_unet(pipe.unet, cat_sd)

                with torch.no_grad():
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=gen,
                    )

                _remove_lora_from_unet(pipe.unet, cat_sd)
                _remove_lora_from_unet(pipe.unet, dog_sd)

            elif mode == "masked":
                # Masked: custom denoising loop with spatial masking
                image = _masked_denoising(
                    pipe=pipe,
                    prompt=prompt,
                    dog_sd=dog_sd,
                    cat_sd=cat_sd,
                    seed=seed,
                )
                # Wrap to match pipeline output format
                class _Result:
                    def __init__(self, imgs):
                        self.images = imgs
                result = _Result([image])

            fname = f"{mode}_{p_idx:02d}_{seed_idx:02d}.png"
            path = output_dir / fname
            result.images[0].save(path)
            results.append((str(path), prompt))

    del pipe, dog_sd, cat_sd
    gpu_cleanup()

    logger.info("Generated %d %s images.", len(results), mode)
    return results


def _apply_lora_to_unet(unet, lora_sd):
    """Add LoRA deltas to UNet weights in-place."""
    modules = {}
    for key, tensor in lora_sd.items():
        if key.endswith(".lora_A"):
            name = key[: -len(".lora_A")]
            modules.setdefault(name, {})["lora_A"] = tensor
        elif key.endswith(".lora_B"):
            name = key[: -len(".lora_B")]
            modules.setdefault(name, {})["lora_B"] = tensor

    for name, params in modules.items():
        if "lora_A" not in params or "lora_B" not in params:
            continue
        target = _get_submodule(unet, name)
        if target is None or not hasattr(target, "weight"):
            continue
        A = params["lora_A"]
        B = params["lora_B"]
        delta = B @ A
        target.weight.data.add_(delta.to(target.weight.dtype))


def _remove_lora_from_unet(unet, lora_sd):
    """Remove LoRA deltas from UNet weights in-place."""
    modules = {}
    for key, tensor in lora_sd.items():
        if key.endswith(".lora_A"):
            name = key[: -len(".lora_A")]
            modules.setdefault(name, {})["lora_A"] = tensor
        elif key.endswith(".lora_B"):
            name = key[: -len(".lora_B")]
            modules.setdefault(name, {})["lora_B"] = tensor

    for name, params in modules.items():
        if "lora_A" not in params or "lora_B" not in params:
            continue
        target = _get_submodule(unet, name)
        if target is None or not hasattr(target, "weight"):
            continue
        A = params["lora_A"]
        B = params["lora_B"]
        delta = B @ A
        target.weight.data.sub_(delta.to(target.weight.dtype))


def _get_submodule(module, name):
    parts = name.split(".")
    cur = module
    for p in parts:
        if hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            return None
    return cur


def _masked_denoising(
    pipe,
    prompt: str,
    dog_sd: dict,
    cat_sd: dict,
    seed: int,
    neg_attn_strength: float = 3.0,
    mask_leakage_alpha: float = 0.05,
) -> Image.Image:
    """Custom denoising loop with spatial masking for two subjects.

    Dog is placed on the left half, cat on the right half.
    """
    # Layout: dog left half, cat right half
    latent_h, latent_w = 64, 64  # 512/8
    dog_mask = torch.zeros(1, 1, latent_h, latent_w, device=DEVICE, dtype=torch.float16)
    cat_mask = torch.zeros(1, 1, latent_h, latent_w, device=DEVICE, dtype=torch.float16)
    dog_mask[:, :, :, : latent_w // 2] = 1.0
    cat_mask[:, :, :, latent_w // 2 :] = 1.0

    # Encode prompt
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(DEVICE)
    with torch.no_grad():
        encoder_output = text_encoder(input_ids)
    prompt_embeds = encoder_output.last_hidden_state

    # Unconditional embeddings for CFG
    uncond_inputs = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_output = text_encoder(uncond_inputs.input_ids.to(DEVICE))
    uncond_embeds = uncond_output.last_hidden_state

    # Initialize latents
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    in_channels = pipe.unet.config.in_channels
    latents = torch.randn(
        (1, in_channels, latent_h, latent_w),
        generator=gen,
        device=DEVICE,
        dtype=torch.float16,
    )

    # Set up scheduler
    pipe.scheduler.set_timesteps(30, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    latents = latents * pipe.scheduler.init_noise_sigma

    guidance_scale = 7.5

    for t in timesteps:
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)

        # --- Base prediction (no LoRA) ---
        with torch.no_grad():
            base_noise = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            uncond_noise = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_embeds,
            ).sample

        # --- Dog prediction (with dog LoRA) ---
        _apply_lora_to_unet(pipe.unet, dog_sd)
        with torch.no_grad():
            dog_noise = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
        _remove_lora_from_unet(pipe.unet, dog_sd)

        # --- Cat prediction (with cat LoRA) ---
        _apply_lora_to_unet(pipe.unet, cat_sd)
        with torch.no_grad():
            cat_noise = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
        _remove_lora_from_unet(pipe.unet, cat_sd)

        # --- Spatial blending ---
        dog_delta = dog_noise - base_noise
        cat_delta = cat_noise - base_noise

        dog_soft = dog_mask * (1.0 - mask_leakage_alpha) + mask_leakage_alpha
        cat_soft = cat_mask * (1.0 - mask_leakage_alpha) + mask_leakage_alpha

        blended_cond = base_noise + dog_delta * dog_soft + cat_delta * cat_soft

        # --- Negative attention in background ---
        combined_mask = torch.clamp(dog_mask + cat_mask, 0.0, 1.0)
        bg_mask = 1.0 - combined_mask
        if bg_mask.sum() > 0:
            avg_delta = (dog_delta + cat_delta) / 2.0
            blended_cond = blended_cond - neg_attn_strength * avg_delta * bg_mask

        # --- Classifier-free guidance ---
        noise_pred = uncond_noise + guidance_scale * (blended_cond - uncond_noise)

        # --- Scheduler step ---
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    scaling_factor = pipe.vae.config.scaling_factor
    latents_scaled = latents / scaling_factor
    with torch.no_grad():
        image_tensor = pipe.vae.decode(latents_scaled).sample

    image_tensor = (image_tensor / 2.0 + 0.5).clamp(0.0, 1.0)
    image_tensor = image_tensor.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
    image_array = (image_tensor * 255.0).round().astype("uint8")
    return Image.fromarray(image_array)


# ---------------------------------------------------------------
# Step 4: Evaluate multi-subject results
# ---------------------------------------------------------------

def evaluate_multi_subject(
    gen_results: list[tuple[str, str]],
    dog_subject_dir: Path,
    cat_subject_dir: Path,
    label: str,
) -> dict:
    """Evaluate multi-subject generated images.

    Metrics:
    - DINO (subject fidelity for each subject, cropped)
    - CLIP-T (prompt fidelity)
    - IIS (Identity Isolation Score): how well each subject stays
      within its spatial region
    - Composition Accuracy: whether both subjects are present
    """
    from modularbooth.evaluation.clip_score import CLIPScore
    from modularbooth.evaluation.dino_score import DINOScore

    metrics = {}
    gen_images = [Image.open(p).convert("RGB") for p, _ in gen_results]
    gen_prompts = [prompt for _, prompt in gen_results]

    # Load subject reference images
    dog_refs = [Image.open(p).convert("RGB") for p in sorted(dog_subject_dir.glob("*.png"))]
    cat_refs = [Image.open(p).convert("RGB") for p in sorted(cat_subject_dir.glob("*.png"))]

    # ---- CLIP-T (prompt fidelity) ----
    try:
        clip = CLIPScore(device=DEVICE)
        metrics["clip_t"] = clip.clip_t_score(gen_images, gen_prompts)
        logger.info("[%s] CLIP-T: %.4f", label, metrics["clip_t"])
        del clip
        gpu_cleanup()
    except Exception as e:
        logger.warning("CLIP-T failed: %s", e)

    # ---- DINO: full image similarity to dog and cat refs ----
    try:
        dino = DINOScore(device=DEVICE)

        # Full image DINO vs dog refs
        metrics["dino_vs_dog"] = dino.compute_score(gen_images, dog_refs)
        metrics["dino_vs_cat"] = dino.compute_score(gen_images, cat_refs)
        logger.info("[%s] DINO vs dog: %.4f, vs cat: %.4f",
                     label, metrics["dino_vs_dog"], metrics["dino_vs_cat"])

        # ---- IIS: Identity Isolation Score ----
        # Crop left/right halves of generated images
        # Dog should be on left, cat on right
        left_crops = []
        right_crops = []
        for img in gen_images:
            w, h = img.size
            left_crops.append(img.crop((0, 0, w // 2, h)))
            right_crops.append(img.crop((w // 2, 0, w, h)))

        # Dog in left crop should be similar to dog refs
        dog_left_sim = dino.compute_score(left_crops, dog_refs)
        # Cat in right crop should be similar to cat refs
        cat_right_sim = dino.compute_score(right_crops, cat_refs)
        # Cross-contamination: dog in right (should be low)
        dog_right_sim = dino.compute_score(right_crops, dog_refs)
        # Cross-contamination: cat in left (should be low)
        cat_left_sim = dino.compute_score(left_crops, cat_refs)

        # IIS = correct_region_similarity - wrong_region_similarity
        dog_iis = dog_left_sim - dog_right_sim
        cat_iis = cat_right_sim - cat_left_sim
        metrics["iis"] = (dog_iis + cat_iis) / 2.0

        metrics["dog_correct_sim"] = dog_left_sim
        metrics["cat_correct_sim"] = cat_right_sim
        metrics["dog_cross_sim"] = dog_right_sim
        metrics["cat_cross_sim"] = cat_left_sim

        logger.info("[%s] IIS: %.4f (dog_correct=%.3f, cat_correct=%.3f, "
                     "dog_cross=%.3f, cat_cross=%.3f)",
                     label, metrics["iis"],
                     dog_left_sim, cat_right_sim,
                     dog_right_sim, cat_left_sim)

        # ---- Composition Accuracy (via DINO embedding diversity) ----
        # Check that left and right crops are different (both subjects present)
        comp_scores = []
        for lc, rc in zip(left_crops, right_crops):
            # If both halves look different, both subjects are likely present
            cross_sim = dino.compute_score([lc], [rc])
            # Score: lower cross-sim => better composition (distinct subjects)
            comp_scores.append(1.0 - cross_sim)
        metrics["composition_diversity"] = sum(comp_scores) / len(comp_scores)
        logger.info("[%s] Composition diversity: %.4f", label, metrics["composition_diversity"])

        del dino
        gpu_cleanup()
    except Exception as e:
        logger.warning("DINO/IIS eval failed: %s", e)

    return metrics


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def main():
    start = time.time()
    MULTI_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate cat images
    logger.info("=" * 60)
    logger.info("STEP 1: Generate cat training images")
    logger.info("=" * 60)
    generate_cat_images()
    gpu_cleanup()

    # Step 2: Train cat LoRA
    logger.info("=" * 60)
    logger.info("STEP 2: Train cat LoRA (blockwise + CCD)")
    logger.info("=" * 60)
    cat_lora_path = train_cat_lora()
    gpu_cleanup()

    # Step 3: Multi-subject generation
    logger.info("=" * 60)
    logger.info("STEP 3: Multi-subject generation")
    logger.info("=" * 60)

    # 3a: Masked composition (our method)
    masked_dir = MULTI_OUTPUT / "masked"
    masked_results = multi_subject_generation(
        DOG_LORA, cat_lora_path, masked_dir, mode="masked",
    )
    gpu_cleanup()

    # 3b: Naive LoRA addition (baseline)
    naive_dir = MULTI_OUTPUT / "naive"
    naive_results = multi_subject_generation(
        DOG_LORA, cat_lora_path, naive_dir, mode="naive",
    )
    gpu_cleanup()

    # Step 4: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluate multi-subject results")
    logger.info("=" * 60)

    masked_metrics = evaluate_multi_subject(
        masked_results, DOG_SUBJECT_DIR, CAT_SUBJECT_DIR, "masked",
    )
    gpu_cleanup()

    naive_metrics = evaluate_multi_subject(
        naive_results, DOG_SUBJECT_DIR, CAT_SUBJECT_DIR, "naive",
    )
    gpu_cleanup()

    # Save all results
    all_results = {
        "masked": masked_metrics,
        "naive": naive_metrics,
        "total_time_s": time.time() - start,
    }

    results_path = MULTI_OUTPUT / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("MULTI-SUBJECT RESULTS")
    logger.info("=" * 60)
    logger.info("Masked composition: %s", json.dumps(masked_metrics, indent=2))
    logger.info("Naive LoRA addition: %s", json.dumps(naive_metrics, indent=2))
    logger.info("Results saved to %s", results_path)
    logger.info("Total time: %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
