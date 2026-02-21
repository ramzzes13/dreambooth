"""Multi-subject composition using blockwise LoRAs WITHOUT CCD.

Compares to the CCD-trained multi-subject results to show CCD improves composition.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("multisubject_noccd")

DEVICE = "cuda"
OUTPUT_BASE = Path("outputs/experiments")
CAT_SUBJECT_DIR = Path("outputs/cat_training/subject_images")
CAT_CLASS_DIR = Path("outputs/cat_training/class_images")
DOG_SUBJECT_DIR = Path("outputs/sd14_training/subject_images")
DOG_LORA_NOCCD = Path("outputs/experiments/blockwise_no_ccd/lora_weights.safetensors")


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


def train_cat_lora_noccd() -> Path:
    """Train a blockwise (no CCD) LoRA for cat."""
    cat_lora_path = OUTPUT_BASE / "cat_blockwise_noccd" / "lora_weights.safetensors"
    if cat_lora_path.exists():
        logger.info("Cat no-CCD LoRA already exists, skipping.")
        return cat_lora_path

    logger.info("Training cat LoRA (blockwise, no CCD)...")
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

    dataset = DreamBoothDataset(
        subject_images_dir=str(CAT_SUBJECT_DIR),
        class_images_dir=str(CAT_CLASS_DIR),
        token="[V]",
        class_noun="cat",
        resolution=512,
    )

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
        identity_rank=16, context_rank=4, shared_rank=8,
        alpha_ratio=1.0, dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()
    cast_lora_to_fp32(lora)

    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "cat", "num_images": 4},
        "lora": {"identity_rank": 16, "context_rank": 4, "shared_rank": 8},
        "training": {
            "num_steps": 500, "batch_size": 1, "gradient_accumulation": 1,
            "learning_rate": 5e-4, "weight_decay": 0.01, "warmup_steps": 20,
            "max_grad_norm": 1.0, "mixed_precision": "no", "seed": 42,
            "log_every": 50, "save_every": 500, "validate_every": 500,
            "scheduler": "cosine",
        },
        "prior_preservation": {"enabled": True, "num_class_images": 8, "lambda_ppl": 1.0},
        "ccd": {"enabled": False, "lambda_ccd": 0.0, "temperature": 0.07,
                "warmup_steps": 100, "feature_layer": "middle"},
        "inference": {"num_steps": 30, "guidance_scale": 7.5, "resolution": 512},
    })

    loss_fn = ModularBoothLoss(lambda_ppl=1.0, lambda_ccd=0.0, ccd_warmup_steps=0)

    trainer = ModularBoothTrainer(
        config=cfg, model=pipe, lora=lora, dataset=dataset, loss_fn=loss_fn, device=DEVICE,
    )

    t0 = time.time()
    results = trainer.train()
    logger.info("Cat no-CCD training: %.1fs, loss=%.4f", time.time() - t0, results.get("loss_total", 0))

    out_dir = OUTPUT_BASE / "cat_blockwise_noccd"
    out_dir.mkdir(parents=True, exist_ok=True)
    lora.save_lora(str(cat_lora_path))
    logger.info("Cat no-CCD LoRA saved: %s", cat_lora_path)

    del trainer, pipe, lora, loss_fn, dataset
    gpu_cleanup()
    return cat_lora_path


def _apply_lora_to_unet(unet, lora_sd):
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
        delta = params["lora_B"] @ params["lora_A"]
        target.weight.data.add_(delta.to(target.weight.dtype))


def _remove_lora_from_unet(unet, lora_sd):
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
        delta = params["lora_B"] @ params["lora_A"]
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


def _masked_denoising(pipe, prompt, dog_sd, cat_sd, seed,
                       neg_attn_strength=3.0, mask_leakage_alpha=0.05):
    latent_h, latent_w = 64, 64
    dog_mask = torch.zeros(1, 1, latent_h, latent_w, device=DEVICE, dtype=torch.float16)
    cat_mask = torch.zeros(1, 1, latent_h, latent_w, device=DEVICE, dtype=torch.float16)
    dog_mask[:, :, :, : latent_w // 2] = 1.0
    cat_mask[:, :, :, latent_w // 2 :] = 1.0

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_inputs = tokenizer(prompt, padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE)).last_hidden_state
    uncond_inputs = tokenizer("", padding="max_length",
                              max_length=tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
    with torch.no_grad():
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(DEVICE)).last_hidden_state

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = torch.randn((1, pipe.unet.config.in_channels, latent_h, latent_w),
                          generator=gen, device=DEVICE, dtype=torch.float16)
    pipe.scheduler.set_timesteps(30, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    latents = latents * pipe.scheduler.init_noise_sigma

    for t in timesteps:
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)
        with torch.no_grad():
            base_noise = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            uncond_noise = pipe.unet(latent_model_input, t, encoder_hidden_states=uncond_embeds).sample

        _apply_lora_to_unet(pipe.unet, dog_sd)
        with torch.no_grad():
            dog_noise = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
        _remove_lora_from_unet(pipe.unet, dog_sd)

        _apply_lora_to_unet(pipe.unet, cat_sd)
        with torch.no_grad():
            cat_noise = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
        _remove_lora_from_unet(pipe.unet, cat_sd)

        dog_delta = dog_noise - base_noise
        cat_delta = cat_noise - base_noise
        dog_soft = dog_mask * (1.0 - mask_leakage_alpha) + mask_leakage_alpha
        cat_soft = cat_mask * (1.0 - mask_leakage_alpha) + mask_leakage_alpha
        blended_cond = base_noise + dog_delta * dog_soft + cat_delta * cat_soft

        combined_mask = torch.clamp(dog_mask + cat_mask, 0.0, 1.0)
        bg_mask = 1.0 - combined_mask
        if bg_mask.sum() > 0:
            avg_delta = (dog_delta + cat_delta) / 2.0
            blended_cond = blended_cond - neg_attn_strength * avg_delta * bg_mask

        noise_pred = uncond_noise + 7.5 * (blended_cond - uncond_noise)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    latents_scaled = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image_tensor = pipe.vae.decode(latents_scaled).sample
    image_tensor = (image_tensor / 2.0 + 0.5).clamp(0.0, 1.0)
    image_tensor = image_tensor.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
    image_array = (image_tensor * 255.0).round().astype("uint8")
    return Image.fromarray(image_array)


def run_multi_subject(dog_lora_path, cat_lora_path, output_dir, mode="masked"):
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file as load_safetensors
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,
    ).to(DEVICE)
    dog_sd = {k: v.to(DEVICE) for k, v in load_safetensors(str(dog_lora_path)).items()}
    cat_sd = {k: v.to(DEVICE) for k, v in load_safetensors(str(cat_lora_path)).items()}

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
            if mode == "masked":
                img = _masked_denoising(pipe, prompt, dog_sd, cat_sd, seed)
                class _R:
                    def __init__(self, imgs): self.images = imgs
                result = _R([img])
            else:
                _apply_lora_to_unet(pipe.unet, dog_sd)
                _apply_lora_to_unet(pipe.unet, cat_sd)
                with torch.no_grad():
                    result = pipe(prompt=prompt, num_inference_steps=30,
                                  guidance_scale=7.5, height=512, width=512, generator=gen)
                _remove_lora_from_unet(pipe.unet, cat_sd)
                _remove_lora_from_unet(pipe.unet, dog_sd)

            fname = f"{mode}_{p_idx:02d}_{seed_idx:02d}.png"
            result.images[0].save(output_dir / fname)
            results.append((str(output_dir / fname), prompt))

    del pipe, dog_sd, cat_sd
    gpu_cleanup()
    return results


def evaluate(gen_results, dog_dir, cat_dir, label):
    from modularbooth.evaluation.clip_score import CLIPScore
    from modularbooth.evaluation.dino_score import DINOScore

    metrics = {}
    gen_images = [Image.open(p).convert("RGB") for p, _ in gen_results]
    gen_prompts = [prompt for _, prompt in gen_results]
    dog_refs = [Image.open(p).convert("RGB") for p in sorted(dog_dir.glob("*.png"))]
    cat_refs = [Image.open(p).convert("RGB") for p in sorted(cat_dir.glob("*.png"))]

    try:
        clip = CLIPScore(device=DEVICE)
        metrics["clip_t"] = clip.clip_t_score(gen_images, gen_prompts)
        del clip; gpu_cleanup()
    except Exception as e:
        logger.warning("CLIP failed: %s", e)

    try:
        dino = DINOScore(device=DEVICE)
        metrics["dino_vs_dog"] = dino.compute_score(gen_images, dog_refs)
        metrics["dino_vs_cat"] = dino.compute_score(gen_images, cat_refs)

        left_crops = [img.crop((0, 0, img.size[0] // 2, img.size[1])) for img in gen_images]
        right_crops = [img.crop((img.size[0] // 2, 0, img.size[0], img.size[1])) for img in gen_images]

        dog_left = dino.compute_score(left_crops, dog_refs)
        cat_right = dino.compute_score(right_crops, cat_refs)
        dog_right = dino.compute_score(right_crops, dog_refs)
        cat_left = dino.compute_score(left_crops, cat_refs)

        metrics["iis"] = ((dog_left - dog_right) + (cat_right - cat_left)) / 2.0
        metrics["dog_correct_sim"] = dog_left
        metrics["cat_correct_sim"] = cat_right
        metrics["dog_cross_sim"] = dog_right
        metrics["cat_cross_sim"] = cat_left

        comp_scores = [1.0 - dino.compute_score([lc], [rc]) for lc, rc in zip(left_crops, right_crops)]
        metrics["composition_diversity"] = sum(comp_scores) / len(comp_scores)

        del dino; gpu_cleanup()
    except Exception as e:
        logger.warning("DINO failed: %s", e)

    logger.info("[%s] IIS=%.4f, CLIP-T=%.4f, CompDiv=%.4f", label,
                metrics.get("iis", 0), metrics.get("clip_t", 0), metrics.get("composition_diversity", 0))
    return metrics


def main():
    start = time.time()

    # Train cat (no CCD)
    cat_lora_noccd = train_cat_lora_noccd()
    gpu_cleanup()

    # Run masked composition with no-CCD LoRAs
    out_dir = OUTPUT_BASE / "multi_subject_noccd"
    out_dir.mkdir(parents=True, exist_ok=True)

    masked_dir = out_dir / "masked"
    masked_results = run_multi_subject(DOG_LORA_NOCCD, cat_lora_noccd, masked_dir, "masked")
    gpu_cleanup()

    naive_dir = out_dir / "naive"
    naive_results = run_multi_subject(DOG_LORA_NOCCD, cat_lora_noccd, naive_dir, "naive")
    gpu_cleanup()

    # Evaluate
    masked_metrics = evaluate(masked_results, DOG_SUBJECT_DIR, CAT_SUBJECT_DIR, "noccd_masked")
    gpu_cleanup()
    naive_metrics = evaluate(naive_results, DOG_SUBJECT_DIR, CAT_SUBJECT_DIR, "noccd_naive")
    gpu_cleanup()

    all_results = {
        "masked": masked_metrics,
        "naive": naive_metrics,
        "total_time_s": time.time() - start,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("No-CCD masked: %s", json.dumps(masked_metrics, indent=2))
    logger.info("No-CCD naive: %s", json.dumps(naive_metrics, indent=2))
    logger.info("Total: %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
