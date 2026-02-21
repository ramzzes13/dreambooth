"""Re-run just the blockwise_ccd experiment after fixing the CCD loss bug."""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("ccd_experiment")

DEVICE = "cuda"
OUTPUT_BASE = Path("outputs/experiments")
SUBJECT_DIR = Path("outputs/sd14_training/subject_images")
CLASS_DIR = Path("outputs/sd14_training/class_images")

EVAL_PROMPTS = [
    "a photo of sks dog on a beach at sunset",
    "a photo of sks dog in a snowy mountain landscape",
    "a photo of sks dog in a colorful garden with flowers",
    "a photo of sks dog wearing a red hat",
    "a painting of sks dog in watercolor style",
    "a photo of sks dog sleeping on a couch",
]

CAE_PROMPTS = [
    "a photo of sks dog on a beach",
    "a photo of sks dog in a forest",
    "a photo of sks dog in a city street",
    "a photo of sks dog in a kitchen",
    "a photo of sks dog on a snowy field",
    "a photo of sks dog in a desert",
]


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


def main():
    start = time.time()
    gpu_cleanup()

    # Discover blocks
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,
    )
    block_indices = set()
    for name, _ in pipe.unet.named_modules():
        match = re.search(r"(?:blocks|transformer_blocks)\.(\d+)\.", name)
        if match:
            block_indices.add(int(match.group(1)))
    sorted_blocks = sorted(block_indices)
    n_blocks = len(sorted_blocks)
    del pipe
    gpu_cleanup()

    blockwise_config = {}
    for i, idx in enumerate(sorted_blocks):
        if i < n_blocks // 3:
            blockwise_config[idx] = "identity"
        elif i >= 2 * n_blocks // 3:
            blockwise_config[idx] = "context"
        else:
            blockwise_config[idx] = "shared"

    # Ensure augmented images exist
    aug_dir = OUTPUT_BASE / "augmented_images"
    if not aug_dir.exists():
        jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        aug_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(SUBJECT_DIR.glob("*.png")):
            variant_dir = aug_dir / img_path.stem
            variant_dir.mkdir(exist_ok=True)
            img = Image.open(img_path).convert("RGB")
            for v in range(3):
                torch.manual_seed(v * 1000 + hash(img_path.stem) % 10000)
                augmented = jitter(img)
                augmented.save(variant_dir / f"variant_{v:02d}.png")

    # ===== TRAIN =====
    from modularbooth.data.dataset import DreamBoothDataset
    from modularbooth.losses.combined import ModularBoothLoss
    from modularbooth.models.blockwise_lora import BlockwiseLoRA
    from modularbooth.training.trainer import ModularBoothTrainer

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,
    ).to(DEVICE)
    logger.info("GPU mem: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    dataset = DreamBoothDataset(
        subject_images_dir=str(SUBJECT_DIR),
        class_images_dir=str(CLASS_DIR),
        token="[V]", class_noun="dog", resolution=512,
        augmented_images_dir=str(aug_dir),
    )

    lora = BlockwiseLoRA(
        model=pipe.unet,
        block_config=blockwise_config,
        identity_rank=16, context_rank=4, shared_rank=8,
        alpha_ratio=1.0, dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()
    cast_lora_to_fp32(lora)

    param_count = lora.get_parameter_count()
    total_params = sum(param_count.values())
    logger.info("LoRA params: %d", total_params)

    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "dog", "num_images": 4},
        "lora": {"identity_rank": 16, "context_rank": 4, "shared_rank": 8},
        "training": {
            "num_steps": 500, "batch_size": 1, "gradient_accumulation": 1,
            "learning_rate": 5e-4, "weight_decay": 0.01, "warmup_steps": 20,
            "max_grad_norm": 1.0, "mixed_precision": "no", "seed": 42,
            "log_every": 50, "save_every": 500, "validate_every": 500,
            "scheduler": "cosine",
        },
        "prior_preservation": {"enabled": True, "num_class_images": 8, "lambda_ppl": 1.0},
        "ccd": {
            "enabled": True, "lambda_ccd": 0.3, "temperature": 0.07,
            "warmup_steps": 100, "feature_layer": "middle",
        },
        "inference": {"num_steps": 30, "guidance_scale": 7.5, "resolution": 512},
    })

    loss_fn = ModularBoothLoss(lambda_ppl=1.0, lambda_ccd=0.3, ccd_warmup_steps=100)

    trainer = ModularBoothTrainer(
        config=cfg, model=pipe, lora=lora, dataset=dataset,
        loss_fn=loss_fn, device=DEVICE,
    )

    logger.info("Training 500 steps with CCD loss...")
    t0 = time.time()
    results = trainer.train()
    train_time = time.time() - t0
    logger.info("Training: %.1fs, loss=%.4f", train_time, results.get("loss_total", 0))

    # Check LoRA B norms
    total_b, n = 0.0, 0
    for lora_mod in lora._lora_modules.values():
        total_b += lora_mod.lora_B.data.norm().item()
        n += 1
    logger.info("avg |B|=%.6f (%d modules)", total_b / n if n else 0, n)

    # Save LoRA
    output_dir = OUTPUT_BASE / "blockwise_ccd"
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = output_dir / "lora_weights.safetensors"
    lora.save_lora(str(lora_path))
    lora_size = lora_path.stat().st_size / (1024 * 1024)

    # Generate images
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    seed_offset = abs(hash("blockwise_ccd")) % 10000
    gen_paths, gen_prompts = [], []

    for i, prompt in enumerate(EVAL_PROMPTS):
        for j in range(2):
            seed = 1000 + i * 100 + j + seed_offset
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            with torch.no_grad():
                result = pipe(prompt=prompt, num_inference_steps=30,
                            guidance_scale=7.5, height=512, width=512, generator=gen)
            p = gen_dir / f"gen_{i:02d}_{j:02d}.png"
            result.images[0].save(p)
            gen_paths.append(str(p))
            gen_prompts.append(prompt)

    # CAE images
    cae_dir = output_dir / "cae_images"
    cae_dir.mkdir(parents=True, exist_ok=True)
    cae_paths = []
    for i, prompt in enumerate(CAE_PROMPTS):
        gen = torch.Generator(device=DEVICE).manual_seed(42 + i)
        with torch.no_grad():
            result = pipe(prompt=prompt, num_inference_steps=30,
                        guidance_scale=7.5, height=512, width=512, generator=gen)
        p = cae_dir / f"cae_{i:02d}.png"
        result.images[0].save(p)
        cae_paths.append(str(p))

    # Free pipeline
    del trainer, pipe, lora, loss_fn, dataset
    gpu_cleanup()

    # ===== EVALUATE =====
    gen_images = [Image.open(p).convert("RGB") for p in gen_paths]
    subject_images = [Image.open(p).convert("RGB") for p in sorted(SUBJECT_DIR.glob("*.png"))]
    metrics = {}

    # DINO + CAE
    try:
        from modularbooth.evaluation.dino_score import DINOScore
        from modularbooth.evaluation.entanglement import ContextAppearanceEntanglement
        dino = DINOScore(device=DEVICE)
        metrics["dino_score"] = dino.compute_score(gen_images, subject_images)
        cae_imgs = [Image.open(p).convert("RGB") for p in cae_paths]
        cae_scorer = ContextAppearanceEntanglement(dino_scorer=dino)
        metrics["cae"] = cae_scorer.compute_cae(cae_imgs)
        logger.info("DINO: %.4f, CAE: %.6f", metrics["dino_score"], metrics["cae"])
        del dino, cae_scorer
        gpu_cleanup()
    except Exception as e:
        logger.warning("DINO/CAE failed: %s", e)

    # CLIP
    try:
        from modularbooth.evaluation.clip_score import CLIPScore
        clip = CLIPScore(device=DEVICE)
        metrics["clip_t_score"] = clip.clip_t_score(gen_images, gen_prompts)
        metrics["clip_i_score"] = clip.clip_i_score(gen_images, subject_images)
        logger.info("CLIP-T: %.4f, CLIP-I: %.4f", metrics["clip_t_score"], metrics["clip_i_score"])
        del clip
        gpu_cleanup()
    except Exception as e:
        logger.warning("CLIP failed: %s", e)

    # VQA
    try:
        from modularbooth.evaluation.vqa_alignment import VQAAlignment
        vqa = VQAAlignment(device=DEVICE)
        metrics["vqa_alignment"] = vqa.compute_batch_alignment(gen_images, gen_prompts)
        logger.info("VQA: %.4f", metrics["vqa_alignment"])
        del vqa
        gpu_cleanup()
    except Exception as e:
        logger.warning("VQA failed: %s", e)

    # LPIPS
    try:
        from modularbooth.evaluation.diversity import LPIPSDiversity
        lpips_scorer = LPIPSDiversity(device=DEVICE)
        metrics["lpips_diversity"] = lpips_scorer.compute_diversity(gen_images)
        logger.info("LPIPS: %.4f", metrics["lpips_diversity"])
        del lpips_scorer
        gpu_cleanup()
    except Exception as e:
        logger.warning("LPIPS failed: %s", e)

    # Save results
    final = {
        "name": "blockwise_ccd",
        "block_config": {str(k): v for k, v in blockwise_config.items()},
        "config": {"identity_rank": 16, "context_rank": 4, "shared_rank": 8,
                   "use_ccd": True, "lambda_ccd": 0.3, "num_steps": 500, "learning_rate": 5e-4},
        "training_time_s": train_time,
        "final_loss_total": results.get("loss_total", 0),
        "final_loss_diffusion": results.get("loss_diffusion", 0),
        "final_loss_ppl": results.get("loss_ppl", 0),
        "final_loss_ccd": results.get("loss_ccd", 0),
        "lora_params": total_params,
        "lora_param_breakdown": param_count,
        "lora_size_mb": lora_size,
        "metrics": metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info("\nCCD experiment done in %.1fs", elapsed)
    logger.info("Metrics: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})


if __name__ == "__main__":
    main()
