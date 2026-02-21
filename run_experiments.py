"""Comprehensive experiment suite for ModularBooth paper.

Memory-efficient design: generation and evaluation are separated to avoid
loading the pipeline and evaluation models simultaneously.

Experiments:
1. baseline_r8 - Uniform rank 8, no CCD
2. blockwise_no_ccd - Blockwise (identity=16, context=4, shared=8), no CCD
3. blockwise_ccd - Blockwise + CCD loss
4. uniform_r4 - Uniform rank 4 (low capacity)
5. uniform_r16 - Uniform rank 16 (high capacity)
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
from omegaconf import OmegaConf
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("experiments")

DEVICE = "cuda"
OUTPUT_BASE = Path("outputs/experiments")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

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
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cast_lora_to_fp32(lora):
    """Convert LoRA parameters to fp32 for stable training."""
    for lora_mod in lora._lora_modules.values():
        lora_mod.lora_A.data = lora_mod.lora_A.data.float()
        lora_mod.lora_B.data = lora_mod.lora_B.data.float()
        lora_mod.lora_A.requires_grad_(True)
        lora_mod.lora_B.requires_grad_(True)


def verify_lora_learning(lora, tag=""):
    """Check that LoRA B matrices moved from zero."""
    total_b, total_a, n = 0.0, 0.0, 0
    for lora_mod in lora._lora_modules.values():
        total_b += lora_mod.lora_B.data.norm().item()
        total_a += lora_mod.lora_A.data.norm().item()
        n += 1
    if n > 0:
        avg_b, avg_a = total_b / n, total_a / n
        logger.info("[%s] avg |B|=%.6f, avg |A|=%.6f (%d modules)", tag, avg_b, avg_a, n)
        return avg_b, avg_a
    return 0.0, 0.0


def create_augmented_images(subject_dir: Path, output_dir: Path, n_variants: int = 3):
    """Create color-jittered augmented images for CCD loss."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    for img_path in sorted(subject_dir.glob("*.png")):
        variant_dir = output_dir / img_path.stem
        variant_dir.mkdir(exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        for v in range(n_variants):
            torch.manual_seed(v * 1000 + hash(img_path.stem) % 10000)
            augmented = jitter(img)
            augmented.save(variant_dir / f"variant_{v:02d}.png")


def phase_train(config: dict) -> dict:
    """Phase 1: Train and generate images. Returns training results + paths."""
    name = config["name"]
    output_dir = OUTPUT_BASE / name
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_cleanup()
    logger.info("GPU mem before load: %.0f MB used", torch.cuda.memory_allocated() / 1e6)

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
    logger.info("GPU mem after pipeline load: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    # Dataset
    dataset_kwargs = dict(
        subject_images_dir=str(SUBJECT_DIR),
        class_images_dir=str(CLASS_DIR),
        token="[V]",
        class_noun="dog",
        resolution=512,
    )
    if config.get("use_ccd", False):
        aug_dir = OUTPUT_BASE / "augmented_images"
        if not aug_dir.exists():
            create_augmented_images(SUBJECT_DIR, aug_dir, n_variants=3)
        dataset_kwargs["augmented_images_dir"] = str(aug_dir)

    dataset = DreamBoothDataset(**dataset_kwargs)

    # Apply LoRA
    unet = pipe.unet
    lora = BlockwiseLoRA(
        model=unet,
        block_config=config["block_config"],
        identity_rank=config["identity_rank"],
        context_rank=config["context_rank"],
        shared_rank=config["shared_rank"],
        alpha_ratio=1.0,
        dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()
    cast_lora_to_fp32(lora)

    param_count = lora.get_parameter_count()
    total_params = sum(param_count.values())
    logger.info("LoRA params: %d (%s)", total_params, param_count)

    # Training config
    num_steps = config.get("num_steps", 500)
    lr = config.get("learning_rate", 5e-4)
    use_ccd = config.get("use_ccd", False)
    lambda_ccd = config.get("lambda_ccd", 0.3) if use_ccd else 0.0

    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "dog", "num_images": 4},
        "lora": {
            "identity_rank": config["identity_rank"],
            "context_rank": config["context_rank"],
            "shared_rank": config["shared_rank"],
        },
        "training": {
            "num_steps": num_steps,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": lr,
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
        "prior_preservation": {"enabled": True, "num_class_images": 8, "lambda_ppl": 1.0},
        "ccd": {
            "enabled": use_ccd,
            "lambda_ccd": lambda_ccd,
            "temperature": 0.07,
            "warmup_steps": 100,
            "feature_layer": "middle",
        },
        "inference": {"num_steps": 30, "guidance_scale": 7.5, "resolution": 512},
    })

    loss_fn = ModularBoothLoss(
        lambda_ppl=1.0,
        lambda_ccd=lambda_ccd,
        ccd_warmup_steps=100 if use_ccd else 0,
    )

    trainer = ModularBoothTrainer(
        config=cfg,
        model=pipe,
        lora=lora,
        dataset=dataset,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    # Train
    logger.info("Training %d steps...", num_steps)
    t0 = time.time()
    results = trainer.train()
    train_time = time.time() - t0
    logger.info("Training: %.1fs, loss=%.4f", train_time, results.get("loss_total", 0))

    b_norm, a_norm = verify_lora_learning(lora, name)

    # Save LoRA
    lora_path = output_dir / "lora_weights.safetensors"
    lora.save_lora(str(lora_path))
    lora_size_mb = lora_path.stat().st_size / (1024 * 1024)

    # Generate eval images
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    seed_offset = abs(hash(name)) % 10000

    gen_images_paths = []
    gen_prompts = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        for j in range(2):  # 2 images per prompt
            seed = 1000 + i * 100 + j + seed_offset
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            with torch.no_grad():
                result = pipe(prompt=prompt, num_inference_steps=30,
                            guidance_scale=7.5, height=512, width=512, generator=gen)
            img_path = gen_dir / f"gen_{i:02d}_{j:02d}.png"
            result.images[0].save(img_path)
            gen_images_paths.append(str(img_path))
            gen_prompts.append(prompt)

    # Generate CAE images (same subject in different contexts)
    cae_dir = output_dir / "cae_images"
    cae_dir.mkdir(parents=True, exist_ok=True)
    cae_paths = []
    for i, prompt in enumerate(CAE_PROMPTS):
        gen = torch.Generator(device=DEVICE).manual_seed(42 + i)
        with torch.no_grad():
            result = pipe(prompt=prompt, num_inference_steps=30,
                        guidance_scale=7.5, height=512, width=512, generator=gen)
        img_path = cae_dir / f"cae_{i:02d}.png"
        result.images[0].save(img_path)
        cae_paths.append(str(img_path))

    logger.info("Generated %d eval + %d CAE images", len(gen_images_paths), len(cae_paths))

    # Delete pipeline to free GPU memory
    del trainer, pipe, lora, loss_fn, dataset, unet
    gpu_cleanup()
    logger.info("GPU mem after cleanup: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    return {
        "name": name,
        "output_dir": str(output_dir),
        "gen_images_paths": gen_images_paths,
        "gen_prompts": gen_prompts,
        "cae_paths": cae_paths,
        "train_time": train_time,
        "train_results": results,
        "lora_params": total_params,
        "lora_param_breakdown": param_count,
        "lora_size_mb": lora_size_mb,
        "lora_b_norm": b_norm,
        "lora_a_norm": a_norm,
        "config": {k: v for k, v in config.items() if k != "block_config"},
        "block_config": {str(k): v for k, v in config["block_config"].items()},
    }


def phase_evaluate(train_result: dict) -> dict:
    """Phase 2: Evaluate generated images (pipeline NOT loaded)."""
    name = train_result["name"]
    logger.info("Evaluating %s...", name)

    # Load images from disk
    gen_images = [Image.open(p).convert("RGB") for p in train_result["gen_images_paths"]]
    subject_images = [Image.open(p).convert("RGB") for p in sorted(SUBJECT_DIR.glob("*.png"))]
    prompts = train_result["gen_prompts"]

    metrics = {}

    # DINO
    try:
        from modularbooth.evaluation.dino_score import DINOScore
        dino = DINOScore(device=DEVICE)
        metrics["dino_score"] = dino.compute_score(gen_images, subject_images)
        logger.info("[%s] DINO: %.4f", name, metrics["dino_score"])

        # CAE (uses same DINO model)
        try:
            from modularbooth.evaluation.entanglement import ContextAppearanceEntanglement
            cae_imgs = [Image.open(p).convert("RGB") for p in train_result["cae_paths"]]
            cae_scorer = ContextAppearanceEntanglement(dino_scorer=dino)
            metrics["cae"] = cae_scorer.compute_cae(cae_imgs)
            logger.info("[%s] CAE: %.6f", name, metrics["cae"])
            del cae_scorer, cae_imgs
        except Exception as e:
            logger.warning("[%s] CAE failed: %s", name, e)

        del dino
        gpu_cleanup()
    except Exception as e:
        logger.warning("[%s] DINO failed: %s", name, e)

    # CLIP
    try:
        from modularbooth.evaluation.clip_score import CLIPScore
        clip = CLIPScore(device=DEVICE)
        metrics["clip_t_score"] = clip.clip_t_score(gen_images, prompts)
        metrics["clip_i_score"] = clip.clip_i_score(gen_images, subject_images)
        logger.info("[%s] CLIP-T: %.4f, CLIP-I: %.4f", name, metrics["clip_t_score"], metrics["clip_i_score"])
        del clip
        gpu_cleanup()
    except Exception as e:
        logger.warning("[%s] CLIP failed: %s", name, e)

    # VQA
    try:
        from modularbooth.evaluation.vqa_alignment import VQAAlignment
        vqa = VQAAlignment(device=DEVICE)
        metrics["vqa_alignment"] = vqa.compute_batch_alignment(gen_images, prompts)
        logger.info("[%s] VQA: %.4f", name, metrics["vqa_alignment"])
        del vqa
        gpu_cleanup()
    except Exception as e:
        logger.warning("[%s] VQA failed: %s", name, e)

    # LPIPS
    try:
        from modularbooth.evaluation.diversity import LPIPSDiversity
        lpips_scorer = LPIPSDiversity(device=DEVICE)
        metrics["lpips_diversity"] = lpips_scorer.compute_diversity(gen_images)
        logger.info("[%s] LPIPS: %.4f", name, metrics["lpips_diversity"])
        del lpips_scorer
        gpu_cleanup()
    except Exception as e:
        logger.warning("[%s] LPIPS failed: %s", name, e)

    return metrics


def main():
    start_time = time.time()

    # Discover block structure
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
    logger.info("Block indices: %s", sorted_blocks)
    del pipe
    gpu_cleanup()

    n_blocks = len(sorted_blocks)
    uniform_config = {idx: "shared" for idx in sorted_blocks}

    blockwise_config = {}
    for i, idx in enumerate(sorted_blocks):
        if i < n_blocks // 3:
            blockwise_config[idx] = "identity"
        elif i >= 2 * n_blocks // 3:
            blockwise_config[idx] = "context"
        else:
            blockwise_config[idx] = "shared"

    if not SUBJECT_DIR.exists():
        logger.error("Subject images not found. Run run_real_training.py first.")
        return

    experiments = [
        {
            "name": "uniform_r4",
            "block_config": uniform_config,
            "identity_rank": 4, "context_rank": 4, "shared_rank": 4,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        {
            "name": "baseline_r8",
            "block_config": uniform_config,
            "identity_rank": 8, "context_rank": 8, "shared_rank": 8,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        {
            "name": "uniform_r16",
            "block_config": uniform_config,
            "identity_rank": 16, "context_rank": 16, "shared_rank": 16,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        {
            "name": "blockwise_no_ccd",
            "block_config": blockwise_config,
            "identity_rank": 16, "context_rank": 4, "shared_rank": 8,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        {
            "name": "blockwise_ccd",
            "block_config": blockwise_config,
            "identity_rank": 16, "context_rank": 4, "shared_rank": 8,
            "num_steps": 500, "learning_rate": 5e-4,
            "use_ccd": True, "lambda_ccd": 0.3,
        },
    ]

    all_results = []
    for exp_cfg in experiments:
        try:
            # Phase 1: Train and generate
            train_result = phase_train(exp_cfg)

            # Phase 2: Evaluate (pipeline freed)
            metrics = phase_evaluate(train_result)

            # Combine
            final = {
                "name": train_result["name"],
                "block_config": train_result["block_config"],
                "config": train_result["config"],
                "training_time_s": train_result["train_time"],
                "final_loss_total": train_result["train_results"].get("loss_total", 0),
                "final_loss_diffusion": train_result["train_results"].get("loss_diffusion", 0),
                "final_loss_ppl": train_result["train_results"].get("loss_ppl", 0),
                "final_loss_ccd": train_result["train_results"].get("loss_ccd", 0),
                "lora_params": train_result["lora_params"],
                "lora_param_breakdown": train_result["lora_param_breakdown"],
                "lora_size_mb": train_result["lora_size_mb"],
                "lora_b_norm": train_result["lora_b_norm"],
                "metrics": metrics,
            }

            # Save per-experiment
            output_dir = Path(train_result["output_dir"])
            with open(output_dir / "results.json", "w") as f:
                json.dump(final, f, indent=2, default=str)

            all_results.append(final)
            logger.info("[%s] DONE: %s", final["name"],
                       {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})

        except Exception as e:
            logger.error("EXPERIMENT %s FAILED: %s", exp_cfg["name"], e, exc_info=True)
            all_results.append({"name": exp_cfg["name"], "error": str(e)})
            gpu_cleanup()

    # Summary
    elapsed = time.time() - start_time
    with open(OUTPUT_BASE / "experiment_summary.json", "w") as f:
        json.dump({"total_time_s": elapsed, "experiments": all_results}, f, indent=2, default=str)

    logger.info("\n" + "=" * 100)
    logger.info("ALL EXPERIMENTS COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 100)
    logger.info("%-20s | %7s | %7s | %7s | %7s | %7s | %10s | %8s",
                "Name", "DINO", "CLIP-T", "CLIP-I", "VQA", "LPIPS", "CAE", "Params")
    logger.info("-" * 100)
    for r in all_results:
        if "error" in r:
            logger.info("%-20s | FAILED", r["name"])
            continue
        m = r.get("metrics", {})
        logger.info("%-20s | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %10.6f | %8d",
                    r["name"],
                    m.get("dino_score", 0), m.get("clip_t_score", 0), m.get("clip_i_score", 0),
                    m.get("vqa_alignment", 0), m.get("lpips_diversity", 0), m.get("cae", 0),
                    r.get("lora_params", 0))


if __name__ == "__main__":
    main()
