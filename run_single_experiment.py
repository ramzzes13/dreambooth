"""Run a single experiment as a standalone process (for GPU memory isolation)."""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("experiment")

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


def create_augmented_images(subject_dir: Path, output_dir: Path, n_variants: int = 3):
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


def run_experiment(config: dict):
    """Run a complete experiment: train, generate, evaluate, save."""
    name = config["name"]
    output_dir = OUTPUT_BASE / name
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    gpu_cleanup()
    logger.info("GPU mem before load: %.0f MB used", torch.cuda.memory_allocated() / 1e6)

    from diffusers import StableDiffusionPipeline
    from modularbooth.data.dataset import DreamBoothDataset
    from modularbooth.losses.combined import ModularBoothLoss
    from modularbooth.models.blockwise_lora import BlockwiseLoRA
    from modularbooth.training.trainer import ModularBoothTrainer

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,
    ).to(DEVICE)
    logger.info("GPU mem after load: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    # Dataset
    dataset_kwargs = dict(
        subject_images_dir=str(SUBJECT_DIR), class_images_dir=str(CLASS_DIR),
        token="[V]", class_noun="dog", resolution=512,
    )
    use_ccd = config.get("use_ccd", False)
    if use_ccd:
        aug_dir = OUTPUT_BASE / "augmented_images"
        if not aug_dir.exists():
            create_augmented_images(SUBJECT_DIR, aug_dir, n_variants=3)
        dataset_kwargs["augmented_images_dir"] = str(aug_dir)

    dataset = DreamBoothDataset(**dataset_kwargs)

    # LoRA
    lora = BlockwiseLoRA(
        model=pipe.unet, block_config=config["block_config"],
        identity_rank=config["identity_rank"], context_rank=config["context_rank"],
        shared_rank=config["shared_rank"], alpha_ratio=1.0, dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()
    cast_lora_to_fp32(lora)

    param_count = lora.get_parameter_count()
    total_params = sum(param_count.values())
    logger.info("LoRA params: %d (%s)", total_params, param_count)

    # Config
    num_steps = config.get("num_steps", 500)
    lr = config.get("learning_rate", 5e-4)
    lambda_ccd = config.get("lambda_ccd", 0.3) if use_ccd else 0.0

    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "dog", "num_images": 4},
        "lora": {"identity_rank": config["identity_rank"], "context_rank": config["context_rank"],
                 "shared_rank": config["shared_rank"]},
        "training": {
            "num_steps": num_steps, "batch_size": 1, "gradient_accumulation": 1,
            "learning_rate": lr, "weight_decay": 0.01, "warmup_steps": 20,
            "max_grad_norm": 1.0, "mixed_precision": "no", "seed": 42,
            "log_every": 50, "save_every": 500, "validate_every": 500, "scheduler": "cosine",
        },
        "prior_preservation": {"enabled": True, "num_class_images": 8, "lambda_ppl": 1.0},
        "ccd": {"enabled": use_ccd, "lambda_ccd": lambda_ccd, "temperature": 0.07,
                "warmup_steps": 100, "feature_layer": "middle"},
        "inference": {"num_steps": 30, "guidance_scale": 7.5, "resolution": 512},
    })

    loss_fn = ModularBoothLoss(lambda_ppl=1.0, lambda_ccd=lambda_ccd,
                                ccd_warmup_steps=100 if use_ccd else 0)

    trainer = ModularBoothTrainer(
        config=cfg, model=pipe, lora=lora, dataset=dataset, loss_fn=loss_fn, device=DEVICE,
    )

    # Train
    logger.info("Training %d steps...", num_steps)
    t0 = time.time()
    results = trainer.train()
    train_time = time.time() - t0
    logger.info("Training: %.1fs, loss=%.4f", train_time, results.get("loss_total", 0))

    # Check LoRA norms
    total_b, total_a, n = 0.0, 0.0, 0
    for lora_mod in lora._lora_modules.values():
        total_b += lora_mod.lora_B.data.norm().item()
        total_a += lora_mod.lora_A.data.norm().item()
        n += 1
    avg_b = total_b / n if n else 0
    avg_a = total_a / n if n else 0
    logger.info("avg |B|=%.6f, avg |A|=%.6f (%d modules)", avg_b, avg_a, n)

    # Save LoRA
    lora_path = output_dir / "lora_weights.safetensors"
    lora.save_lora(str(lora_path))
    lora_size_mb = lora_path.stat().st_size / (1024 * 1024)

    # Generate eval images
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    seed_offset = abs(hash(name)) % 10000
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

    # Generate CAE images
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

    logger.info("Generated %d eval + %d CAE images", len(gen_paths), len(cae_paths))

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
        "name": name,
        "block_config": {str(k): v for k, v in config["block_config"].items()},
        "config": {k: v for k, v in config.items() if k != "block_config"},
        "training_time_s": train_time,
        "final_loss_total": results.get("loss_total", 0),
        "final_loss_diffusion": results.get("loss_diffusion", 0),
        "final_loss_ppl": results.get("loss_ppl", 0),
        "final_loss_ccd": results.get("loss_ccd", 0),
        "lora_params": total_params,
        "lora_param_breakdown": param_count,
        "lora_size_mb": lora_size_mb,
        "lora_b_norm": avg_b,
        "metrics": metrics,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info("[%s] Complete in %.1fs. Metrics: %s", name, elapsed,
                {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
    return final


def get_block_config():
    """Discover SD1.4 block structure."""
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,
    )
    block_indices = set()
    for mod_name, _ in pipe.unet.named_modules():
        match = re.search(r"(?:blocks|transformer_blocks)\.(\d+)\.", mod_name)
        if match:
            block_indices.add(int(match.group(1)))
    sorted_blocks = sorted(block_indices)
    del pipe
    gpu_cleanup()
    return sorted_blocks


if __name__ == "__main__":
    exp_name = sys.argv[1]

    sorted_blocks = get_block_config()
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

    experiments = {
        "uniform_r4": {
            "name": "uniform_r4", "block_config": uniform_config,
            "identity_rank": 4, "context_rank": 4, "shared_rank": 4,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        "baseline_r8": {
            "name": "baseline_r8", "block_config": uniform_config,
            "identity_rank": 8, "context_rank": 8, "shared_rank": 8,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        "uniform_r16": {
            "name": "uniform_r16", "block_config": uniform_config,
            "identity_rank": 16, "context_rank": 16, "shared_rank": 16,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        "blockwise_no_ccd": {
            "name": "blockwise_no_ccd", "block_config": blockwise_config,
            "identity_rank": 16, "context_rank": 4, "shared_rank": 8,
            "num_steps": 500, "learning_rate": 5e-4,
        },
        "blockwise_ccd": {
            "name": "blockwise_ccd", "block_config": blockwise_config,
            "identity_rank": 16, "context_rank": 4, "shared_rank": 8,
            "num_steps": 500, "learning_rate": 5e-4,
            "use_ccd": True, "lambda_ccd": 0.3,
        },
    }

    if exp_name not in experiments:
        print(f"Unknown experiment: {exp_name}. Available: {list(experiments.keys())}")
        sys.exit(1)

    run_experiment(experiments[exp_name])
