"""Ablation study: Baseline vs. Blockwise LoRA rank configuration.

Compares training with:
1. Uniform shared-rank LoRA (all blocks rank 8)
2. Blockwise LoRA (identity blocks rank 16, context blocks rank 4, shared rank 8)

Both trained on the same data with prior-preservation loss.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("run_ablation")

DEVICE = "cuda"
OUTPUT_BASE = Path("outputs/ablation")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def run_experiment(
    name: str,
    block_config: dict[int, str],
    identity_rank: int,
    context_rank: int,
    shared_rank: int,
    num_steps: int = 200,
) -> dict:
    """Run a single ablation experiment."""
    output_dir = OUTPUT_BASE / name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT: %s", name)
    logger.info("=" * 60)
    logger.info("  block_config: %s", block_config)
    logger.info("  ranks: identity=%d, context=%d, shared=%d", identity_rank, context_rank, shared_rank)

    from diffusers import StableDiffusionPipeline
    from modularbooth.data.dataset import DreamBoothDataset
    from modularbooth.losses.combined import ModularBoothLoss
    from modularbooth.models.blockwise_lora import BlockwiseLoRA
    from modularbooth.training.trainer import ModularBoothTrainer

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    # Use pre-generated images from the first run
    subject_dir = Path("outputs/sd14_training/subject_images")
    class_dir = Path("outputs/sd14_training/class_images")

    if not subject_dir.exists():
        logger.error("Subject images not found. Run run_real_training.py first.")
        return {}

    # Dataset
    dataset = DreamBoothDataset(
        subject_images_dir=str(subject_dir),
        class_images_dir=str(class_dir),
        token="[V]",
        class_noun="dog",
        resolution=512,
    )

    # Apply LoRA
    unet = pipe.unet
    lora = BlockwiseLoRA(
        model=unet,
        block_config=block_config,
        identity_rank=identity_rank,
        context_rank=context_rank,
        shared_rank=shared_rank,
        alpha_ratio=1.0,
        dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()

    param_count = lora.get_parameter_count()
    total_params = sum(param_count.values())
    logger.info("LoRA params: %d (%s)", total_params, param_count)

    # Config
    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "dog", "num_images": 4},
        "lora": {
            "identity_rank": identity_rank,
            "context_rank": context_rank,
            "shared_rank": shared_rank,
            "alpha_ratio": 1.0,
            "dropout": 0.05,
        },
        "training": {
            "num_steps": num_steps,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "seed": 42,
            "log_every": 10,
            "save_every": 100,
            "validate_every": 100,
            "scheduler": "cosine",
        },
        "prior_preservation": {
            "enabled": True,
            "num_class_images": 8,
            "lambda_ppl": 1.0,
        },
        "ccd": {
            "enabled": False,
            "lambda_ccd": 0.0,
            "temperature": 0.07,
            "warmup_steps": 100,
            "feature_layer": "middle",
        },
        "inference": {
            "num_steps": 20,
            "guidance_scale": 7.5,
            "resolution": 512,
        },
    })

    loss_fn = ModularBoothLoss(lambda_ppl=1.0, lambda_ccd=0.0, ccd_warmup_steps=0)

    trainer = ModularBoothTrainer(
        config=cfg,
        model=pipe,
        lora=lora,
        dataset=dataset,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    # Train
    train_start = time.time()
    results = trainer.train()
    train_elapsed = time.time() - train_start
    logger.info("Training: %.1fs, loss=%.4f", train_elapsed, results.get("loss_total", 0))

    # Save LoRA
    lora_path = output_dir / "lora_weights.safetensors"
    lora.save_lora(str(lora_path))
    lora_size_mb = lora_path.stat().st_size / (1024 * 1024)

    # Generate images
    gen_dir = output_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    eval_prompts = [
        "a photo of sks dog on a beach",
        "a photo of sks dog in a garden",
        "a photo of sks dog wearing a hat",
        "a photo of sks dog in a snowy landscape",
    ]

    generated_images = []
    for i, prompt in enumerate(eval_prompts):
        with torch.no_grad():
            result = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.5, height=512, width=512)
        img = result.images[0]
        img.save(gen_dir / f"generated_{i:02d}.png")
        generated_images.append(img)

    # Load subject references
    subject_images = [Image.open(p).convert("RGB") for p in sorted(subject_dir.glob("*.png"))]

    # Evaluate
    metrics = {}

    try:
        from modularbooth.evaluation.dino_score import DINOScore
        dino = DINOScore(device=DEVICE)
        metrics["dino_score"] = dino.compute_score(generated_images, subject_images)
    except Exception as e:
        logger.warning("DINO failed: %s", e)

    try:
        from modularbooth.evaluation.clip_score import CLIPScore
        clip_scorer = CLIPScore(device=DEVICE)
        metrics["clip_t_score"] = clip_scorer.clip_t_score(generated_images, eval_prompts)
        metrics["clip_i_score"] = clip_scorer.clip_i_score(generated_images, subject_images)
    except Exception as e:
        logger.warning("CLIP failed: %s", e)

    try:
        from modularbooth.evaluation.vqa_alignment import VQAAlignment
        vqa = VQAAlignment(device=DEVICE)
        metrics["vqa_alignment"] = vqa.compute_batch_alignment(generated_images, eval_prompts)
    except Exception as e:
        logger.warning("VQA failed: %s", e)

    try:
        from modularbooth.evaluation.diversity import LPIPSDiversity
        lpips_scorer = LPIPSDiversity(device=DEVICE)
        metrics["lpips_diversity"] = lpips_scorer.compute_diversity(generated_images)
    except Exception as e:
        logger.warning("LPIPS failed: %s", e)

    experiment_data = {
        "name": name,
        "block_config": {str(k): v for k, v in block_config.items()},
        "identity_rank": identity_rank,
        "context_rank": context_rank,
        "shared_rank": shared_rank,
        "training_steps": num_steps,
        "training_time_s": train_elapsed,
        "final_loss_total": results.get("loss_total", 0),
        "final_loss_diffusion": results.get("loss_diffusion", 0),
        "final_loss_ppl": results.get("loss_ppl", 0),
        "lora_params": total_params,
        "lora_param_breakdown": param_count,
        "lora_size_mb": lora_size_mb,
        "metrics": metrics,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(experiment_data, f, indent=2, default=str)

    # Cleanup
    del trainer, pipe, lora, loss_fn
    gc.collect()
    torch.cuda.empty_cache()

    return experiment_data


def main():
    start_time = time.time()

    # Discover block structure from SD 1.4
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    unet = pipe.unet
    block_indices = set()
    for name, _ in unet.named_modules():
        match = re.search(r"(?:blocks|transformer_blocks)\.(\d+)\.", name)
        if match:
            block_indices.add(int(match.group(1)))
    sorted_blocks = sorted(block_indices)
    logger.info("Discovered transformer block indices: %s", sorted_blocks)
    del pipe
    gc.collect()

    # Experiment 1: Uniform shared-rank (all blocks rank 8)
    uniform_config = {idx: "shared" for idx in sorted_blocks}
    exp1 = run_experiment(
        name="uniform_rank8",
        block_config=uniform_config,
        identity_rank=8,
        context_rank=8,
        shared_rank=8,
        num_steps=200,
    )

    # Experiment 2: Blockwise ranks (identity=16, context=4, shared=8)
    # Heuristic: early blocks are identity, late blocks are context, middle are shared
    n_blocks = len(sorted_blocks)
    blockwise_config = {}
    for i, idx in enumerate(sorted_blocks):
        if i < n_blocks // 3:
            blockwise_config[idx] = "identity"
        elif i >= 2 * n_blocks // 3:
            blockwise_config[idx] = "context"
        else:
            blockwise_config[idx] = "shared"
    exp2 = run_experiment(
        name="blockwise_16_4_8",
        block_config=blockwise_config,
        identity_rank=16,
        context_rank=4,
        shared_rank=8,
        num_steps=200,
    )

    # Experiment 3: Higher rank everywhere (rank 16)
    exp3 = run_experiment(
        name="uniform_rank16",
        block_config=uniform_config,
        identity_rank=16,
        context_rank=16,
        shared_rank=16,
        num_steps=200,
    )

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 60)

    summary = []
    for exp in [exp1, exp2, exp3]:
        if not exp:
            continue
        m = exp.get("metrics", {})
        logger.info(
            "  %-20s | loss=%.4f | DINO=%.3f | CLIP-T=%.3f | CLIP-I=%.3f | VQA=%.3f | LPIPS=%.3f | params=%d",
            exp["name"],
            exp.get("final_loss_total", 0),
            m.get("dino_score", 0),
            m.get("clip_t_score", 0),
            m.get("clip_i_score", 0),
            m.get("vqa_alignment", 0),
            m.get("lpips_diversity", 0),
            exp.get("lora_params", 0),
        )
        summary.append(exp)

    with open(OUTPUT_BASE / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", OUTPUT_BASE / "ablation_summary.json")


if __name__ == "__main__":
    main()
