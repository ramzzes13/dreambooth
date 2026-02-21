"""Real DreamBooth training run on SD 1.4 with blockwise LoRA.

This script demonstrates the full ModularBooth pipeline:
1. Load Stable Diffusion 1.4 in fp16
2. Generate subject and class images
3. Apply blockwise LoRA
4. Train with prior-preservation loss
5. Save checkpoint and LoRA weights
6. Generate images with trained model
7. Evaluate DINO/CLIP metrics
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("run_real_training")

DEVICE = "cuda"
OUTPUT_DIR = Path("outputs/sd14_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_subject_images(pipe, output_dir: Path, class_noun: str = "dog", n: int = 4) -> Path:
    """Generate subject images using the pipeline itself."""
    subject_dir = output_dir / "subject_images"
    subject_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        f"a photo of a {class_noun}, professional photography, detailed",
        f"a portrait of a {class_noun}, studio lighting, high quality",
        f"a {class_noun} sitting in a garden, natural lighting",
        f"a close-up of a {class_noun}, sharp focus, detailed fur",
    ]

    logger.info("Generating %d subject images...", n)
    for i in range(n):
        with torch.no_grad():
            result = pipe(
                prompt=prompts[i % len(prompts)],
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
            )
            img = result.images[0]
            img.save(subject_dir / f"subject_{i:02d}.png")
            logger.info("  Generated subject_%02d.png", i)

    return subject_dir


def create_class_images(pipe, output_dir: Path, class_noun: str = "dog", n: int = 8) -> Path:
    """Generate class images for prior-preservation loss."""
    class_dir = output_dir / "class_images"
    class_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d class images...", n)
    for i in range(n):
        with torch.no_grad():
            result = pipe(
                prompt=f"a photo of a {class_noun}",
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
            )
            img = result.images[0]
            img.save(class_dir / f"class_{i:02d}.png")
            logger.info("  Generated class_%02d.png", i)

    return class_dir


def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("ModularBooth Real Training Run")
    logger.info("=" * 60)

    # Check GPU memory
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("GPU: %s (%.1f GB total)", torch.cuda.get_device_name(0), gpu_mem)

    # ====================================================================
    # Step 1: Load pipeline
    # ====================================================================
    logger.info("\n--- Step 1: Loading Stable Diffusion 1.4 ---")
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    logger.info("Pipeline loaded. Memory used: %.1f MB", torch.cuda.memory_allocated() / 1e6)

    # ====================================================================
    # Step 2: Generate training data
    # ====================================================================
    logger.info("\n--- Step 2: Generating training data ---")
    subject_dir = create_subject_images(pipe, OUTPUT_DIR, "dog", n=4)
    class_dir = create_class_images(pipe, OUTPUT_DIR, "dog", n=8)

    # ====================================================================
    # Step 3: Set up dataset
    # ====================================================================
    logger.info("\n--- Step 3: Setting up dataset ---")
    from modularbooth.data.dataset import DreamBoothDataset

    dataset = DreamBoothDataset(
        subject_images_dir=str(subject_dir),
        class_images_dir=str(class_dir),
        token="[V]",
        class_noun="dog",
        resolution=512,
    )
    logger.info("Dataset created: %d samples", len(dataset))

    # ====================================================================
    # Step 4: Apply blockwise LoRA
    # ====================================================================
    logger.info("\n--- Step 4: Applying blockwise LoRA ---")
    import re
    from modularbooth.models.blockwise_lora import BlockwiseLoRA

    unet = pipe.unet

    # Discover transformer block indices in the UNet
    block_indices = set()
    for name, _ in unet.named_modules():
        match = re.search(r"(?:blocks|transformer_blocks|joint_blocks|single_blocks)\.(\d+)\.", name)
        if match:
            block_indices.add(int(match.group(1)))

    # Assign all blocks as shared for this baseline run
    block_config = {idx: "shared" for idx in block_indices}
    logger.info("Discovered %d transformer block indices: %s", len(block_indices), sorted(block_indices))

    lora = BlockwiseLoRA(
        model=unet,
        block_config=block_config,
        identity_rank=8,
        context_rank=4,
        shared_rank=8,
        alpha_ratio=1.0,
        dropout=0.05,
        target_modules=[r"to_q", r"to_k", r"to_v", r"to_out\.0"],
    )
    lora.apply_lora()

    param_count = lora.get_parameter_count()
    total_params = sum(param_count.values())
    logger.info("LoRA applied: %d trainable parameters (%s)", total_params, param_count)

    # ====================================================================
    # Step 5: Build config and train
    # ====================================================================
    logger.info("\n--- Step 5: Training ---")
    from modularbooth.losses.combined import ModularBoothLoss
    from modularbooth.training.trainer import ModularBoothTrainer

    cfg = OmegaConf.create({
        "model": {"backbone": "CompVis/stable-diffusion-v1-4", "dtype": "float16"},
        "subject": {"token": "[V]", "class_noun": "dog", "num_images": 4},
        "lora": {
            "identity_rank": 8,
            "context_rank": 4,
            "shared_rank": 8,
            "alpha_ratio": 1.0,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "block_config": None,
        },
        "training": {
            "num_steps": 100,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "seed": 42,
            "log_every": 10,
            "save_every": 50,
            "validate_every": 50,
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

    loss_fn = ModularBoothLoss(
        lambda_ppl=1.0,
        lambda_ccd=0.0,
        ccd_warmup_steps=0,
    )

    trainer = ModularBoothTrainer(
        config=cfg,
        model=pipe,
        lora=lora,
        dataset=dataset,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    logger.info("Starting training for %d steps...", cfg.training.num_steps)
    train_start = time.time()
    results = trainer.train()
    train_elapsed = time.time() - train_start

    logger.info("Training completed in %.1f seconds", train_elapsed)
    logger.info("Results: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in results.items()})

    # ====================================================================
    # Step 6: Save checkpoint
    # ====================================================================
    logger.info("\n--- Step 6: Saving checkpoint ---")
    ckpt_dir = OUTPUT_DIR / "checkpoint"
    trainer.save_checkpoint(ckpt_dir, global_step=cfg.training.num_steps)

    lora_path = OUTPUT_DIR / "lora_weights.safetensors"
    lora.save_lora(str(lora_path))
    lora_size_mb = lora_path.stat().st_size / (1024 * 1024)
    logger.info("LoRA weights saved: %.2f MB", lora_size_mb)

    # ====================================================================
    # Step 7: Generate images with trained model
    # ====================================================================
    logger.info("\n--- Step 7: Generating images with trained LoRA ---")
    gen_dir = OUTPUT_DIR / "generated"
    gen_dir.mkdir(exist_ok=True)

    eval_prompts = [
        "a photo of [V] dog on a beach",
        "a photo of [V] dog in a garden",
        "a photo of [V] dog wearing a hat",
        "a photo of [V] dog in a snowy landscape",
    ]

    generated_images = []
    # Use the pipeline directly (LoRA is already injected into unet)
    pipe.to(DEVICE)
    for i, prompt in enumerate(eval_prompts):
        # Replace [V] with sks for the tokenizer (since we can't really add
        # a new token without training the text encoder too)
        with torch.no_grad():
            result = pipe(
                prompt=prompt.replace("[V]", "sks"),
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
            )
        img = result.images[0]
        img.save(gen_dir / f"generated_{i:02d}.png")
        generated_images.append(img)
        logger.info("  Generated image %d: %s", i, prompt)

    # ====================================================================
    # Step 8: Evaluate metrics
    # ====================================================================
    logger.info("\n--- Step 8: Evaluating metrics ---")

    # Load subject reference images
    subject_images = []
    for p in sorted(subject_dir.glob("*.png")):
        subject_images.append(Image.open(p).convert("RGB"))

    # DINO score
    try:
        from modularbooth.evaluation.dino_score import DINOScore
        dino = DINOScore(device=DEVICE)
        dino_score = dino.compute_score(generated_images, subject_images)
        logger.info("DINO score: %.4f", dino_score)
    except Exception as e:
        logger.warning("DINO score failed: %s", e)
        dino_score = None

    # CLIP-T score
    try:
        from modularbooth.evaluation.clip_score import CLIPScore
        clip_scorer = CLIPScore(device=DEVICE)
        clip_t = clip_scorer.clip_t_score(
            generated_images, [p.replace("[V]", "sks") for p in eval_prompts]
        )
        logger.info("CLIP-T score: %.4f", clip_t)
    except Exception as e:
        logger.warning("CLIP-T score failed: %s", e)
        clip_t = None

    # CLIP-I score (image-image similarity to subject references)
    try:
        clip_i = clip_scorer.clip_i_score(generated_images, subject_images)
        logger.info("CLIP-I score: %.4f", clip_i)
    except Exception as e:
        logger.warning("CLIP-I score failed: %s", e)
        clip_i = None

    # VQA alignment
    try:
        from modularbooth.evaluation.vqa_alignment import VQAAlignment
        vqa = VQAAlignment(device=DEVICE)
        vqa_score = vqa.compute_batch_alignment(
            generated_images, [p.replace("[V]", "sks") for p in eval_prompts]
        )
        logger.info("VQA alignment: %.4f", vqa_score)
    except Exception as e:
        logger.warning("VQA alignment failed: %s", e)
        vqa_score = None

    # LPIPS diversity
    try:
        from modularbooth.evaluation.diversity import LPIPSDiversity
        lpips_scorer = LPIPSDiversity(device=DEVICE)
        lpips_div = lpips_scorer.compute_diversity(generated_images)
        logger.info("LPIPS diversity: %.4f", lpips_div)
    except Exception as e:
        logger.warning("LPIPS diversity failed: %s", e)
        lpips_div = None

    # ====================================================================
    # Summary
    # ====================================================================
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RUN COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time: %.1f seconds", elapsed)
    logger.info("Training time: %.1f seconds", train_elapsed)
    logger.info("Training steps: %d", results.get("total_steps", 0))
    logger.info("Final loss: %.4f", results.get("loss_total", 0))
    logger.info("LoRA params: %d", total_params)
    logger.info("LoRA size: %.2f MB", lora_size_mb)
    if dino_score is not None:
        logger.info("DINO score: %.4f", dino_score)
    if clip_t is not None:
        logger.info("CLIP-T score: %.4f", clip_t)
    if clip_i is not None:
        logger.info("CLIP-I score: %.4f", clip_i)
    if vqa_score is not None:
        logger.info("VQA alignment: %.4f", vqa_score)
    if lpips_div is not None:
        logger.info("LPIPS diversity: %.4f", lpips_div)

    # Save results to JSON
    import json
    results_data = {
        "model": "CompVis/stable-diffusion-v1-4",
        "total_time_s": elapsed,
        "training_time_s": train_elapsed,
        "training_steps": results.get("total_steps", 0),
        "final_loss_total": results.get("loss_total", 0),
        "final_loss_diffusion": results.get("loss_diffusion", 0),
        "final_loss_ppl": results.get("loss_ppl", 0),
        "lora_params": total_params,
        "lora_size_mb": lora_size_mb,
        "metrics": {
            "dino_score": dino_score,
            "clip_t_score": clip_t,
            "clip_i_score": clip_i,
            "vqa_alignment": vqa_score,
            "lpips_diversity": lpips_div,
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    logger.info("Results saved to %s", OUTPUT_DIR / "results.json")


if __name__ == "__main__":
    main()
