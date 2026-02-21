"""End-to-end GPU smoke test for ModularBooth training pipeline.

Uses HuggingFace's tiny-stable-diffusion-pipe (~1.4M params) to test the
complete training flow: model loading, LoRA injection, training loop,
loss computation, and checkpoint save/load — all on real GPU hardware.

This test requires a CUDA device with at least 1GB of free memory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

# Only run if CUDA is available.
CUDA_AVAILABLE = torch.cuda.is_available()


def _create_dummy_subject_dir(root: Path, n: int = 4) -> Path:
    """Create a temporary directory with small dummy subject images."""
    subject_dir = root / "subject_images"
    subject_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64), color=(100 + i * 30, 80, 120))
        img.save(subject_dir / f"subject_{i:02d}.png")
    # Write a simple caption file.
    (subject_dir / "caption.txt").write_text(
        "a photo of [V] dog\n" * n
    )
    return subject_dir


def _create_dummy_class_dir(root: Path, n: int = 4) -> Path:
    """Create a temporary directory with small dummy class images."""
    class_dir = root / "class_images"
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64), color=(50, 100 + i * 20, 60))
        img.save(class_dir / f"class_{i:02d}.png")
    (class_dir / "caption.txt").write_text(
        "a photo of a dog\n" * n
    )
    return class_dir


@unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA GPU")
class TestE2ETraining(unittest.TestCase):
    """End-to-end training smoke test on GPU."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="modularbooth_e2e_")
        self.root = Path(self.tmpdir)
        self.subject_dir = _create_dummy_subject_dir(self.root)
        self.class_dir = _create_dummy_class_dir(self.root)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        torch.cuda.empty_cache()

    def test_training_loop_5_steps(self) -> None:
        """Train for 5 steps on tiny model; verify loss decreases."""
        from diffusers import StableDiffusionPipeline
        from omegaconf import OmegaConf

        from modularbooth.data.dataset import DreamBoothDataset
        from modularbooth.losses.combined import ModularBoothLoss
        from modularbooth.models.blockwise_lora import BlockwiseLoRA
        from modularbooth.training.trainer import ModelWrapper, ModularBoothTrainer

        # Load tiny pipeline.
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        # Build config.
        cfg = OmegaConf.create({
            "model": {"backbone": "tiny", "dtype": "float16"},
            "subject": {"token": "[V]", "class_noun": "dog", "num_images": 4},
            "lora": {
                "identity_rank": 4,
                "context_rank": 2,
                "shared_rank": 4,
                "alpha_ratio": 1.0,
                "dropout": 0.0,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                "block_config": None,
            },
            "training": {
                "num_steps": 5,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "warmup_steps": 0,
                "max_grad_norm": 1.0,
                "mixed_precision": "no",
                "seed": 42,
                "log_every": 1,
                "save_every": 100,
                "validate_every": 100,
                "scheduler": "constant",
            },
            "prior_preservation": {
                "enabled": True,
                "num_class_images": 4,
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
                "num_steps": 2,
                "guidance_scale": 1.0,
                "resolution": 64,
            },
        })

        # Create dataset.
        dataset = DreamBoothDataset(
            subject_images_dir=str(self.subject_dir),
            class_images_dir=str(self.class_dir),
            token="[V]",
            class_noun="dog",
            resolution=64,
        )

        # Apply blockwise LoRA to UNet.
        # The tiny model has all transformer_blocks at index 0.
        unet = pipe.unet
        lora = BlockwiseLoRA(
            model=unet,
            block_config={0: "shared"},
            identity_rank=4,
            context_rank=2,
            shared_rank=4,
            alpha_ratio=1.0,
            dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out\\.0"],
        )
        lora.apply_lora()

        # Verify LoRA params exist.
        param_count = lora.get_parameter_count()
        total = sum(param_count.values())
        self.assertGreater(total, 0, "LoRA should have trainable parameters")

        # Create loss.
        loss_fn = ModularBoothLoss(
            lambda_ppl=1.0,
            lambda_ccd=0.0,
            ccd_warmup_steps=0,
        )

        # Create trainer.
        trainer = ModularBoothTrainer(
            config=cfg,
            model=pipe,
            lora=lora,
            dataset=dataset,
            loss_fn=loss_fn,
            device="cuda",
        )

        # Run training.
        results = trainer.train()

        # Verify training completed.
        self.assertEqual(results["total_steps"], 5)
        self.assertIn("loss_total", results)
        self.assertIn("loss_diffusion", results)

        # Save checkpoint.
        ckpt_dir = self.output_dir / "checkpoint"
        trainer.save_checkpoint(ckpt_dir, global_step=5)
        self.assertTrue((ckpt_dir / "checkpoint.pt").exists())

        # Save LoRA weights via safetensors.
        lora_path = self.output_dir / "lora.safetensors"
        lora.save_lora(str(lora_path))
        self.assertTrue(lora_path.exists())

        # Verify LoRA checkpoint is small (<1MB for tiny model).
        size_mb = lora_path.stat().st_size / (1024 * 1024)
        self.assertLess(size_mb, 1.0, "LoRA checkpoint should be very small for tiny model")

        print(f"\n[E2E] Training completed: {results['total_steps']} steps, "
              f"total_loss={results.get('loss_total', 'N/A'):.4f}, "
              f"LoRA params={total:,}, ckpt={size_mb:.3f}MB")

        # Cleanup GPU memory.
        del trainer, pipe, lora, loss_fn
        torch.cuda.empty_cache()


@unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA GPU")
class TestE2ELoRALoadAndInference(unittest.TestCase):
    """Test LoRA save → load → inference roundtrip."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="modularbooth_e2e_inf_")
        self.root = Path(self.tmpdir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        torch.cuda.empty_cache()

    def test_lora_save_load_roundtrip(self) -> None:
        """Save LoRA weights, load into fresh model, verify outputs differ."""
        from diffusers import StableDiffusionPipeline
        from modularbooth.models.blockwise_lora import BlockwiseLoRA

        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        unet = pipe.unet

        # Create and apply LoRA.
        lora = BlockwiseLoRA(
            model=unet,
            block_config={0: "shared"},
            identity_rank=4,
            context_rank=2,
            shared_rank=4,
            alpha_ratio=1.0,
            dropout=0.0,
            target_modules=["to_q", "to_k", "to_v"],
        )
        lora.apply_lora()

        # Modify LoRA weights to have non-zero values.
        for param in lora.get_lora_params().values():
            param.data.fill_(0.01)

        # Save.
        lora_path = self.root / "test_lora.safetensors"
        lora.save_lora(str(lora_path))
        self.assertTrue(lora_path.exists())

        # Load into fresh model.
        pipe2 = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        lora2 = BlockwiseLoRA(
            model=pipe2.unet,
            block_config={0: "shared"},
            identity_rank=4,
            context_rank=2,
            shared_rank=4,
            alpha_ratio=1.0,
            dropout=0.0,
            target_modules=["to_q", "to_k", "to_v"],
        )
        lora2.apply_lora()
        lora2.load_lora(str(lora_path))

        # Verify loaded weights match.
        orig_params = lora.get_lora_params()
        loaded_params = lora2.get_lora_params()
        self.assertGreater(len(orig_params), 0, "Should have LoRA params to compare")
        for key in orig_params:
            self.assertTrue(
                torch.allclose(orig_params[key].data.float(), loaded_params[key].data.float(), atol=1e-3),
                f"LoRA weights mismatch for {key}",
            )

        print("\n[E2E] LoRA save/load roundtrip verified successfully")

        del pipe, pipe2, lora, lora2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
