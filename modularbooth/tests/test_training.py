"""Smoke tests for the ModularBooth training pipeline.

All heavy dependencies (diffusers pipeline, real LoRA modules, GPU) are
mocked.  These tests verify construction, scheduling, loss forward pass,
and callback contracts without loading any real models.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ===================================================================
# ModelWrapper tests
# ===================================================================


class TestModelWrapperInit(unittest.TestCase):
    """ModelWrapper should initialise successfully when provided with
    a mock pipeline that has the expected attributes.
    """

    def test_model_wrapper_init(self) -> None:
        from modularbooth.training.trainer import ModelWrapper

        # Build a minimal mock pipeline with a transformer and scheduler.
        mock_pipeline = MagicMock()
        mock_pipeline.transformer = nn.Linear(4, 4)
        mock_pipeline.unet = None
        mock_pipeline.scheduler = MagicMock()

        wrapper = ModelWrapper(pipeline=mock_pipeline, device="cpu", dtype=torch.float32)

        self.assertIs(wrapper.denoiser, mock_pipeline.transformer)
        self.assertEqual(wrapper.device, "cpu")
        self.assertEqual(wrapper.dtype, torch.float32)


class TestModelWrapperInitNoBackbone(unittest.TestCase):
    """ModelWrapper should raise ValueError when neither transformer nor
    unet is present on the pipeline.
    """

    def test_model_wrapper_init_no_backbone(self) -> None:
        from modularbooth.training.trainer import ModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.transformer = None
        mock_pipeline.unet = None

        with self.assertRaises(ValueError):
            ModelWrapper(pipeline=mock_pipeline, device="cpu")


# ===================================================================
# WarmupScheduler tests
# ===================================================================


class TestWarmupScheduler(unittest.TestCase):
    """Step the scheduler through warmup and verify that the LR ramps
    up linearly, then follows the base schedule.
    """

    def test_warmup_scheduler(self) -> None:
        from modularbooth.training.scheduler import WarmupScheduler

        base_lr = 1e-4
        warmup_steps = 5

        # Simple model to provide parameters.
        model = nn.Linear(4, 2)
        optimizer = AdamW(model.parameters(), lr=base_lr)

        # Base scheduler: constant LR after warmup.
        from torch.optim.lr_scheduler import LambdaLR
        base_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        scheduler = WarmupScheduler(
            optimizer, base_scheduler, warmup_steps=warmup_steps
        )

        # During warmup the LR should ramp linearly from 0 to base_lr.
        lrs: list[float] = []
        for step in range(warmup_steps + 5):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # At step 0 the LR should be 0 (or very close).
        self.assertAlmostEqual(lrs[0], 0.0, places=8)

        # LR should strictly increase during warmup.
        for i in range(1, warmup_steps):
            self.assertGreater(
                lrs[i], lrs[i - 1],
                f"LR should increase during warmup: step {i-1}={lrs[i-1]}, step {i}={lrs[i]}",
            )

        # After warmup completes, LR should be at (or near) base_lr.
        self.assertAlmostEqual(lrs[warmup_steps], base_lr, places=7)


# ===================================================================
# ModularBoothLoss tests
# ===================================================================


class TestCombinedLossForward(unittest.TestCase):
    """Forward pass of ModularBoothLoss with dummy tensors should return
    a dict with all expected keys.
    """

    def test_combined_loss_forward(self) -> None:
        from modularbooth.losses.combined import ModularBoothLoss

        loss_fn = ModularBoothLoss(
            lambda_ppl=1.0,
            lambda_ccd=0.3,
            ccd_warmup_steps=0,
            ccd_temperature=0.07,
        )

        B, D = 2, 16
        pred_s = torch.randn(B, D)
        target_s = torch.randn(B, D)
        pred_c = torch.randn(B, D)
        target_c = torch.randn(B, D)

        # With CCD features
        subject_feat = torch.randn(B, D)
        positive_feat = torch.randn(B, D)
        negative_feat = torch.randn(B, 3, D)  # 3 negatives

        result = loss_fn(
            model_pred_subject=pred_s,
            noise_target_subject=target_s,
            model_pred_class=pred_c,
            noise_target_class=target_c,
            subject_features=subject_feat,
            positive_features=positive_feat,
            negative_features=negative_feat,
            global_step=10,
        )

        expected_keys = {"total_loss", "diffusion_loss", "ppl_loss", "ccd_loss", "loss_components"}
        self.assertTrue(
            expected_keys.issubset(result.keys()),
            f"Missing keys: {expected_keys - result.keys()}",
        )

        # total_loss should be a scalar tensor suitable for backward
        self.assertIsInstance(result["total_loss"], torch.Tensor)
        self.assertEqual(result["total_loss"].ndim, 0)

        # loss_components should be a dict of floats
        self.assertIsInstance(result["loss_components"], dict)
        for k, v in result["loss_components"].items():
            self.assertIsInstance(v, float, f"loss_components[{k!r}] should be float")


class TestCombinedLossCCDWarmup(unittest.TestCase):
    """CCD loss should be 0 before ccd_warmup_steps and non-zero after."""

    def test_combined_loss_ccd_warmup(self) -> None:
        from modularbooth.losses.combined import ModularBoothLoss

        warmup_steps = 50
        loss_fn = ModularBoothLoss(
            lambda_ppl=1.0,
            lambda_ccd=0.3,
            ccd_warmup_steps=warmup_steps,
        )

        B, D = 2, 16
        pred_s = torch.randn(B, D)
        target_s = torch.randn(B, D)
        pred_c = torch.randn(B, D)
        target_c = torch.randn(B, D)
        subject_feat = torch.randn(B, D)
        positive_feat = torch.randn(B, D)
        negative_feat = torch.randn(B, 3, D)

        # Before warmup: CCD should be 0.
        result_before = loss_fn(
            pred_s, target_s, pred_c, target_c,
            subject_feat, positive_feat, negative_feat,
            global_step=warmup_steps - 1,
        )
        self.assertAlmostEqual(
            result_before["ccd_loss"].item(), 0.0, places=6,
            msg="CCD loss should be 0 before warmup completes",
        )

        # After warmup: CCD should be > 0.
        result_after = loss_fn(
            pred_s, target_s, pred_c, target_c,
            subject_feat, positive_feat, negative_feat,
            global_step=warmup_steps,
        )
        self.assertGreater(
            result_after["ccd_loss"].item(), 0.0,
            "CCD loss should be non-zero after warmup completes",
        )


# ===================================================================
# Callback tests
# ===================================================================


class TestLoggingCallback(unittest.TestCase):
    """LoggingCallback.on_step_end should not crash when called with
    a mock trainer and a dict of log values.
    """

    def test_logging_callback(self) -> None:
        from modularbooth.training.callbacks import LoggingCallback

        cb = LoggingCallback(log_every=1, use_wandb=False)

        mock_trainer = MagicMock()
        mock_trainer.config.training.num_steps = 100
        mock_trainer.config.training.batch_size = 1
        mock_trainer.config.training.gradient_accumulation = 1

        logs = {
            "loss_total": 0.45,
            "loss_diffusion": 0.30,
            "lr": 1e-4,
        }

        # Should not raise.
        cb.on_train_begin(mock_trainer)
        cb.on_step_end(mock_trainer, global_step=1, logs=logs)
        cb.on_step_end(mock_trainer, global_step=2, logs=logs)


class TestCheckpointCallback(unittest.TestCase):
    """CheckpointCallback should create checkpoint directories when
    save_checkpoint is called on the trainer mock.
    """

    def test_checkpoint_callback(self) -> None:
        from modularbooth.training.callbacks import CheckpointCallback

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ckpts"

            cb = CheckpointCallback(save_every=1, output_dir=str(output_dir))

            mock_trainer = MagicMock()
            mock_trainer.config.training.num_steps = 10

            # save_checkpoint should create the directory.
            def _fake_save(path, step):
                Path(path).mkdir(parents=True, exist_ok=True)

            mock_trainer.save_checkpoint.side_effect = _fake_save

            cb.on_train_begin(mock_trainer)

            # Simulate step 1 (save_every=1 triggers save).
            cb.on_step_end(mock_trainer, global_step=1, logs={})

            ckpt_dir = output_dir / "step_1"
            self.assertTrue(ckpt_dir.exists(), f"Checkpoint dir should exist: {ckpt_dir}")


if __name__ == "__main__":
    unittest.main()
