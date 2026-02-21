"""Tests for modularbooth.losses.ccd_loss.

All tests run on CPU without GPU or large model downloads.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from modularbooth.losses.ccd_loss import CCDLoss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCCDLossShape:
    """test_ccd_loss_shape: forward returns a scalar tensor."""

    def test_output_is_scalar(self) -> None:
        torch.manual_seed(0)
        B, D, N_neg = 4, 64, 6

        loss_fn = CCDLoss(temperature=0.07)

        subject = torch.randn(B, D)
        positive = torch.randn(B, D)
        negative = torch.randn(B, N_neg, D)

        loss = loss_fn(subject, positive, negative)

        assert loss.dim() == 0, f"Expected scalar (0-D), got {loss.dim()}-D tensor"
        assert loss.shape == torch.Size([]), f"Expected shape (), got {loss.shape}"
        assert torch.isfinite(loss), "Loss is not finite"


class TestCCDLossPositivePairLower:
    """test_ccd_loss_positive_pair_lower: similar positives yield lower loss."""

    def test_similar_positive_gives_lower_loss(self) -> None:
        torch.manual_seed(1)
        B, D, N_neg = 4, 64, 8

        loss_fn = CCDLoss(temperature=0.07)

        subject = torch.randn(B, D)
        negative = torch.randn(B, N_neg, D)

        # Case 1: positive is very similar to subject (small noise)
        positive_similar = subject + 0.01 * torch.randn(B, D)
        loss_similar = loss_fn(subject, positive_similar, negative)

        # Case 2: positive is completely random (dissimilar)
        positive_random = torch.randn(B, D)
        loss_random = loss_fn(subject, positive_random, negative)

        assert loss_similar < loss_random, (
            f"Expected loss with similar positive ({loss_similar.item():.4f}) "
            f"to be lower than with random positive ({loss_random.item():.4f})"
        )


class TestCCDLossGradientFlow:
    """test_ccd_loss_gradient_flow: gradients flow to all inputs."""

    def test_gradients_flow_to_all_inputs(self) -> None:
        torch.manual_seed(2)
        B, D, N_neg = 4, 32, 5

        loss_fn = CCDLoss(temperature=0.1)

        subject = torch.randn(B, D, requires_grad=True)
        positive = torch.randn(B, D, requires_grad=True)
        negative = torch.randn(B, N_neg, D, requires_grad=True)

        loss = loss_fn(subject, positive, negative)
        loss.backward()

        assert subject.grad is not None, "No gradient for subject_features"
        assert positive.grad is not None, "No gradient for positive_features"
        assert negative.grad is not None, "No gradient for negative_features"

        # Gradients should be non-zero (not a degenerate case)
        assert subject.grad.abs().sum() > 0, "subject_features gradient is all zeros"
        assert positive.grad.abs().sum() > 0, "positive_features gradient is all zeros"
        assert negative.grad.abs().sum() > 0, "negative_features gradient is all zeros"


class TestExtractSubjectFeatures:
    """test_extract_subject_features: masked average pooling correctness."""

    def test_output_shape(self) -> None:
        B, C, H, W = 2, 16, 8, 8
        loss_fn = CCDLoss()

        features = torch.randn(B, C, H, W)
        masks = torch.ones(B, 1, H, W)  # full mask

        pooled = loss_fn.extract_subject_features(features, masks)
        assert pooled.shape == (B, C), f"Expected shape ({B}, {C}), got {pooled.shape}"

    def test_masking_isolates_region(self) -> None:
        """Only the masked region should contribute to the output."""
        B, C, H, W = 1, 4, 8, 8
        loss_fn = CCDLoss()

        # Create features with known values: top half = 1.0, bottom half = -1.0
        features = torch.full((B, C, H, W), -1.0)
        features[:, :, :H // 2, :] = 1.0  # top half is 1.0

        # Mask selects only the top half
        masks = torch.zeros(B, 1, H, W)
        masks[:, :, :H // 2, :] = 1.0

        pooled = loss_fn.extract_subject_features(features, masks)

        # The pooled result should be close to 1.0 (the top-half value)
        assert torch.allclose(pooled, torch.ones(B, C), atol=1e-5), (
            f"Expected pooled features ~1.0, got {pooled}"
        )

    def test_mask_resized_to_feature_spatial_dims(self) -> None:
        """Masks of different spatial size should be resized automatically."""
        B, C = 2, 8
        feat_H, feat_W = 4, 4
        mask_H, mask_W = 16, 16  # 4x larger than features

        loss_fn = CCDLoss()

        features = torch.randn(B, C, feat_H, feat_W)
        # Full mask at higher resolution
        masks = torch.ones(B, 1, mask_H, mask_W)

        pooled = loss_fn.extract_subject_features(features, masks)
        assert pooled.shape == (B, C), (
            f"Expected shape ({B}, {C}) after mask resize, got {pooled.shape}"
        )

    def test_empty_mask_does_not_crash(self) -> None:
        """An all-zero mask should not cause division by zero (clamped area)."""
        B, C, H, W = 1, 4, 8, 8
        loss_fn = CCDLoss()

        features = torch.randn(B, C, H, W)
        masks = torch.zeros(B, 1, H, W)  # empty mask

        pooled = loss_fn.extract_subject_features(features, masks)
        assert pooled.shape == (B, C)
        assert torch.isfinite(pooled).all(), "Non-finite values with empty mask"
