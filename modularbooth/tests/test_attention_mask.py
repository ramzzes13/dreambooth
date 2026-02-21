"""Tests for modularbooth.models.attention_mask.

All tests run on CPU without GPU or large model downloads.
"""

from __future__ import annotations

import pytest
import torch

from modularbooth.models.attention_mask import TokenAwareAttentionMask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateSpatialMask:
    """test_create_spatial_mask: shape, values inside/outside boxes."""

    def test_shape(self) -> None:
        masker = TokenAwareAttentionMask()
        boxes = [(10, 20, 50, 60), (70, 80, 120, 150)]
        H, W = 256, 256
        masks = masker.create_spatial_mask(boxes, image_size=(H, W))

        N = len(boxes)
        assert masks.shape == (N, H, W), f"Expected ({N}, {H}, {W}), got {masks.shape}"

    def test_values_inside_box_are_one(self) -> None:
        masker = TokenAwareAttentionMask()
        x_min, y_min, x_max, y_max = 10, 20, 50, 60
        H, W = 128, 128
        alpha = 0.05

        masks = masker.create_spatial_mask(
            [(x_min, y_min, x_max, y_max)],
            image_size=(H, W),
            leakage_alpha=alpha,
        )

        inside = masks[0, y_min:y_max, x_min:x_max]
        assert (inside == 1.0).all(), "Expected 1.0 inside the bounding box"

    def test_values_outside_box_are_alpha(self) -> None:
        masker = TokenAwareAttentionMask()
        x_min, y_min, x_max, y_max = 10, 20, 50, 60
        H, W = 128, 128
        alpha = 0.05

        masks = masker.create_spatial_mask(
            [(x_min, y_min, x_max, y_max)],
            image_size=(H, W),
            leakage_alpha=alpha,
        )

        # Check a region fully outside the box (above the box)
        outside_above = masks[0, 0:y_min, :]
        assert (outside_above == alpha).all(), (
            f"Expected {alpha} outside the box, found other values"
        )

        # Check a region fully to the right of the box
        outside_right = masks[0, :, x_max:]
        assert (outside_right == alpha).all(), (
            f"Expected {alpha} to the right of the box"
        )


class TestCreateLatentMask:
    """test_create_latent_mask: correct downscaling from pixel to latent coords."""

    def test_downscale_shape(self) -> None:
        masker = TokenAwareAttentionMask()
        # Pixel-space box
        boxes = [(16, 16, 48, 48)]
        latent_H, latent_W = 8, 8
        downscale = 8

        masks = masker.create_latent_mask(
            boxes,
            latent_size=(latent_H, latent_W),
            downscale_factor=downscale,
        )
        assert masks.shape == (1, latent_H, latent_W)

    def test_downscale_places_box_correctly(self) -> None:
        masker = TokenAwareAttentionMask()
        # Box at pixel (0, 0) to (16, 16) with downscale=8 should cover
        # latent (0,0) to (2,2)
        boxes = [(0, 0, 16, 16)]
        latent_H, latent_W = 8, 8
        downscale = 8
        alpha = 0.05

        masks = masker.create_latent_mask(
            boxes,
            latent_size=(latent_H, latent_W),
            leakage_alpha=alpha,
            downscale_factor=downscale,
        )

        # The scaled box is (0, 0, 2, 2), so latent positions (0:2, 0:2) = 1.0
        inside = masks[0, 0:2, 0:2]
        assert (inside == 1.0).all(), "Expected 1.0 in downscaled box region"

        # Outside region should be alpha
        outside = masks[0, 3:, 3:]
        assert (outside == alpha).all(), "Expected alpha outside downscaled box"


class TestApplyNegativeAttention:
    """test_apply_negative_attention: scores reduced outside mask region."""

    def test_scores_reduced_outside_mask(self) -> None:
        masker = TokenAwareAttentionMask()

        H, W = 4, 4
        seq_len = H * W
        B, heads = 2, 2
        gamma = 3.0

        # Attention scores: all ones
        attn = torch.ones(B, heads, seq_len, seq_len)

        # Mask: first 4 positions are inside (1.0), rest are outside (0.05)
        mask = torch.full((H, W), 0.05)
        mask[0, :] = 1.0  # first row = inside region

        adjusted = masker.apply_negative_attention(attn, mask, gamma=gamma)

        # Flatten mask to check
        flat_mask = mask.reshape(-1)

        # For positions where mask ~ 1.0, penalty ~ 0, so adjusted ~ 1.0
        inside_idx = (flat_mask > 0.5).nonzero(as_tuple=True)[0]
        outside_idx = (flat_mask < 0.5).nonzero(as_tuple=True)[0]

        # adjusted[..., inside_idx] should be close to original (1.0)
        inside_vals = adjusted[:, :, :, inside_idx]
        assert torch.allclose(inside_vals, torch.ones_like(inside_vals), atol=0.01), (
            "Inside-region attention scores should be nearly unchanged"
        )

        # adjusted[..., outside_idx] should be reduced (1.0 - gamma*(1-0.05)) = 1.0 - 2.85 < 0
        outside_vals = adjusted[:, :, :, outside_idx]
        assert (outside_vals < attn[:, :, :, outside_idx]).all(), (
            "Outside-region attention scores should be reduced"
        )

    def test_3d_attention_scores(self) -> None:
        """Also works with (B, Q, K) shaped attention."""
        masker = TokenAwareAttentionMask()
        seq_len = 16
        B = 2
        gamma = 2.0

        attn = torch.ones(B, seq_len, seq_len)
        mask = torch.ones(seq_len)  # all inside -> no penalty
        adjusted = masker.apply_negative_attention(attn, mask, gamma=gamma)

        assert torch.allclose(attn, adjusted), (
            "With a full mask of 1.0, attention should be unchanged"
        )


class TestBlendMasksNormalize:
    """test_blend_masks_normalize: overlapping masks sum to ~1 per pixel."""

    def test_normalize_sums_to_one(self) -> None:
        masker = TokenAwareAttentionMask()
        H, W = 32, 32

        # Two overlapping masks
        mask1 = torch.zeros(H, W)
        mask1[:, :20] = 1.0  # left region

        mask2 = torch.zeros(H, W)
        mask2[:, 12:] = 1.0  # right region (overlaps cols 12-19)

        blended = masker.blend_masks([mask1, mask2], overlap_strategy="normalize")
        assert blended.shape == (2, H, W)

        # Per-pixel sum should be ~1.0 everywhere at least one mask is active
        pixel_sum = blended.sum(dim=0)

        # Where both masks are zero, the sum might not be exactly 1 (both are 0),
        # but in our setup every column has at least one active mask.
        assert torch.allclose(pixel_sum, torch.ones(H, W), atol=1e-5), (
            "Per-pixel sum of normalized masks should be 1.0"
        )

    def test_distance_blend_sums_to_one(self) -> None:
        masker = TokenAwareAttentionMask()
        H, W = 32, 32

        mask1 = torch.zeros(H, W)
        mask1[5:15, 5:15] = 1.0

        mask2 = torch.zeros(H, W)
        mask2[10:25, 10:25] = 1.0

        blended = masker.blend_masks([mask1, mask2], overlap_strategy="distance")
        assert blended.shape == (2, H, W)

        pixel_sum = blended.sum(dim=0)
        # Only check pixels where at least one original mask is active.
        # Uncovered pixels (both masks == 0) will have sum ~0 which is expected.
        active = (mask1 + mask2) > 0
        assert torch.allclose(pixel_sum[active], torch.ones_like(pixel_sum[active]), atol=1e-4), (
            "Per-pixel sum of distance-blended masks should be ~1.0 where masks are active"
        )


class TestFeatherMask:
    """test_feather_mask: feathered mask has smoothed edges."""

    def test_feathered_edges_are_smooth(self) -> None:
        """A hard-edged mask should have smoother edges after feathering."""
        H, W = 64, 64
        mask = torch.zeros(H, W)
        mask[16:48, 16:48] = 1.0  # hard square

        masker = TokenAwareAttentionMask()
        feathered = masker.feather_mask(mask, kernel_size=7)

        assert feathered.shape == mask.shape

        # The hard mask has a sharp jump from 0 to 1 at boundary.
        # After feathering, pixels just outside the box should be > 0 (smoothed).
        just_outside_row = feathered[15, 16:48]  # one row above the box
        assert (just_outside_row > 0).all(), (
            "Feathered mask should have non-zero values just outside the hard boundary"
        )

        # Pixels just inside should be < 1.0 (blurred inward too)
        just_inside_row = feathered[16, 16:48]
        assert (just_inside_row < 1.0).any(), (
            "Feathered mask should have values < 1.0 at the box edge"
        )

    def test_feathered_interior_nearly_unchanged(self) -> None:
        """Center of a large box should remain close to 1.0."""
        H, W = 64, 64
        mask = torch.zeros(H, W)
        mask[16:48, 16:48] = 1.0

        masker = TokenAwareAttentionMask()
        feathered = masker.feather_mask(mask, kernel_size=5)

        # Deep interior should be very close to 1.0
        interior = feathered[24:40, 24:40]
        assert (interior > 0.99).all(), (
            "Interior of feathered mask should remain close to 1.0"
        )

    def test_feather_3d_input(self) -> None:
        """Feathering should also work on (N, H, W) input."""
        N, H, W = 3, 32, 32
        masks = torch.zeros(N, H, W)
        masks[:, 8:24, 8:24] = 1.0

        masker = TokenAwareAttentionMask()
        feathered = masker.feather_mask(masks, kernel_size=5)

        assert feathered.shape == (N, H, W), (
            f"Expected shape ({N}, {H}, {W}), got {feathered.shape}"
        )

    def test_feather_even_kernel_raises(self) -> None:
        """Even kernel_size should raise ValueError."""
        mask = torch.ones(8, 8)
        masker = TokenAwareAttentionMask()
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            masker.feather_mask(mask, kernel_size=4)
