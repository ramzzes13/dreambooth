"""Token-Aware Attention Masking for multi-subject spatial composition.

When generating images with multiple personalised subjects, each subject's
LoRA adapter should primarily influence the spatial region assigned to that
subject. This module provides utilities to:

1. Create soft spatial masks from bounding boxes (at image or latent resolution).
2. Apply *negative attention* to suppress a subject's influence outside its
   region by biasing attention logits before softmax.
3. Blend overlapping masks using distance-based soft weighting.
4. Feather mask edges with Gaussian smoothing for seamless transitions.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


class TokenAwareAttentionMask:
    """Create and apply spatial attention masks for multi-subject composition.

    All methods are stateless and operate directly on tensors. The class serves
    as a logical namespace.

    Typical usage::

        masker = TokenAwareAttentionMask()

        # Define bounding boxes: (x_min, y_min, x_max, y_max) in pixel coords
        boxes = [(10, 20, 110, 220), (150, 30, 280, 240)]

        # Create masks at latent resolution
        masks = masker.create_latent_mask(boxes, latent_size=(32, 32))

        # Apply negative attention during the DiT forward pass
        attn_scores = masker.apply_negative_attention(attn_scores, masks[0])
    """

    # ------------------------------------------------------------------ #
    # Spatial mask creation
    # ------------------------------------------------------------------ #

    def create_spatial_mask(
        self,
        bounding_boxes: list[tuple[int, int, int, int]],
        image_size: tuple[int, int],
        leakage_alpha: float = 0.05,
    ) -> torch.Tensor:
        """Create soft spatial masks for *N* subjects at image resolution.

        Each mask is 1.0 inside the corresponding bounding box and
        *leakage_alpha* outside.  A small non-zero outside value prevents
        hard discontinuities and allows minimal cross-subject influence.

        Args:
            bounding_boxes: List of ``(x_min, y_min, x_max, y_max)`` tuples
                in pixel coordinates.  Coordinates are clamped to
                *image_size*.
            image_size: ``(H, W)`` of the target image.
            leakage_alpha: Value assigned to regions outside the bounding box.
                Must be in ``[0, 1)``.

        Returns:
            Tensor of shape ``(N, H, W)`` with values in
            ``[leakage_alpha, 1.0]``.
        """
        H, W = image_size
        N = len(bounding_boxes)

        masks = torch.full((N, H, W), leakage_alpha, dtype=torch.float32)

        for i, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
            # Clamp to image boundaries
            x_min = max(0, min(x_min, W))
            y_min = max(0, min(y_min, H))
            x_max = max(0, min(x_max, W))
            y_max = max(0, min(y_max, H))

            if x_max > x_min and y_max > y_min:
                masks[i, y_min:y_max, x_min:x_max] = 1.0

        return masks

    def create_latent_mask(
        self,
        bounding_boxes: list[tuple[int, int, int, int]],
        latent_size: tuple[int, int],
        leakage_alpha: float = 0.05,
        downscale_factor: int = 8,
    ) -> torch.Tensor:
        """Create soft spatial masks at latent resolution.

        Bounding boxes are specified in *pixel* coordinates and automatically
        scaled down by *downscale_factor* (default 8 for standard VAE).

        Args:
            bounding_boxes: ``(x_min, y_min, x_max, y_max)`` in pixel coords.
            latent_size: ``(H_latent, W_latent)``.
            leakage_alpha: Background mask value.
            downscale_factor: Ratio between pixel and latent resolution.

        Returns:
            Tensor of shape ``(N, H_latent, W_latent)``.
        """
        scaled_boxes: list[tuple[int, int, int, int]] = []
        for x_min, y_min, x_max, y_max in bounding_boxes:
            scaled_boxes.append((
                x_min // downscale_factor,
                y_min // downscale_factor,
                max(x_min // downscale_factor + 1, x_max // downscale_factor),
                max(y_min // downscale_factor + 1, y_max // downscale_factor),
            ))
        return self.create_spatial_mask(scaled_boxes, latent_size, leakage_alpha)

    # ------------------------------------------------------------------ #
    # Negative attention
    # ------------------------------------------------------------------ #

    def apply_negative_attention(
        self,
        attention_scores: torch.Tensor,
        mask: torch.Tensor,
        gamma: float = 3.0,
    ) -> torch.Tensor:
        """Apply negative attention to suppress cross-subject leakage.

        For each spatial position **outside** the subject's region, a penalty
        of ``-gamma`` is added to the attention logits (before softmax).  This
        discourages the subject's LoRA tokens from attending to positions that
        belong to other subjects.

        The formula is::

            adjusted = attention_scores - gamma * (1 - mask)

        where *mask* has value 1.0 inside the subject's region and a small
        positive value (leakage_alpha) outside.

        Args:
            attention_scores: Pre-softmax attention logits. Supported shapes:

                * ``(B, heads, seq_len, seq_len)`` -- standard multi-head
                  attention.
                * ``(B, seq_len, seq_len)`` -- single-head or merged heads.

            mask: Spatial mask of shape ``(H, W)`` or ``(seq_len,)``.  If 2-D
                the mask is flattened to ``(H*W,)`` and broadcast.
            gamma: Penalty strength.  Larger values produce harder masking.

        Returns:
            Adjusted attention logits (same shape as *attention_scores*).
        """
        # Flatten 2-D mask to 1-D sequence
        if mask.dim() == 2:
            mask = mask.reshape(-1)  # (seq_len,)

        seq_len = mask.shape[0]

        # Build penalty: gamma * (1 - mask), shape (seq_len,)
        penalty = gamma * (1.0 - mask.to(attention_scores.device, dtype=attention_scores.dtype))

        # Broadcast penalty to attention shape.  The penalty applies to the
        # *key* dimension (last axis): for each query position we penalise
        # attending to out-of-region keys.
        if attention_scores.dim() == 4:
            # (B, heads, Q, K) -- penalty on K dimension
            penalty = penalty.view(1, 1, 1, seq_len)
        elif attention_scores.dim() == 3:
            # (B, Q, K)
            penalty = penalty.view(1, 1, seq_len)
        else:
            raise ValueError(
                f"Expected attention_scores with 3 or 4 dimensions, got {attention_scores.dim()}."
            )

        return attention_scores - penalty

    # ------------------------------------------------------------------ #
    # Mask blending for overlapping regions
    # ------------------------------------------------------------------ #

    def blend_masks(
        self,
        masks: list[torch.Tensor],
        overlap_strategy: str = "distance",
    ) -> torch.Tensor:
        """Blend multiple subject masks, handling overlapping regions.

        Args:
            masks: List of *N* tensors each of shape ``(H, W)`` with values
                in ``[0, 1]``.
            overlap_strategy: Blending strategy.

                * ``"distance"`` -- weight each mask proportionally to the
                  distance from its bounding-box center (closer center wins).
                * ``"normalize"`` -- simply normalise so that per-pixel weights
                  sum to 1.

        Returns:
            Stacked, blended masks of shape ``(N, H, W)`` where per-pixel
            values sum to 1.
        """
        if not masks:
            raise ValueError("At least one mask is required.")

        stacked = torch.stack(masks, dim=0)  # (N, H, W)

        if overlap_strategy == "normalize":
            total = stacked.sum(dim=0, keepdim=True).clamp(min=1e-8)
            return stacked / total

        if overlap_strategy == "distance":
            return self._distance_blend(stacked)

        raise ValueError(
            f"Unknown overlap_strategy '{overlap_strategy}'. "
            "Choose 'distance' or 'normalize'."
        )

    # ------------------------------------------------------------------ #
    # Feathering
    # ------------------------------------------------------------------ #

    @staticmethod
    def feather_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Apply Gaussian feathering to mask edges for smooth transitions.

        The mask is convolved with a 2-D Gaussian kernel so that hard edges
        are softened.

        Args:
            mask: Tensor of shape ``(H, W)`` or ``(N, H, W)``.
            kernel_size: Size of the Gaussian kernel (must be odd).

        Returns:
            Feathered mask with the same shape.
        """
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")

        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # OpenCV convention

        # Build 1-D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()

        # Reshape kernel for F.conv2d: (out_channels=1, in_channels=1, kH, kW)
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0).to(mask.device, dtype=mask.dtype)

        added_batch = False
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)
            added_batch = True

        # Add channel dim for conv2d: (N, 1, H, W)
        x = mask.unsqueeze(1)
        padding = kernel_size // 2
        feathered = F.conv2d(x, kernel, padding=padding)
        feathered = feathered.squeeze(1)  # (N, H, W)

        if added_batch:
            feathered = feathered.squeeze(0)  # (H, W)

        return feathered

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _distance_blend(stacked: torch.Tensor) -> torch.Tensor:
        """Blend masks by inverse distance from each mask's center of mass.

        For each mask we compute its centre of mass (weighted mean of
        coordinates), then assign per-pixel blending weights proportional to
        ``1 / (distance_to_center + eps)``.

        Args:
            stacked: ``(N, H, W)`` mask tensor.

        Returns:
            ``(N, H, W)`` normalised blended masks.
        """
        N, H, W = stacked.shape
        device = stacked.device
        dtype = stacked.dtype

        # Coordinate grids: (H, W)
        ys = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
        xs = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)

        distance_weights = torch.zeros_like(stacked)

        for i in range(N):
            m = stacked[i]
            total_mass = m.sum().clamp(min=1e-8)
            cy = (m * ys).sum() / total_mass
            cx = (m * xs).sum() / total_mass

            dist = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2 + 1e-6)
            # Weight is proportional to mask value * inverse distance
            distance_weights[i] = m / dist

        # Normalise across subjects at each pixel
        total = distance_weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
        return distance_weights / total
