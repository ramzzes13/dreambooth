"""Contrastive Context Disentanglement (CCD) loss for subject-background separation.

This module implements the CCD loss, which operates on intermediate DiT features
to encourage the model to disentangle subject identity from background context.
The loss uses an InfoNCE-style contrastive objective: subject features should be
pulled closer to features of the same subject in different contexts (positives)
and pushed apart from features of different subjects or background regions
(negatives).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCDLoss(nn.Module):
    """Contrastive Context Disentanglement loss.

    Uses InfoNCE contrastive learning on intermediate Diffusion Transformer
    features to separate subject identity representations from background
    context. This encourages the model to learn identity-preserving features
    that are invariant to changes in surrounding context.

    Args:
        temperature: Temperature scaling factor for the contrastive logits.
            Lower values sharpen the distribution, making the loss more
            sensitive to small similarity differences. Defaults to 0.07.
        feature_dim: If provided, a linear projection head is created to
            map input features from their native dimensionality to
            ``feature_dim`` before computing similarities. If ``None``,
            features are used as-is.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {temperature}"
            )
        self.temperature = temperature
        self.feature_dim = feature_dim

        # Optional linear projection head — created lazily if feature_dim is
        # set but the input dimensionality is not yet known.  When feature_dim
        # is None we skip projection entirely.
        self.projection: Optional[nn.Linear] = None
        self._projection_initialized = feature_dim is None  # True means "no projection needed"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_build_projection(self, input_dim: int) -> None:
        """Lazily initialise the projection head on first forward pass.

        This avoids requiring the caller to know the input feature
        dimensionality at construction time.
        """
        if self._projection_initialized:
            return
        assert self.feature_dim is not None
        self.projection = nn.Linear(input_dim, self.feature_dim, bias=False)
        # Place on same device / dtype as the rest of the module (if any
        # parameters already exist this is a no-op for device).
        self._projection_initialized = True

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        """Optionally project features through the linear head."""
        if self.projection is not None:
            return self.projection(features)
        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_subject_features(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Extract subject-region features via masked average pooling.

        Args:
            features: Intermediate DiT feature maps of shape ``(B, C, H, W)``.
            masks: Binary subject masks of shape ``(B, 1, H, W)``.  The masks
                are interpolated to the spatial resolution of *features* if
                their sizes differ.

        Returns:
            Tensor of shape ``(B, C)`` containing the mean feature vector
            within the masked region for each sample in the batch.
        """
        if features.ndim != 4:
            raise ValueError(
                f"features must be 4-D (B, C, H, W), got shape {features.shape}"
            )
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(
                f"masks must be 4-D (B, 1, H, W), got shape {masks.shape}"
            )

        # Resize masks to match feature spatial dims if necessary.
        if masks.shape[2:] != features.shape[2:]:
            masks = F.interpolate(
                masks.float(),
                size=features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Binarise after interpolation to keep a hard mask.
        masks = (masks > 0.5).float()

        # Masked average pooling: (B, C, H, W) * (B, 1, H, W) -> (B, C)
        masked_features = features * masks
        # Sum over spatial dims.
        pooled = masked_features.sum(dim=(2, 3))  # (B, C)
        # Number of active spatial positions per sample.
        area = masks.sum(dim=(2, 3)).clamp(min=1.0)  # (B, 1)
        pooled = pooled / area  # (B, C)

        return pooled

    def forward(
        self,
        subject_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the InfoNCE contrastive loss.

        Args:
            subject_features: Anchor features pooled over the subject region
                from the original image.  Shape ``(B, D)``.
            positive_features: Features from the same subject in a different
                context (e.g. augmented view).  Shape ``(B, D)``.
            negative_features: Features from different subjects or background
                regions.  Shape ``(B, N_neg, D)``.

        Returns:
            Scalar loss averaged over the batch.
        """
        if subject_features.ndim != 2:
            raise ValueError(
                f"subject_features must be 2-D (B, D), got {subject_features.shape}"
            )
        if positive_features.ndim != 2:
            raise ValueError(
                f"positive_features must be 2-D (B, D), got {positive_features.shape}"
            )
        if negative_features.ndim != 3:
            raise ValueError(
                f"negative_features must be 3-D (B, N_neg, D), got {negative_features.shape}"
            )

        # Lazily build projection if needed.
        self._maybe_build_projection(subject_features.shape[-1])

        # Project all features through the (optional) projection head.
        subject_features = self._project(subject_features)
        positive_features = self._project(positive_features)
        # negative_features is (B, N_neg, D) — reshape, project, reshape back.
        batch_size, n_neg, dim = negative_features.shape
        negative_features = self._project(
            negative_features.reshape(batch_size * n_neg, dim)
        ).reshape(batch_size, n_neg, -1)

        # L2-normalise for cosine similarity.
        subject_features = F.normalize(subject_features, dim=-1)
        positive_features = F.normalize(positive_features, dim=-1)
        negative_features = F.normalize(negative_features, dim=-1)

        # Positive similarity: (B,)
        sim_pos = (subject_features * positive_features).sum(dim=-1) / self.temperature

        # Negative similarities: (B, N_neg)
        # einsum: for each batch element, dot subject (D,) with each negative (N_neg, D).
        sim_neg = torch.einsum("bd,bnd->bn", subject_features, negative_features) / self.temperature

        # InfoNCE: -log( exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))) )
        # Numerically stable implementation via log-sum-exp.
        # Concatenate pos and neg logits -> (B, 1 + N_neg), then log_softmax over dim=-1.
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # (B, 1 + N_neg)
        # The positive is at index 0 for every sample.
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss
