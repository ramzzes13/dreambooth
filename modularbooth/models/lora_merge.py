"""LoRA Composer: Load and compose multiple subject LoRA modules for multi-subject generation.

During multi-subject inference each subject is represented by its own LoRA
checkpoint (trained with :class:`BlockwiseLoRA`).  The ``LoRAComposer`` merges
these into the base DiT model, optionally weighting each LoRA's contribution
by a spatial mask so that different regions of the image are influenced by
different subjects.

Composition modes:

* **Single-subject** -- apply a single LoRA checkpoint uniformly across all
  spatial positions.
* **Multi-subject (spatially-masked)** -- for each linear layer, compute the
  LoRA delta as a spatially-weighted sum of per-subject deltas.  In attention
  layers this means modifying the attention computation to apply per-region
  LoRA weights: ``delta_W(pos) = sum_i mask_i(pos) * B_i @ A_i``.
"""

from __future__ import annotations

import copy
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatially-conditioned LoRA linear layer
# ---------------------------------------------------------------------------


class SpatialLoRALinear(nn.Module):
    """A linear layer augmented with multiple spatially-masked LoRA branches.

    For each spatial position in the input sequence, the output is::

        y(pos) = W @ x(pos) + sum_i mask_i(pos) * (B_i @ A_i) @ x(pos)

    where *i* iterates over subject LoRA modules and *mask_i(pos)* is the
    subject's spatial weight at that position.

    Args:
        original_linear: The frozen ``nn.Linear`` to augment.
        lora_branches: List of ``(A_i, B_i, scaling_i)`` tuples.
        spatial_masks: Tensor of shape ``(num_subjects, seq_len)`` giving
            per-position weights for each subject.  Must sum to ~1 at each
            position (see :meth:`TokenAwareAttentionMask.blend_masks`).
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        lora_branches: list[tuple[torch.Tensor, torch.Tensor, float]],
        spatial_masks: torch.Tensor,
    ) -> None:
        super().__init__()
        self.original_linear = original_linear
        self.lora_branches = lora_branches
        self.spatial_masks = spatial_masks  # (num_subjects, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spatially-weighted LoRA composition.

        Args:
            x: Input tensor.  Supported shapes:
                - ``(B, seq_len, d_in)`` -- sequence input
                - ``(B, d_in)`` -- non-sequence input (masks are ignored and
                  a uniform average is used instead)

        Returns:
            Output tensor of the same leading shape.
        """
        base_out = self.original_linear(x)

        if x.dim() == 2:
            # Non-sequence: uniform weighted combination
            lora_sum = torch.zeros_like(base_out)
            weight = 1.0 / max(len(self.lora_branches), 1)
            for A_i, B_i, scaling_i in self.lora_branches:
                A_i = A_i.to(x.device, dtype=x.dtype)
                B_i = B_i.to(x.device, dtype=x.dtype)
                lora_out = F.linear(F.linear(x, A_i), B_i)
                lora_sum = lora_sum + weight * scaling_i * lora_out
            return base_out + lora_sum

        # x: (B, seq_len, d_in)
        B_size, seq_len, d_in = x.shape
        lora_sum = torch.zeros_like(base_out)  # (B, seq_len, d_out)

        masks = self.spatial_masks.to(x.device, dtype=x.dtype)  # (num_subjects, seq_len)

        for i, (A_i, B_i, scaling_i) in enumerate(self.lora_branches):
            A_i = A_i.to(x.device, dtype=x.dtype)
            B_i = B_i.to(x.device, dtype=x.dtype)

            # LoRA output for subject i: (B, seq_len, d_out)
            lora_out = F.linear(F.linear(x, A_i), B_i)

            # Spatial weighting: mask_i has shape (seq_len,), broadcast over B and d_out
            if i < masks.shape[0]:
                mask_i = masks[i]
                # Handle case where mask seq_len does not match input seq_len
                if mask_i.shape[0] != seq_len:
                    mask_i = F.interpolate(
                        mask_i.unsqueeze(0).unsqueeze(0),
                        size=seq_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                mask_i = mask_i.view(1, seq_len, 1)  # broadcast
            else:
                mask_i = torch.ones(1, seq_len, 1, device=x.device, dtype=x.dtype)

            lora_sum = lora_sum + scaling_i * mask_i * lora_out

        return base_out + lora_sum


# ---------------------------------------------------------------------------
# Single-subject LoRA linear (no spatial masking)
# ---------------------------------------------------------------------------


class SingleLoRALinear(nn.Module):
    """Linear layer with a single LoRA adapter applied uniformly.

    Args:
        original_linear: Frozen base ``nn.Linear``.
        lora_A: LoRA down-projection, shape ``(rank, d_in)``.
        lora_B: LoRA up-projection, shape ``(d_out, rank)``.
        scaling: ``alpha / rank`` scaling factor.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.original_linear = original_linear
        self.register_buffer("lora_A", lora_A)
        self.register_buffer("lora_B", lora_B)
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out


# ---------------------------------------------------------------------------
# LoRA Composer
# ---------------------------------------------------------------------------


class LoRAComposer:
    """Load and compose multiple subject LoRA modules for multi-subject generation.

    Args:
        base_model: The DiT backbone (``nn.Module``).  The composer modifies
            this model in-place when applying LoRA adapters.
    """

    def __init__(self, base_model: nn.Module) -> None:
        self.base_model = base_model

        # Stash of original nn.Linear references so we can restore them later
        self._original_linears: dict[str, nn.Linear] = {}

        # Loaded subject LoRA checkpoints
        self._loaded_subjects: dict[str, dict[str, torch.Tensor]] = OrderedDict()

        self._is_applied = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_subject_lora(
        self,
        lora_path: str | Path,
        subject_id: str,
    ) -> dict[str, torch.Tensor]:
        """Load a LoRA checkpoint for a subject.

        Args:
            lora_path: Path to a safetensors file saved by
                :meth:`BlockwiseLoRA.save_lora`.
            subject_id: Unique identifier for this subject (e.g.
                ``"subject_dog"``, ``"subject_cat"``).

        Returns:
            Dictionary mapping parameter names to tensors (the raw checkpoint
            state dict).

        Raises:
            FileNotFoundError: If *lora_path* does not exist.
        """
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

        state = load_file(str(lora_path))
        self._loaded_subjects[subject_id] = state
        logger.info(
            "Loaded LoRA for subject '%s' from %s (%d tensors).",
            subject_id, lora_path, len(state),
        )
        return state

    def compose_loras(
        self,
        lora_modules: list[dict[str, torch.Tensor]],
        spatial_masks: torch.Tensor,
    ) -> None:
        """Apply a spatially-weighted combination of multiple LoRA modules.

        For each ``nn.Linear`` in the model that has LoRA parameters in the
        checkpoints, the layer is replaced with a :class:`SpatialLoRALinear`
        that computes a position-dependent weighted sum of all subject LoRA
        deltas.

        Args:
            lora_modules: List of LoRA state dicts (one per subject), as
                returned by :meth:`load_subject_lora`.
            spatial_masks: Tensor of shape ``(num_subjects, H_latent, W_latent)``
                (from :class:`TokenAwareAttentionMask`).  Masks are
                flattened to ``(num_subjects, seq_len)`` internally.

        Raises:
            RuntimeError: If LoRA modules are already applied.
        """
        if self._is_applied:
            raise RuntimeError(
                "LoRA modules are already applied.  Call clear_loras() first."
            )

        if len(lora_modules) == 0:
            logger.warning("No LoRA modules provided; nothing to compose.")
            return

        # Flatten spatial masks to (num_subjects, seq_len)
        if spatial_masks.dim() == 3:
            N, H, W = spatial_masks.shape
            flat_masks = spatial_masks.reshape(N, H * W)
        elif spatial_masks.dim() == 2:
            flat_masks = spatial_masks
        else:
            raise ValueError(
                f"spatial_masks must be 2-D or 3-D, got {spatial_masks.dim()}-D."
            )

        # Collect all module names that have LoRA params across any subject
        module_names = self._collect_lora_module_names(lora_modules)

        # Stash originals and perform replacements
        for module_name in module_names:
            original = self._get_module(self.base_model, module_name)
            if not isinstance(original, nn.Linear):
                logger.debug("Skipping '%s' -- not nn.Linear.", module_name)
                continue

            self._original_linears[module_name] = original

            # Build LoRA branches for each subject
            branches: list[tuple[torch.Tensor, torch.Tensor, float]] = []
            for subj_state in lora_modules:
                key_a = f"{module_name}.lora_A"
                key_b = f"{module_name}.lora_B"
                if key_a in subj_state and key_b in subj_state:
                    A_i = subj_state[key_a]
                    B_i = subj_state[key_b]
                    rank = A_i.shape[0]
                    scaling = 1.0 / rank  # default alpha = rank -> scaling = 1.0; conservative default
                    branches.append((A_i, B_i, scaling))

            if not branches:
                continue

            spatial_linear = SpatialLoRALinear(
                original_linear=original,
                lora_branches=branches,
                spatial_masks=flat_masks,
            )

            parent_name, attr_name = self._split_name(module_name)
            parent = self._get_module(self.base_model, parent_name)
            setattr(parent, attr_name, spatial_linear)

        self._is_applied = True
        logger.info(
            "Composed %d subject LoRAs across %d layers with spatial masking.",
            len(lora_modules), len(module_names),
        )

    def apply_single_lora(
        self,
        lora_weights: dict[str, torch.Tensor],
        scaling: float | None = None,
    ) -> None:
        """Apply a single LoRA for single-subject inference.

        This is a simpler path than :meth:`compose_loras` -- no spatial masks
        are involved.

        Args:
            lora_weights: State dict from :meth:`load_subject_lora`.
            scaling: Override for the LoRA scaling factor.  If ``None``,
                ``alpha / rank`` is used with ``alpha = rank`` (i.e. scaling
                = 1.0).

        Raises:
            RuntimeError: If LoRA modules are already applied.
        """
        if self._is_applied:
            raise RuntimeError(
                "LoRA modules are already applied.  Call clear_loras() first."
            )

        module_names = set()
        for key in lora_weights:
            if key.endswith(".lora_A"):
                name = key[: -len(".lora_A")]
                module_names.add(name)

        for module_name in module_names:
            key_a = f"{module_name}.lora_A"
            key_b = f"{module_name}.lora_B"
            if key_a not in lora_weights or key_b not in lora_weights:
                continue

            original = self._get_module(self.base_model, module_name)
            if not isinstance(original, nn.Linear):
                continue

            self._original_linears[module_name] = original

            A = lora_weights[key_a]
            B = lora_weights[key_b]
            rank = A.shape[0]
            s = scaling if scaling is not None else 1.0

            lora_linear = SingleLoRALinear(
                original_linear=original,
                lora_A=A,
                lora_B=B,
                scaling=s,
            )

            parent_name, attr_name = self._split_name(module_name)
            parent = self._get_module(self.base_model, parent_name)
            setattr(parent, attr_name, lora_linear)

        self._is_applied = True
        logger.info(
            "Applied single-subject LoRA across %d layers.",
            len(module_names),
        )

    def clear_loras(self) -> None:
        """Remove all applied LoRA modifications and restore original layers.

        After calling this method the model is in its original state and new
        LoRA modules can be applied.
        """
        if not self._is_applied:
            logger.debug("No LoRA modules to clear.")
            return

        for module_name, original_linear in self._original_linears.items():
            parent_name, attr_name = self._split_name(module_name)
            parent = self._get_module(self.base_model, parent_name)
            setattr(parent, attr_name, original_linear)

        self._original_linears.clear()
        self._is_applied = False
        logger.info("Cleared all LoRA modifications from the model.")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_lora_module_names(
        lora_modules: list[dict[str, torch.Tensor]],
    ) -> list[str]:
        """Collect unique module names that have LoRA params in any checkpoint."""
        names: set[str] = set()
        for state in lora_modules:
            for key in state:
                if key.endswith(".lora_A"):
                    name = key[: -len(".lora_A")]
                    names.add(name)
        return sorted(names)

    @staticmethod
    def _split_name(full_name: str) -> tuple[str, str]:
        """Split ``'a.b.c'`` into ``('a.b', 'c')``."""
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    @staticmethod
    def _get_module(model: nn.Module, name: str) -> nn.Module:
        """Retrieve a sub-module by its dotted name."""
        if name == "":
            return model
        tokens = name.split(".")
        mod = model
        for tok in tokens:
            mod = getattr(mod, tok)
        return mod
