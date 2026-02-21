"""Blockwise LoRA: Adaptive-rank Low-Rank Adaptation for Diffusion Transformer blocks.

This module applies LoRA with per-block rank customization to a DiT backbone.
Each transformer block is classified as identity-encoding, context-encoding, or
shared, and receives a LoRA adapter with an appropriate rank:

    - Identity blocks: higher rank to capture fine-grained subject appearance.
    - Context blocks: lower rank to preserve prompt-following ability.
    - Shared blocks: intermediate rank for general capacity.

The implementation uses standard low-rank decomposition: for a weight matrix W of
shape (d_out, d_in), the update is delta_W = B @ A where B has shape (d_out, rank)
and A has shape (rank, d_in). A is initialized via Kaiming uniform and B is
initialized to zeros so that the adapter starts as an identity function.
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-rank linear layer
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """A low-rank adapter that wraps an existing ``nn.Linear`` layer.

    The forward pass computes::

        y = W_original @ x + (alpha / rank) * B @ A @ x + bias

    where *A* is (rank, d_in), *B* is (d_out, rank), *alpha* controls the
    overall scaling, and dropout is applied to the input of the LoRA branch.

    Args:
        original_linear: The frozen ``nn.Linear`` to augment.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor (the effective scale is ``alpha / rank``).
        dropout: Dropout probability applied before the LoRA branch.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_out, d_in = original_linear.weight.shape

        # A: (rank, d_in) — Kaiming uniform initialisation
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (d_out, rank) — zero initialisation
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Freeze the original weight and bias
        self.original_linear.weight.requires_grad_(False)
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad_(False)

    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original linear and LoRA branch."""
        base_out = self.original_linear(x)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    # --------------------------------------------------------------------- #

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the original linear and return it.

        After merging the original linear contains the combined weight and
        can be used without the LoRA wrapper.
        """
        with torch.no_grad():
            self.original_linear.weight.add_(
                self.scaling * (self.lora_B @ self.lora_A)
            )
        return self.original_linear


# ---------------------------------------------------------------------------
# Blockwise LoRA manager
# ---------------------------------------------------------------------------


class BlockwiseLoRA:
    """Apply LoRA with adaptive rank to different DiT transformer blocks.

    Args:
        model: The DiT backbone (``nn.Module``).
        block_config: Mapping from block index (``int``) to block role
            (``"identity"`` | ``"context"`` | ``"shared"``).
        identity_rank: LoRA rank for identity blocks.
        context_rank: LoRA rank for context blocks.
        shared_rank: LoRA rank for shared blocks.
        alpha_ratio: Multiplier applied to rank to compute the LoRA alpha
            (``alpha = rank * alpha_ratio``).
        dropout: Dropout rate for the LoRA branch.
        target_modules: List of module-name patterns (regex) that select which
            ``nn.Linear`` layers receive a LoRA adapter. For example
            ``["attn.qkv", "attn.proj", "mlp.fc"]``.
    """

    VALID_ROLES = {"identity", "context", "shared"}

    def __init__(
        self,
        model: nn.Module,
        block_config: dict[int, str],
        identity_rank: int = 16,
        context_rank: int = 4,
        shared_rank: int = 8,
        alpha_ratio: float = 1.0,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        self.model = model
        self.block_config = block_config
        self.identity_rank = identity_rank
        self.context_rank = context_rank
        self.shared_rank = shared_rank
        self.alpha_ratio = alpha_ratio
        self.dropout = dropout
        self.target_modules = target_modules or [
            r"attn\.qkv",
            r"attn\.proj",
            r"mlp\.fc1",
            r"mlp\.fc2",
        ]

        # Validate block_config values
        for idx, role in block_config.items():
            if role not in self.VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{role}' for block {idx}. "
                    f"Must be one of {self.VALID_ROLES}."
                )

        # Registry of injected LoRA modules keyed by their full parameter path
        self._lora_modules: dict[str, LoRALinear] = OrderedDict()
        self._applied = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply_lora(self) -> None:
        """Inject LoRA adapters into the model.

        Walks all named modules and, for each ``nn.Linear`` whose name matches
        one of the *target_modules* patterns **and** resides in a block present
        in *block_config*, replaces it with a ``LoRALinear`` wrapper.

        Raises:
            RuntimeError: If LoRA has already been applied.
        """
        if self._applied:
            raise RuntimeError("LoRA adapters have already been applied to this model.")

        replacements: list[tuple[nn.Module, str, LoRALinear]] = []

        for full_name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            # Check if module name matches any target pattern
            if not self._matches_target(full_name):
                continue

            # Determine block role
            role = self._classify_block(full_name)
            if role is None:
                # Module is not inside any configured block — skip
                continue

            rank = self._rank_for_role(role)
            alpha = rank * self.alpha_ratio

            lora_linear = LoRALinear(
                original_linear=module,
                rank=rank,
                alpha=alpha,
                dropout=self.dropout,
            )

            # Schedule replacement (we must not mutate the tree while iterating)
            parent_name, attr_name = self._split_name(full_name)
            parent_module = self._get_module(self.model, parent_name)
            replacements.append((parent_module, attr_name, lora_linear))
            self._lora_modules[full_name] = lora_linear

        # Perform replacements
        for parent, attr, lora_mod in replacements:
            setattr(parent, attr, lora_mod)

        self._applied = True
        counts = self.get_parameter_count()
        total = sum(counts.values())
        logger.info(
            "Applied BlockwiseLoRA: %d adapters, %d trainable parameters "
            "(identity=%d, context=%d, shared=%d).",
            len(self._lora_modules),
            total,
            counts.get("identity", 0),
            counts.get("context", 0),
            counts.get("shared", 0),
        )

    def get_lora_params(self) -> dict[str, nn.Parameter]:
        """Return only the LoRA parameters (suitable for an optimizer group).

        Returns:
            Dictionary mapping ``"<module_path>.lora_A"`` / ``"…lora_B"`` to
            their ``nn.Parameter`` objects.
        """
        params: dict[str, nn.Parameter] = {}
        for name, lora_mod in self._lora_modules.items():
            params[f"{name}.lora_A"] = lora_mod.lora_A
            params[f"{name}.lora_B"] = lora_mod.lora_B
        return params

    def save_lora(self, path: str | Path) -> None:
        """Save LoRA weights to a safetensors file.

        Args:
            path: Destination file path (should end in ``.safetensors``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, torch.Tensor] = {}
        for name, lora_mod in self._lora_modules.items():
            state[f"{name}.lora_A"] = lora_mod.lora_A.data
            state[f"{name}.lora_B"] = lora_mod.lora_B.data
        save_file(state, str(path))
        logger.info("Saved LoRA weights to %s (%d tensors).", path, len(state))

    def load_lora(self, path: str | Path) -> None:
        """Load LoRA weights from a safetensors file.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If the checkpoint is missing expected keys.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

        state = load_file(str(path))
        loaded = 0
        for name, lora_mod in self._lora_modules.items():
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a not in state or key_b not in state:
                raise KeyError(
                    f"Checkpoint is missing keys for module '{name}'. "
                    f"Expected '{key_a}' and '{key_b}'."
                )
            lora_mod.lora_A.data.copy_(state[key_a])
            lora_mod.lora_B.data.copy_(state[key_b])
            loaded += 1

        logger.info("Loaded LoRA weights from %s (%d modules).", path, loaded)

    def get_parameter_count(self) -> dict[str, int]:
        """Return the number of trainable LoRA parameters per block role.

        Returns:
            Dictionary with keys ``"identity"``, ``"context"``, ``"shared"``
            mapped to their respective trainable parameter counts.
        """
        counts: dict[str, int] = {"identity": 0, "context": 0, "shared": 0}
        for name, lora_mod in self._lora_modules.items():
            role = self._classify_block(name)
            if role is None:
                continue
            n_params = lora_mod.lora_A.numel() + lora_mod.lora_B.numel()
            counts[role] += n_params
        return counts

    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights into the base model and remove adapters.

        After calling this method the model contains the combined weights and
        no longer has any ``LoRALinear`` wrappers.

        Returns:
            The modified base model with merged weights.
        """
        for full_name, lora_mod in self._lora_modules.items():
            merged_linear = lora_mod.merge()
            parent_name, attr_name = self._split_name(full_name)
            parent_module = self._get_module(self.model, parent_name)
            setattr(parent_module, attr_name, merged_linear)

        self._lora_modules.clear()
        self._applied = False
        logger.info("Merged LoRA weights into base model and removed adapters.")
        return self.model

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _classify_block(self, module_name: str) -> str | None:
        """Determine whether *module_name* belongs to an identity, context, or shared block.

        The function searches *module_name* for a numeric block index pattern
        (e.g. ``blocks.12.attn.proj``, ``transformer_blocks.3.ff``) and looks
        up the index in *block_config*.

        Args:
            module_name: Fully-qualified dotted module path.

        Returns:
            ``"identity"``, ``"context"``, ``"shared"``, or ``None`` if the
            module does not reside in a configured block.
        """
        # Common DiT naming conventions: "blocks.N.", "transformer_blocks.N.",
        # "joint_blocks.N.", "single_blocks.N."
        match = re.search(r"(?:blocks|transformer_blocks|joint_blocks|single_blocks)\.(\d+)\.", module_name)
        if match is None:
            return None
        block_idx = int(match.group(1))
        return self.block_config.get(block_idx)

    def _matches_target(self, module_name: str) -> bool:
        """Return True if *module_name* matches any pattern in *target_modules*."""
        return any(re.search(pat, module_name) for pat in self.target_modules)

    def _rank_for_role(self, role: str) -> int:
        """Return the LoRA rank for the given block role."""
        return {
            "identity": self.identity_rank,
            "context": self.context_rank,
            "shared": self.shared_rank,
        }[role]

    @staticmethod
    def _split_name(full_name: str) -> tuple[str, str]:
        """Split ``'a.b.c'`` into ``('a.b', 'c')``."""
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    @staticmethod
    def _get_module(model: nn.Module, name: str) -> nn.Module:
        """Retrieve a sub-module by its dotted name (empty string returns *model*)."""
        if name == "":
            return model
        tokens = name.split(".")
        mod = model
        for tok in tokens:
            mod = getattr(mod, tok)
        return mod
