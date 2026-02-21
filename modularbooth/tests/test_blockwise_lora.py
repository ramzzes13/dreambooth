"""Tests for modularbooth.models.blockwise_lora.

All tests run on CPU without GPU or large model downloads.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from modularbooth.models.blockwise_lora import BlockwiseLoRA, LoRALinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDiTBlock(nn.Module):
    """Minimal transformer block stub with linear layers whose names
    match the default target_modules patterns used by BlockwiseLoRA."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        # nn.Module.__setattr__ automatically registers child modules.
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)  # matches "attn\.qkv"
        self.attn.proj = nn.Linear(dim * 3, dim)  # matches "attn\.proj"
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)  # matches "mlp\.fc1"
        self.mlp.fc2 = nn.Linear(dim * 4, dim)  # matches "mlp\.fc2"


class _FakeDiT(nn.Module):
    """Minimal DiT-like model with numbered transformer blocks."""

    def __init__(self, n_blocks: int = 2, dim: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeDiTBlock(dim) for _ in range(n_blocks)])


def _make_simple_model_with_to_q() -> nn.Module:
    """Model with `blocks.0.to_q` and `blocks.1.to_q` linear layers,
    using a target pattern that matches `to_q`."""

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(16, 32)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block(), _Block()])

    return _Model()


# ---------------------------------------------------------------------------
# Tests for LoRALinear
# ---------------------------------------------------------------------------


class TestLoRALinearForward:
    """test_lora_linear_forward: output shape and parameter count."""

    def test_output_shape(self) -> None:
        base_linear = nn.Linear(16, 32)
        rank = 4
        lora = LoRALinear(base_linear, rank=rank, alpha=float(rank))

        batch = 5
        x = torch.randn(batch, 16)
        y = lora(x)

        assert y.shape == (batch, 32), f"Expected (5, 32), got {y.shape}"

    def test_parameter_count(self) -> None:
        d_in, d_out, rank = 16, 32, 4
        base_linear = nn.Linear(d_in, d_out)
        lora = LoRALinear(base_linear, rank=rank, alpha=float(rank))

        # LoRA params: A is (rank, d_in) + B is (d_out, rank)
        expected = rank * (d_in + d_out)
        actual = lora.lora_A.numel() + lora.lora_B.numel()
        assert actual == expected, f"Expected {expected} LoRA params, got {actual}"


class TestLoRALinearMerge:
    """test_lora_linear_merge: merged linear produces same output."""

    def test_merge_produces_same_output(self) -> None:
        torch.manual_seed(42)
        base_linear = nn.Linear(16, 32)
        rank = 4
        lora = LoRALinear(base_linear, rank=rank, alpha=float(rank))

        # Give LoRA non-trivial B weights so the adapter actually does something
        nn.init.normal_(lora.lora_B, std=0.1)

        x = torch.randn(3, 16)
        output_before = lora(x).detach().clone()

        merged = lora.merge()
        output_after = merged(x).detach()

        assert torch.allclose(output_before, output_after, atol=1e-5), (
            "Merged linear output differs from LoRA wrapper output."
        )


# ---------------------------------------------------------------------------
# Tests for BlockwiseLoRA
# ---------------------------------------------------------------------------


class TestBlockwiseLoRAApply:
    """test_blockwise_lora_apply: identity block gets higher rank than context."""

    def test_identity_higher_rank_than_context(self) -> None:
        model = _make_simple_model_with_to_q()

        block_config = {0: "identity", 1: "context"}
        identity_rank = 16
        context_rank = 4

        bw = BlockwiseLoRA(
            model=model,
            block_config=block_config,
            identity_rank=identity_rank,
            context_rank=context_rank,
            target_modules=[r"to_q"],
        )
        bw.apply_lora()

        # After apply, blocks.0.to_q should be a LoRALinear with rank=16
        lora_0 = model.blocks[0].to_q
        assert isinstance(lora_0, LoRALinear), "Block 0's to_q was not replaced with LoRALinear"
        assert lora_0.rank == identity_rank, (
            f"Expected identity rank {identity_rank}, got {lora_0.rank}"
        )

        # blocks.1.to_q should be LoRALinear with rank=4
        lora_1 = model.blocks[1].to_q
        assert isinstance(lora_1, LoRALinear), "Block 1's to_q was not replaced with LoRALinear"
        assert lora_1.rank == context_rank, (
            f"Expected context rank {context_rank}, got {lora_1.rank}"
        )

        # Identity rank should be strictly higher than context rank
        assert lora_0.rank > lora_1.rank


class TestBlockwiseLoRASaveLoad:
    """test_blockwise_lora_save_load: save/load preserves parameters."""

    def test_save_and_load_roundtrip(self) -> None:
        torch.manual_seed(99)
        model = _FakeDiT(n_blocks=2, dim=32)
        block_config = {0: "identity", 1: "context"}

        bw = BlockwiseLoRA(
            model=model,
            block_config=block_config,
            identity_rank=8,
            context_rank=4,
        )
        bw.apply_lora()

        # Randomise LoRA params so they are non-trivial
        for name, param in bw.get_lora_params().items():
            param.data.normal_(0, 0.5)

        # Snapshot original params
        original_params = {
            name: param.data.clone() for name, param in bw.get_lora_params().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lora.safetensors"
            bw.save_lora(path)

            # Zero out params to simulate fresh state
            for param in bw.get_lora_params().values():
                param.data.zero_()

            bw.load_lora(path)

        # Verify loaded params match originals
        for name, param in bw.get_lora_params().items():
            assert torch.allclose(param.data, original_params[name], atol=1e-6), (
                f"Parameter '{name}' does not match after load."
            )


class TestBlockwiseLoRAParameterCount:
    """test_blockwise_lora_parameter_count: per-role counts are correct."""

    def test_parameter_counts_per_role(self) -> None:
        dim = 32
        model = _FakeDiT(n_blocks=2, dim=dim)
        block_config = {0: "identity", 1: "context"}

        identity_rank = 8
        context_rank = 4

        bw = BlockwiseLoRA(
            model=model,
            block_config=block_config,
            identity_rank=identity_rank,
            context_rank=context_rank,
        )
        bw.apply_lora()

        counts = bw.get_parameter_count()

        # Block 0 (identity, rank=8) has 4 target linears:
        #   attn.qkv: (32, 96) -> A(8,32)+B(96,8) = 256+768 = 1024
        #   attn.proj: (96, 32) -> A(8,96)+B(32,8) = 768+256 = 1024
        #   mlp.fc1: (32, 128) -> A(8,32)+B(128,8) = 256+1024 = 1280
        #   mlp.fc2: (128, 32) -> A(8,128)+B(32,8) = 1024+256 = 1280
        # Total identity = 1024+1024+1280+1280 = 4608
        expected_identity = 0
        for linear_shape in [(32, 96), (96, 32), (32, 128), (128, 32)]:
            d_in, d_out = linear_shape[0], linear_shape[1]
            # weight shape is (d_out, d_in), A is (rank, d_in), B is (d_out, rank)
            expected_identity += identity_rank * d_in + d_out * identity_rank

        # Block 1 (context, rank=4): same linear shapes
        expected_context = 0
        for linear_shape in [(32, 96), (96, 32), (32, 128), (128, 32)]:
            d_in, d_out = linear_shape[0], linear_shape[1]
            expected_context += context_rank * d_in + d_out * context_rank

        assert counts["identity"] == expected_identity, (
            f"Expected identity params={expected_identity}, got {counts['identity']}"
        )
        assert counts["context"] == expected_context, (
            f"Expected context params={expected_context}, got {counts['context']}"
        )
        # Shared should be 0 since no block was assigned "shared"
        assert counts["shared"] == 0

        # Sanity: identity count > context count (higher rank means more params)
        assert counts["identity"] > counts["context"]
