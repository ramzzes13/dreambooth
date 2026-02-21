"""Knowledge Probe: Per-block probing for DiT block classification.

Given a DiT model and a personalisation dataset, this module trains a
single-block LoRA adapter on each transformer block independently and measures
two signals:

* **DINO score** (subject fidelity) -- cosine similarity between DINOv2
  embeddings of generated images and reference subject images.
* **CLIP-T score** (prompt fidelity) -- cosine similarity between CLIP text
  embeddings of the prompt and CLIP image embeddings of the generated images.

Based on the per-block profile, blocks are classified as:

* **identity** -- high DINO score *and* preserved CLIP-T score (the block is
  effective at encoding subject identity without hurting text alignment).
* **context** -- low DINO impact *and* high CLIP-T (the block mainly controls
  text-conditioned context/style).
* **shared** -- everything else (intermediate or mixed contribution).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class KnowledgeProbe:
    """Probe each DiT block to classify it as identity / context / shared.

    The probe trains a lightweight single-block LoRA adapter for each block,
    generates a small set of images, and computes DINO and CLIP-T scores to
    determine the block's role in subject-identity vs. context encoding.

    Args:
        model: The DiT backbone (``nn.Module``). Must support a standard
            forward pass that accepts latent noise, timesteps, and text
            embeddings.
        dataset: A ``torch.utils.data.Dataset`` yielding dicts with at least
            ``"pixel_values"`` and ``"input_ids"`` (or ``"prompt"`` for
            text-based pipelines). This is the personalisation dataset for one
            subject.
        dino_model_name: HuggingFace model identifier for the DINOv2 feature
            extractor (e.g. ``"facebook/dinov2-base"``).
        clip_model_name: HuggingFace model identifier for the CLIP model
            (e.g. ``"openai/clip-vit-large-patch14"``).
        device: Torch device for computation.
        dtype: Torch dtype for model inference.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        dino_model_name: str = "facebook/dinov2-base",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.dino_model_name = dino_model_name
        self.clip_model_name = clip_model_name
        self.device = torch.device(device)
        self.dtype = dtype

        # Lazily-loaded scoring models
        self._dino_model: nn.Module | None = None
        self._dino_processor: Any | None = None
        self._clip_model: nn.Module | None = None
        self._clip_processor: Any | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def probe_block(
        self,
        block_idx: int,
        rank: int = 8,
        num_steps: int = 200,
        lr: float = 1e-4,
        batch_size: int = 1,
        num_inference_samples: int = 4,
    ) -> dict[str, float]:
        """Train a single-block LoRA and measure DINO / CLIP-T scores.

        The procedure is:

        1. Inject a LoRA adapter **only** into block *block_idx*.
        2. Fine-tune for *num_steps* gradient steps on the personalisation
           dataset using a diffusion-denoising loss.
        3. Generate *num_inference_samples* images with the adapted model.
        4. Compute DINO (subject fidelity) and CLIP-T (prompt fidelity) scores.
        5. Remove the adapter and restore the original weights.

        Args:
            block_idx: Index of the transformer block to probe.
            rank: LoRA rank used for the probe adapter.
            num_steps: Number of fine-tuning gradient steps.
            lr: Learning rate for the probe adapter.
            batch_size: Training batch size.
            num_inference_samples: Number of images to generate for scoring.

        Returns:
            Dictionary with keys ``"block_idx"``, ``"dino_score"``,
            ``"clip_t_score"``, ``"rank"``, ``"num_steps"``.
        """
        logger.info("Probing block %d (rank=%d, steps=%d)...", block_idx, rank, num_steps)

        # ----- 1. Inject single-block LoRA -----
        block_module = self._get_block(block_idx)
        original_state = {k: v.clone() for k, v in block_module.state_dict().items()}
        lora_params = self._inject_block_lora(block_module, rank=rank)

        # ----- 2. Fine-tune -----
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        data_iter = iter(dataloader)

        self.model.train()
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            loss = self._training_step(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                logger.debug("  block %d  step %d/%d  loss=%.4f", block_idx, step + 1, num_steps, loss.item())

        # ----- 3. Generate images -----
        self.model.eval()
        generated_images = self._generate_samples(num_samples=num_inference_samples)

        # ----- 4. Score -----
        dino_score = self._compute_dino_score(generated_images)
        clip_t_score = self._compute_clip_t_score(generated_images)

        # ----- 5. Restore original weights -----
        block_module.load_state_dict(original_state, strict=False)
        self._remove_block_lora(block_module)

        result = {
            "block_idx": block_idx,
            "dino_score": float(dino_score),
            "clip_t_score": float(clip_t_score),
            "rank": rank,
            "num_steps": num_steps,
        }
        logger.info(
            "Block %d: DINO=%.4f  CLIP-T=%.4f",
            block_idx, dino_score, clip_t_score,
        )
        return result

    def probe_all_blocks(
        self,
        rank: int = 8,
        num_steps: int = 200,
        **kwargs: Any,
    ) -> dict[int, dict[str, float]]:
        """Probe every transformer block and return per-block scores.

        Args:
            rank: LoRA rank for each probe.
            num_steps: Fine-tuning steps per block.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`probe_block`.

        Returns:
            Dictionary mapping block index to its probe result dict.
        """
        num_blocks = self._count_blocks()
        logger.info("Probing all %d transformer blocks...", num_blocks)

        results: dict[int, dict[str, float]] = {}
        for idx in range(num_blocks):
            results[idx] = self.probe_block(idx, rank=rank, num_steps=num_steps, **kwargs)
        return results

    @staticmethod
    def classify_blocks(
        probe_results: dict[int, dict[str, float]],
        identity_threshold: float = 0.6,
        context_threshold: float = 0.3,
    ) -> dict[int, str]:
        """Classify blocks based on their DINO / CLIP-T profile.

        Classification rules:

        * **identity** -- ``dino_score >= identity_threshold`` **and**
          ``clip_t_score >= context_threshold`` (high subject fidelity while
          preserving prompt fidelity).
        * **context** -- ``dino_score < context_threshold`` **and**
          ``clip_t_score >= identity_threshold`` (low subject impact, high
          text alignment -- primarily a context/style block).
        * **shared** -- everything else.

        Args:
            probe_results: Output of :meth:`probe_all_blocks`.
            identity_threshold: Minimum DINO score to qualify as identity.
            context_threshold: Threshold for DINO (below) and CLIP-T (above)
                to qualify as context.

        Returns:
            Dictionary mapping block index to role string.
        """
        classification: dict[int, str] = {}
        for idx, scores in probe_results.items():
            dino = scores["dino_score"]
            clip_t = scores["clip_t_score"]

            if dino >= identity_threshold and clip_t >= context_threshold:
                classification[idx] = "identity"
            elif dino < context_threshold and clip_t >= identity_threshold:
                classification[idx] = "context"
            else:
                classification[idx] = "shared"

        # Log summary
        role_counts = {"identity": 0, "context": 0, "shared": 0}
        for role in classification.values():
            role_counts[role] += 1
        logger.info(
            "Block classification: identity=%d, context=%d, shared=%d",
            role_counts["identity"],
            role_counts["context"],
            role_counts["shared"],
        )
        return classification

    @staticmethod
    def save_probe_results(
        probe_results: dict[int, dict[str, float]],
        path: str | Path,
    ) -> None:
        """Persist probe results to a JSON file.

        Args:
            probe_results: Output of :meth:`probe_all_blocks`.
            path: Destination path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # JSON requires string keys
        serialisable = {str(k): v for k, v in probe_results.items()}
        with open(path, "w") as fh:
            json.dump(serialisable, fh, indent=2)
        logger.info("Saved probe results to %s.", path)

    @staticmethod
    def load_probe_results(path: str | Path) -> dict[int, dict[str, float]]:
        """Load probe results from a JSON file.

        Args:
            path: Source path.

        Returns:
            Probe results dictionary with integer block indices as keys.
        """
        path = Path(path)
        with open(path) as fh:
            raw = json.load(fh)
        return {int(k): v for k, v in raw.items()}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_block(self, block_idx: int) -> nn.Module:
        """Return the transformer block module at *block_idx*.

        Searches for common DiT block container names.
        """
        for attr_name in (
            "blocks",
            "transformer_blocks",
            "joint_blocks",
            "single_blocks",
        ):
            blocks = getattr(self.model, attr_name, None)
            if blocks is not None and isinstance(blocks, (nn.ModuleList, nn.Sequential, list)):
                if block_idx < len(blocks):
                    return blocks[block_idx]
        raise ValueError(
            f"Could not find transformer block {block_idx} in the model. "
            "Checked attributes: blocks, transformer_blocks, joint_blocks, single_blocks."
        )

    def _count_blocks(self) -> int:
        """Return the total number of transformer blocks in the model."""
        for attr_name in (
            "blocks",
            "transformer_blocks",
            "joint_blocks",
            "single_blocks",
        ):
            blocks = getattr(self.model, attr_name, None)
            if blocks is not None and isinstance(blocks, (nn.ModuleList, nn.Sequential, list)):
                return len(blocks)
        raise ValueError("Could not determine the number of transformer blocks.")

    def _inject_block_lora(
        self, block: nn.Module, rank: int
    ) -> list[nn.Parameter]:
        """Inject LoRA adapters into all ``nn.Linear`` layers within *block*.

        Returns:
            List of trainable LoRA parameters (A and B matrices).
        """
        import math as _math

        params: list[nn.Parameter] = []
        for name, module in list(block.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            d_out, d_in = module.weight.shape
            lora_A = nn.Parameter(torch.empty(rank, d_in, device=self.device, dtype=self.dtype))
            nn.init.kaiming_uniform_(lora_A, a=_math.sqrt(5))
            lora_B = nn.Parameter(torch.zeros(d_out, rank, device=self.device, dtype=self.dtype))

            # Store LoRA matrices as buffer-like attributes on the module
            module.register_parameter("_probe_lora_A", lora_A)
            module.register_parameter("_probe_lora_B", lora_B)

            # Monkey-patch forward to include LoRA branch
            _original_forward = module.forward
            _rank = rank

            def _make_lora_forward(
                orig_fwd: Any,
                mod: nn.Linear,
                r: int,
            ) -> Any:
                def _forward(x: torch.Tensor) -> torch.Tensor:
                    base_out = orig_fwd(x)
                    scaling = 1.0 / r
                    lora_out = torch.nn.functional.linear(
                        torch.nn.functional.linear(x, mod._probe_lora_A),
                        mod._probe_lora_B,
                    )
                    return base_out + scaling * lora_out
                return _forward

            module.forward = _make_lora_forward(_original_forward, module, _rank)  # type: ignore[assignment]
            module._probe_original_forward = _original_forward  # type: ignore[attr-defined]

            params.extend([lora_A, lora_B])

        return params

    @staticmethod
    def _remove_block_lora(block: nn.Module) -> None:
        """Remove probe LoRA adapters injected by :meth:`_inject_block_lora`."""
        for _name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if hasattr(module, "_probe_original_forward"):
                module.forward = module._probe_original_forward  # type: ignore[assignment]
                del module._probe_original_forward
            if hasattr(module, "_probe_lora_A"):
                del module._probe_lora_A
            if hasattr(module, "_probe_lora_B"):
                del module._probe_lora_B

    def _training_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Execute a single denoising training step.

        This is a simplified diffusion training loop stub. In a real
        deployment this would call the full noise-schedule sampling and
        denoising loss computation. Subclass and override for your specific
        pipeline.

        Args:
            batch: A batch dictionary from the dataloader.

        Returns:
            Scalar loss tensor.
        """
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)

        # Simplified: uniform random timestep, Gaussian noise, MSE loss
        bsz = pixel_values.shape[0]
        noise = torch.randn_like(pixel_values)
        timesteps = torch.randint(0, 1000, (bsz,), device=self.device).long()

        # Noisy input: simple linear interpolation for the probe
        alpha_t = (1.0 - timesteps.float() / 1000.0).view(bsz, 1, 1, 1)
        noisy = alpha_t * pixel_values + (1.0 - alpha_t) * noise

        # Forward through model
        text_embeds = batch.get("text_embeds")
        if text_embeds is not None:
            text_embeds = text_embeds.to(self.device, dtype=self.dtype)

        model_output = self.model(noisy, timesteps, encoder_hidden_states=text_embeds)

        # Handle model outputs that may be dataclass / tuple
        if isinstance(model_output, torch.Tensor):
            predicted = model_output
        elif hasattr(model_output, "sample"):
            predicted = model_output.sample
        else:
            predicted = model_output[0]

        loss = torch.nn.functional.mse_loss(predicted, noise)
        return loss

    def _generate_samples(self, num_samples: int = 4) -> list[torch.Tensor]:
        """Generate images for scoring.

        This is a simplified generation stub. Override for your specific
        diffusion sampling pipeline (e.g. DDIM, Euler).

        Returns:
            List of image tensors in [0, 1] range with shape (3, H, W).
        """
        images: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Simple single-step denoising from noise (placeholder)
                noise = torch.randn(1, 3, 256, 256, device=self.device, dtype=self.dtype)
                timestep = torch.zeros(1, device=self.device).long()
                output = self.model(noise, timestep)
                if isinstance(output, torch.Tensor):
                    img = output
                elif hasattr(output, "sample"):
                    img = output.sample
                else:
                    img = output[0]
                img = img.squeeze(0).clamp(0, 1)
                images.append(img)
        return images

    # ----- Scoring helpers ----- #

    def _ensure_dino_loaded(self) -> None:
        """Lazily load the DINOv2 model and processor."""
        if self._dino_model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "The `transformers` library is required for DINO scoring. "
                "Install it with: pip install transformers"
            ) from exc

        self._dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_name)
        self._dino_model = AutoModel.from_pretrained(self.dino_model_name).to(self.device).eval()

    def _ensure_clip_loaded(self) -> None:
        """Lazily load the CLIP model and processor."""
        if self._clip_model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "The `transformers` library is required for CLIP scoring. "
                "Install it with: pip install transformers"
            ) from exc

        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()

    @torch.no_grad()
    def _compute_dino_score(self, generated_images: list[torch.Tensor]) -> float:
        """Compute mean DINO cosine similarity between generated and reference images.

        Args:
            generated_images: List of tensors in [0, 1], shape (3, H, W).

        Returns:
            Mean cosine similarity (float).
        """
        self._ensure_dino_loaded()

        # Get reference images from dataset
        ref_images: list[torch.Tensor] = []
        for i in range(min(4, len(self.dataset))):
            sample = self.dataset[i]
            pv = sample["pixel_values"]
            if isinstance(pv, torch.Tensor):
                ref_images.append(pv)

        if not ref_images or not generated_images:
            return 0.0

        def _embed_images(imgs: list[torch.Tensor]) -> torch.Tensor:
            """Return L2-normalised CLS embeddings for a list of images."""
            from torchvision.transforms.functional import to_pil_image

            pil_images = [to_pil_image(img.cpu().clamp(0, 1)) for img in imgs]
            inputs = self._dino_processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self._dino_model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0]  # CLS token
            return torch.nn.functional.normalize(cls_embeds, dim=-1)

        ref_embeds = _embed_images(ref_images)  # (N_ref, D)
        gen_embeds = _embed_images(generated_images)  # (N_gen, D)

        # Mean pairwise cosine similarity
        similarity_matrix = gen_embeds @ ref_embeds.T  # (N_gen, N_ref)
        return float(similarity_matrix.mean().item())

    @torch.no_grad()
    def _compute_clip_t_score(self, generated_images: list[torch.Tensor]) -> float:
        """Compute CLIP-T score: cosine similarity between image and text embeddings.

        Uses the prompt text from the dataset to compare against generated
        images via CLIP.

        Args:
            generated_images: List of tensors in [0, 1], shape (3, H, W).

        Returns:
            Mean cosine similarity (float).
        """
        self._ensure_clip_loaded()

        # Attempt to get prompt text from dataset
        prompt = "a photo"  # fallback
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if "prompt" in sample:
                prompt = sample["prompt"]
            elif "text" in sample:
                prompt = sample["text"]

        from torchvision.transforms.functional import to_pil_image

        pil_images = [to_pil_image(img.cpu().clamp(0, 1)) for img in generated_images]

        # Encode text
        text_inputs = self._clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        text_embeds = self._clip_model.get_text_features(**text_inputs)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

        # Encode images
        image_inputs = self._clip_processor(images=pil_images, return_tensors="pt").to(self.device)
        image_embeds = self._clip_model.get_image_features(**image_inputs)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

        similarity = (image_embeds @ text_embeds.T).mean()
        return float(similarity.item())
