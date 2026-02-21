"""CLIP-based image-image (CLIP-I) and text-image (CLIP-T) similarity scores.

Uses the ``open_clip`` library to load CLIP models and compute normalised
embeddings for both images and text.  CLIP-I measures how visually similar a
generated image is to reference images, while CLIP-T measures how well a
generated image matches its text prompt.
"""

from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, torch.Tensor]


class CLIPScore:
    """CLIP-based scorer for image-image (CLIP-I) and text-image (CLIP-T) alignment.

    The model is loaded lazily on first use, keeping construction lightweight.

    Args:
        model_name: ``open_clip`` model architecture name (e.g. ``"ViT-L-14"``).
        pretrained: Pretrained weight tag (e.g. ``"openai"``, ``"laion2b_s32b_b82k"``).
        device: Torch device string.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device

        # Populated lazily.
        self._model: torch.nn.Module | None = None
        self._tokenizer = None
        self._preprocess = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the CLIP model, tokenizer, and preprocessing if not already loaded."""
        if self._model is not None:
            return

        import open_clip

        logger.info(
            "Loading CLIP model: %s (pretrained=%s)", self.model_name, self.pretrained
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        logger.info("CLIP model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _prepare_image(self, image: ImageInput) -> torch.Tensor:
        """Convert an image to a model-ready CLIP tensor.

        Args:
            image: A PIL Image or a ``(C, H, W)`` / ``(1, C, H, W)`` tensor.

        Returns:
            A ``(1, C, H, W)`` tensor on ``self.device``.
        """
        assert self._preprocess is not None, "_ensure_model must be called first"

        if isinstance(image, torch.Tensor):
            from torchvision import transforms as T

            if image.ndim == 4:
                image = image.squeeze(0)
            if image.min() < 0:
                image = (image + 1.0) / 2.0
            image = T.ToPILImage()(image.clamp(0, 1).cpu())

        tensor: torch.Tensor = self._preprocess(image)
        return tensor.unsqueeze(0).to(self.device)

    def _prepare_images(self, images: list[ImageInput]) -> torch.Tensor:
        """Batch-prepare a list of images.

        Args:
            images: List of PIL Images or tensors.

        Returns:
            A ``(N, C, H, W)`` tensor on ``self.device``.
        """
        return torch.cat([self._prepare_image(img) for img in images], dim=0)

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_image_embedding(self, image: ImageInput) -> torch.Tensor:
        """Compute the L2-normalised CLIP image embedding for a single image.

        Args:
            image: PIL Image or torch Tensor.

        Returns:
            A 1-D tensor of shape ``(D,)`` with unit norm.
        """
        self._ensure_model()
        assert self._model is not None
        pixel_values = self._prepare_image(image)
        embedding: torch.Tensor = self._model.encode_image(pixel_values)
        return F.normalize(embedding.flatten(), dim=0)

    @torch.no_grad()
    def compute_image_embeddings(self, images: list[ImageInput]) -> torch.Tensor:
        """Compute L2-normalised CLIP image embeddings for a batch of images.

        Args:
            images: List of PIL Images or tensors.

        Returns:
            A tensor of shape ``(N, D)`` with unit-norm rows.
        """
        self._ensure_model()
        assert self._model is not None
        pixel_values = self._prepare_images(images)
        embeddings: torch.Tensor = self._model.encode_image(pixel_values)
        return F.normalize(embeddings, dim=1)

    @torch.no_grad()
    def compute_text_embedding(self, text: str) -> torch.Tensor:
        """Compute the L2-normalised CLIP text embedding for a single prompt.

        Args:
            text: A text prompt string.

        Returns:
            A 1-D tensor of shape ``(D,)`` with unit norm.
        """
        self._ensure_model()
        assert self._model is not None and self._tokenizer is not None
        tokens = self._tokenizer([text]).to(self.device)
        embedding: torch.Tensor = self._model.encode_text(tokens)
        return F.normalize(embedding.flatten(), dim=0)

    @torch.no_grad()
    def compute_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Compute L2-normalised CLIP text embeddings for a batch of prompts.

        Args:
            texts: List of text prompt strings.

        Returns:
            A tensor of shape ``(N, D)`` with unit-norm rows.
        """
        self._ensure_model()
        assert self._model is not None and self._tokenizer is not None
        tokens = self._tokenizer(texts).to(self.device)
        embeddings: torch.Tensor = self._model.encode_text(tokens)
        return F.normalize(embeddings, dim=1)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def clip_i_score(
        self,
        generated_images: list[ImageInput],
        reference_images: list[ImageInput],
    ) -> float:
        """Compute CLIP-I score: average image-image cosine similarity.

        For each generated image the cosine similarity to every reference image
        is computed; the returned value is the grand mean over all pairs.

        Args:
            generated_images: Images produced by the model.
            reference_images: Ground-truth subject reference images.

        Returns:
            Mean cosine similarity in ``[-1, 1]`` (higher is better).
        """
        gen_embs = self.compute_image_embeddings(generated_images)
        ref_embs = self.compute_image_embeddings(reference_images)
        similarity_matrix = gen_embs @ ref_embs.T  # (G, R)
        return similarity_matrix.mean().item()

    def clip_t_score(
        self,
        generated_images: list[ImageInput],
        prompts: list[str],
    ) -> float:
        """Compute CLIP-T score: average text-image cosine similarity.

        Each generated image is paired with its corresponding prompt (by index).
        The returned value is the mean cosine similarity across all pairs.

        Args:
            generated_images: Images produced by the model.
            prompts: Text prompts aligned 1-to-1 with ``generated_images``.

        Returns:
            Mean cosine similarity in ``[-1, 1]`` (higher is better).

        Raises:
            ValueError: If the number of images and prompts do not match.
        """
        if len(generated_images) != len(prompts):
            raise ValueError(
                f"Number of generated images ({len(generated_images)}) must match "
                f"the number of prompts ({len(prompts)})."
            )

        img_embs = self.compute_image_embeddings(generated_images)
        txt_embs = self.compute_text_embeddings(prompts)

        # Paired cosine similarity (diagonal of the full similarity matrix).
        paired_sims = (img_embs * txt_embs).sum(dim=1)  # (N,)
        return paired_sims.mean().item()
