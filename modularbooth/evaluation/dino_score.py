"""DINO / DINOv2 subject-fidelity metrics.

Computes cosine similarity between DINO (or DINOv2) CLS-token embeddings of
generated images and reference subject images.  Higher scores indicate that the
generated subject more faithfully reproduces the reference appearance.
"""

from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, torch.Tensor]


class DINOScore:
    """Subject-fidelity scorer based on DINO ViT embeddings.

    The model is loaded lazily on the first call to any method that requires it,
    keeping construction cheap and allowing metric objects to be created eagerly
    in configuration code.

    Args:
        model_name: HuggingFace hub identifier for a DINO ViT checkpoint.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    """

    # Default preprocessing for DINO ViTs trained on ImageNet.
    _DINO_IMAGE_SIZE: int = 224
    _DINO_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
    _DINO_STD: tuple[float, ...] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "facebook/dino-vits16",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device

        # Populated lazily by ``_ensure_model``.
        self._model: torch.nn.Module | None = None
        self._transform: transforms.Compose | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _build_transform(self) -> transforms.Compose:
        """Build the image preprocessing pipeline for this model.

        Returns:
            A torchvision ``Compose`` transform suitable for DINO ViTs.
        """
        return transforms.Compose([
            transforms.Resize(
                self._DINO_IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self._DINO_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._DINO_MEAN, std=self._DINO_STD),
        ])

    def _ensure_model(self) -> None:
        """Load the model and preprocessing pipeline if not already loaded."""
        if self._model is not None:
            return

        logger.info("Loading DINO model: %s", self.model_name)
        self._model = torch.hub.load("facebookresearch/dino:main", self._hub_name())
        self._model.eval()
        self._model.to(self.device)
        self._transform = self._build_transform()
        logger.info("DINO model loaded on %s", self.device)

    def _hub_name(self) -> str:
        """Map the HuggingFace-style model name to the ``torch.hub`` entry point.

        ``facebook/dino-vits16`` -> ``dino_vits16``.
        """
        # Strip the "facebook/" prefix and replace hyphens with underscores.
        short = self.model_name.split("/")[-1]
        return short.replace("-", "_")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: ImageInput) -> torch.Tensor:
        """Convert an image to a model-ready tensor.

        Args:
            image: A PIL Image or a torch.Tensor.  Tensors are expected in
                ``(C, H, W)`` or ``(1, C, H, W)`` layout with values in [0, 1]
                or [-1, 1].

        Returns:
            A ``(1, C, H, W)`` tensor on ``self.device``.
        """
        assert self._transform is not None, "_ensure_model must be called first"

        if isinstance(image, torch.Tensor):
            # Ensure 3-D (C, H, W) for transforms.
            if image.ndim == 4:
                image = image.squeeze(0)
            # If pixel range is [-1, 1], shift to [0, 1] for ToPILImage.
            if image.min() < 0:
                image = (image + 1.0) / 2.0
            image = transforms.ToPILImage()(image.clamp(0, 1).cpu())

        tensor: torch.Tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_embedding(self, image: ImageInput) -> torch.Tensor:
        """Compute the L2-normalised CLS-token embedding for a single image.

        Args:
            image: PIL Image or torch Tensor.

        Returns:
            A 1-D tensor of shape ``(D,)`` with unit norm.
        """
        self._ensure_model()
        assert self._model is not None
        pixel_values = self._preprocess(image)
        embedding: torch.Tensor = self._model(pixel_values)
        return F.normalize(embedding.flatten(), dim=0)

    @torch.no_grad()
    def compute_embeddings(self, images: list[ImageInput]) -> torch.Tensor:
        """Compute embeddings for a list of images.

        Args:
            images: List of PIL Images or torch Tensors.

        Returns:
            A tensor of shape ``(N, D)`` with unit-norm rows.
        """
        self._ensure_model()
        assert self._model is not None
        batch = torch.cat([self._preprocess(img) for img in images], dim=0)
        embeddings: torch.Tensor = self._model(batch)
        return F.normalize(embeddings, dim=1)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_score(
        self,
        generated_images: list[ImageInput],
        reference_images: list[ImageInput],
    ) -> float:
        """Compute the average pairwise cosine similarity (DINO-I score).

        For each generated image, the cosine similarity to every reference image
        is computed.  The returned score is the mean over all
        ``(generated, reference)`` pairs.

        Args:
            generated_images: Images produced by the model.
            reference_images: Ground-truth subject reference images.

        Returns:
            Mean cosine similarity in ``[-1, 1]`` (higher is better).
        """
        gen_embs = self.compute_embeddings(generated_images)
        ref_embs = self.compute_embeddings(reference_images)
        similarity_matrix = gen_embs @ ref_embs.T  # (G, R)
        return similarity_matrix.mean().item()

    def compute_pairwise_matrix(
        self,
        images_a: list[ImageInput],
        images_b: list[ImageInput],
    ) -> torch.Tensor:
        """Compute the full pairwise cosine similarity matrix.

        Args:
            images_a: First set of images (rows).
            images_b: Second set of images (columns).

        Returns:
            A tensor of shape ``(len(images_a), len(images_b))``.
        """
        embs_a = self.compute_embeddings(images_a)
        embs_b = self.compute_embeddings(images_b)
        return embs_a @ embs_b.T


class DINOv2Score(DINOScore):
    """Subject-fidelity scorer based on DINOv2 ViT-B/14 embeddings.

    DINOv2 offers improved representations over the original DINO, especially
    for fine-grained visual similarity.  This subclass overrides the default
    model and adjusts preprocessing to match the DINOv2 input requirements
    (14-pixel patch size, 518 px recommended resolution).

    Args:
        model_name: HuggingFace hub identifier for a DINOv2 checkpoint.
        device: Torch device string.
    """

    _DINO_IMAGE_SIZE: int = 518
    _DINO_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
    _DINO_STD: tuple[float, ...] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "facebook/dinov2-vitb14",
        device: str = "cuda",
    ) -> None:
        super().__init__(model_name=model_name, device=device)

    def _ensure_model(self) -> None:
        """Load DINOv2 from ``torch.hub`` (facebookresearch/dinov2)."""
        if self._model is not None:
            return

        logger.info("Loading DINOv2 model: %s", self.model_name)
        hub_name = self._hub_name()
        self._model = torch.hub.load("facebookresearch/dinov2:main", hub_name)
        self._model.eval()
        self._model.to(self.device)
        self._transform = self._build_transform()
        logger.info("DINOv2 model loaded on %s", self.device)

    def _hub_name(self) -> str:
        """Map the HuggingFace-style model name to the ``torch.hub`` entry point.

        ``facebook/dinov2-vitb14`` -> ``dinov2_vitb14``.
        """
        short = self.model_name.split("/")[-1]
        return short.replace("-", "_")

    def _build_transform(self) -> transforms.Compose:
        """Build the DINOv2-specific preprocessing pipeline.

        DINOv2 models with a 14 px patch size work best at 518 px
        (518 = 14 * 37) to avoid interpolating positional embeddings.

        Returns:
            A torchvision ``Compose`` transform.
        """
        return transforms.Compose([
            transforms.Resize(
                self._DINO_IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self._DINO_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._DINO_MEAN, std=self._DINO_STD),
        ])
