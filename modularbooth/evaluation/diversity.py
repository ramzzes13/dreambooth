"""Output diversity metric based on LPIPS perceptual distance.

Given a set of images generated from the *same* prompt, diversity is measured as
the average pairwise LPIPS distance.  Higher values indicate greater output
diversity (the model is not mode-collapsing to a single output).
"""

from __future__ import annotations

import logging
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, torch.Tensor]


class LPIPSDiversity:
    """Output diversity scorer based on LPIPS perceptual distance.

    Computes the mean pairwise LPIPS distance among a set of generated images
    that share the same prompt.  Higher average distance indicates the model
    produces more varied outputs.

    The LPIPS network is loaded lazily on first use.

    Args:
        net: Backbone network for LPIPS (``"alex"``, ``"vgg"``, or ``"squeeze"``).
        device: Torch device string.
    """

    _LPIPS_IMAGE_SIZE: int = 256

    def __init__(
        self,
        net: str = "alex",
        device: str = "cuda",
    ) -> None:
        self.net = net
        self.device = device
        self._lpips_fn: torch.nn.Module | None = None
        self._transform: transforms.Compose | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the LPIPS model if not already loaded."""
        if self._lpips_fn is not None:
            return

        import lpips

        logger.info("Loading LPIPS model (net=%s)", self.net)
        self._lpips_fn = lpips.LPIPS(net=self.net).eval().to(self.device)
        self._transform = transforms.Compose([
            transforms.Resize(
                self._LPIPS_IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self._LPIPS_IMAGE_SIZE),
            transforms.ToTensor(),
            # LPIPS expects images in [-1, 1].
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        logger.info("LPIPS model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: ImageInput) -> torch.Tensor:
        """Convert an image to a tensor in [-1, 1] suitable for LPIPS.

        Args:
            image: A PIL Image or a ``(C, H, W)`` / ``(1, C, H, W)`` tensor.

        Returns:
            A ``(1, C, H, W)`` tensor on ``self.device``.
        """
        assert self._transform is not None, "_ensure_model must be called first"

        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image.squeeze(0)
            # If already in [-1, 1], just resize/crop.
            if image.min() < 0:
                image = (image + 1.0) / 2.0
            image = transforms.ToPILImage()(image.clamp(0, 1).cpu())

        tensor: torch.Tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Diversity computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_diversity(self, images: list[ImageInput]) -> float:
        """Compute the mean pairwise LPIPS distance across a set of images.

        Args:
            images: At least 2 images generated from the same prompt.

        Returns:
            Mean pairwise LPIPS distance (higher means more diverse).

        Raises:
            ValueError: If fewer than 2 images are provided.
        """
        if len(images) < 2:
            raise ValueError(
                "Diversity requires at least 2 images; "
                f"received {len(images)}."
            )

        self._ensure_model()
        assert self._lpips_fn is not None

        tensors = [self._preprocess(img) for img in images]
        n = len(tensors)

        distances: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                dist: torch.Tensor = self._lpips_fn(tensors[i], tensors[j])
                distances.append(dist.item())

        mean_distance = sum(distances) / len(distances)
        logger.info(
            "LPIPS diversity = %.4f (over %d pairs from %d images)",
            mean_distance,
            len(distances),
            n,
        )
        return mean_distance
