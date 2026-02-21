"""Context-Appearance Entanglement (CAE) metric.

Measures how much a subject's visual appearance changes when it is placed in
different contexts / backgrounds.  A *lower* CAE value is better, indicating
that the model preserves subject identity consistently regardless of the scene
it is embedded in.

Formally, CAE is the normalised trace of the covariance matrix of DINO
embeddings computed over images of the same subject in different contexts:

    CAE = tr(Cov(embeddings)) / D

where D is the embedding dimensionality.
"""

from __future__ import annotations

import logging
from typing import Union

import torch
from PIL import Image

from modularbooth.evaluation.dino_score import DINOScore

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, torch.Tensor]

# Type alias for bounding-box coordinates: (x1, y1, x2, y2).
BBox = tuple[int, int, int, int]


class ContextAppearanceEntanglement:
    """Context-Appearance Entanglement (CAE) scorer.

    Lower CAE values indicate that the subject's appearance is stable across
    different contexts, which is desirable.

    Args:
        dino_scorer: An existing :class:`DINOScore` instance used to extract
            embeddings.  Must not be ``None``.
    """

    def __init__(self, dino_scorer: DINOScore) -> None:
        self.dino_scorer = dino_scorer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_region(image: Image.Image, bbox: BBox) -> Image.Image:
        """Crop a rectangular region from a PIL Image.

        Args:
            image: Source image.
            bbox: ``(x1, y1, x2, y2)`` bounding box in pixel coordinates.

        Returns:
            Cropped PIL Image.
        """
        x1, y1, x2, y2 = bbox
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _normalised_covariance_trace(embeddings: torch.Tensor) -> float:
        """Compute the normalised trace of the covariance matrix.

        Args:
            embeddings: Tensor of shape ``(N, D)`` with unit-norm rows.

        Returns:
            ``tr(Cov) / D`` where Cov is the ``(D, D)`` sample covariance.
        """
        # Centre the embeddings.
        mean = embeddings.mean(dim=0, keepdim=True)
        centred = embeddings - mean

        # Sample covariance: (1 / (N-1)) * X^T X
        n = centred.shape[0]
        if n < 2:
            return 0.0

        cov = (centred.T @ centred) / (n - 1)  # (D, D)
        trace = torch.trace(cov).item()
        dim = embeddings.shape[1]
        return trace / dim

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_cae(
        self,
        subject_images_across_contexts: list[ImageInput],
        subject_crops: list[BBox] | None = None,
    ) -> float:
        """Compute CAE for a single subject across multiple contexts.

        Args:
            subject_images_across_contexts: N images of the *same* subject
                placed in different contexts / scenes.
            subject_crops: Optional bounding boxes to crop the subject region
                from each image before computing embeddings.  When ``None`` the
                full images are used.  Must be the same length as
                ``subject_images_across_contexts`` if provided.

        Returns:
            CAE value (lower is better).  Zero means no variance at all
            (identical embeddings across contexts).

        Raises:
            ValueError: If fewer than 2 images are provided (covariance is
                undefined for a single sample).
        """
        if len(subject_images_across_contexts) < 2:
            raise ValueError(
                "CAE requires at least 2 images of the subject in different "
                f"contexts; received {len(subject_images_across_contexts)}."
            )

        if subject_crops is not None:
            if len(subject_crops) != len(subject_images_across_contexts):
                raise ValueError(
                    f"Number of crops ({len(subject_crops)}) must match the "
                    f"number of images ({len(subject_images_across_contexts)})."
                )

        # Optionally crop to the subject region in each image.
        images_to_embed: list[ImageInput] = []
        for i, img in enumerate(subject_images_across_contexts):
            if subject_crops is not None:
                if not isinstance(img, Image.Image):
                    raise TypeError(
                        "Cropping requires PIL Images, but received "
                        f"{type(img).__name__} at index {i}."
                    )
                images_to_embed.append(self._crop_region(img, subject_crops[i]))
            else:
                images_to_embed.append(img)

        # Compute DINO embeddings.
        embeddings = self.dino_scorer.compute_embeddings(images_to_embed)

        cae = self._normalised_covariance_trace(embeddings)
        logger.info(
            "CAE = %.6f (over %d contexts)", cae, len(subject_images_across_contexts)
        )
        return cae

    def compute_batch_cae(
        self,
        subjects_images: list[list[ImageInput]],
        subjects_crops: list[list[BBox] | None] | None = None,
    ) -> dict[str, float]:
        """Compute CAE for multiple subjects and return per-subject and mean scores.

        Args:
            subjects_images: A list where each element is a list of images
                showing one subject in different contexts.
            subjects_crops: Optional per-subject crop lists.  Each element is
                either a list of bounding boxes (same length as the
                corresponding image list) or ``None`` for no cropping.

        Returns:
            Dictionary with keys:

            * ``"per_subject"`` -- list of per-subject CAE values.
            * ``"mean_cae"`` -- mean CAE across all subjects.
        """
        if subjects_crops is not None and len(subjects_crops) != len(subjects_images):
            raise ValueError(
                f"Number of crop lists ({len(subjects_crops)}) must match "
                f"the number of subjects ({len(subjects_images)})."
            )

        per_subject: list[float] = []
        for i, images in enumerate(subjects_images):
            crops = None if subjects_crops is None else subjects_crops[i]
            cae = self.compute_cae(images, subject_crops=crops)
            per_subject.append(cae)

        mean_cae = sum(per_subject) / len(per_subject) if per_subject else 0.0

        logger.info(
            "Batch CAE: mean=%.6f over %d subjects", mean_cae, len(per_subject)
        )
        return {
            "per_subject": per_subject,
            "mean_cae": mean_cae,
        }
