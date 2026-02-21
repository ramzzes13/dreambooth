"""Identity Isolation Score (IIS) for multi-subject composition.

Measures how well individual subjects are isolated within a multi-subject
generated image -- i.e. each subject region should resemble its own reference
images and *not* resemble the references for other subjects.  A high IIS
indicates that the model successfully avoids identity mixing / entanglement
between co-occurring subjects.
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


class IdentityIsolationScore:
    """Identity Isolation Score based on DINO similarity.

    For each subject region in a multi-subject generated image the scorer
    computes:

    * **pos_sim** -- mean cosine similarity between the cropped region and the
      reference images of the *correct* subject.
    * **neg_sim** -- maximum mean cosine similarity between the cropped region
      and the reference images of any *incorrect* subject.

    The per-subject isolation is ``pos_sim - neg_sim`` and the overall IIS is
    the mean across subjects.

    Args:
        dino_scorer: An existing :class:`DINOScore` instance.  If ``None``, a
            new scorer with default settings will be created.
        device: Torch device string, only used when creating a new
            ``DINOScore``.
    """

    def __init__(
        self,
        dino_scorer: DINOScore | None = None,
        device: str = "cuda",
    ) -> None:
        self.dino_scorer = dino_scorer or DINOScore(device=device)

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

        Raises:
            ValueError: If the bounding box is degenerate (zero or negative area).
        """
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Degenerate bounding box: ({x1}, {y1}, {x2}, {y2}). "
                "Ensure x2 > x1 and y2 > y1."
            )
        return image.crop((x1, y1, x2, y2))

    def _mean_similarity_to_refs(
        self,
        crop_embedding: torch.Tensor,
        reference_images: list[ImageInput],
    ) -> float:
        """Compute mean cosine similarity between a crop embedding and reference images.

        Args:
            crop_embedding: Normalised 1-D embedding of the cropped region.
            reference_images: List of reference images for a single subject.

        Returns:
            Mean cosine similarity (scalar).
        """
        ref_embs = self.dino_scorer.compute_embeddings(reference_images)
        similarities = crop_embedding @ ref_embs.T  # (R,)
        return similarities.mean().item()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_iis(
        self,
        generated_image: Image.Image,
        subject_crops: list[BBox],
        reference_images: dict[int, list[ImageInput]],
    ) -> float:
        """Compute the Identity Isolation Score for a single generated image.

        Args:
            generated_image: The multi-subject generated image.
            subject_crops: Bounding boxes ``(x1, y1, x2, y2)`` for each subject
                in the generated image, ordered by subject index.  The i-th box
                corresponds to subject ``i``.
            reference_images: Mapping from subject index to a list of reference
                images for that subject.  Must contain an entry for every
                subject index in ``range(len(subject_crops))``.

        Returns:
            The IIS value (higher is better).  A value of 1.0 would mean
            perfect isolation; negative values indicate severe mixing.

        Raises:
            ValueError: If the number of crops and reference entries are
                inconsistent.
        """
        num_subjects = len(subject_crops)
        subject_ids = sorted(reference_images.keys())

        if num_subjects != len(reference_images):
            raise ValueError(
                f"Number of subject crops ({num_subjects}) does not match "
                f"the number of reference image groups ({len(reference_images)})."
            )
        if num_subjects < 2:
            raise ValueError(
                "IIS requires at least 2 subjects; received "
                f"{num_subjects}.  For single-subject evaluation use "
                "DINOScore.compute_score() instead."
            )

        per_subject_isolation: list[float] = []

        for idx, bbox in enumerate(subject_crops):
            subject_id = subject_ids[idx]

            # Crop the subject region from the generated image.
            crop = self._crop_region(generated_image, bbox)
            crop_emb = self.dino_scorer.compute_embedding(crop)

            # Positive: similarity to the correct subject references.
            pos_sim = self._mean_similarity_to_refs(
                crop_emb, reference_images[subject_id]
            )

            # Negative: maximum similarity to any *other* subject's references.
            neg_sims: list[float] = []
            for other_id in subject_ids:
                if other_id == subject_id:
                    continue
                neg_sim = self._mean_similarity_to_refs(
                    crop_emb, reference_images[other_id]
                )
                neg_sims.append(neg_sim)

            max_neg_sim = max(neg_sims)
            isolation = pos_sim - max_neg_sim
            per_subject_isolation.append(isolation)

            logger.debug(
                "Subject %d: pos=%.4f, max_neg=%.4f, isolation=%.4f",
                subject_id, pos_sim, max_neg_sim, isolation,
            )

        iis = sum(per_subject_isolation) / len(per_subject_isolation)
        logger.info("IIS = %.4f (over %d subjects)", iis, num_subjects)
        return iis

    def compute_batch_iis(
        self,
        generated_images: list[Image.Image],
        subject_crops_per_image: list[list[BBox]],
        reference_images: dict[int, list[ImageInput]],
    ) -> float:
        """Compute the mean IIS across a batch of generated images.

        Args:
            generated_images: List of multi-subject generated images.
            subject_crops_per_image: Per-image list of bounding boxes, aligned
                with ``generated_images``.
            reference_images: Shared mapping from subject index to reference
                images (same subjects across all generated images).

        Returns:
            Mean IIS across the batch.

        Raises:
            ValueError: If batch sizes do not match.
        """
        if len(generated_images) != len(subject_crops_per_image):
            raise ValueError(
                f"Number of generated images ({len(generated_images)}) must "
                f"match the number of crop lists ({len(subject_crops_per_image)})."
            )

        scores: list[float] = []
        for gen_img, crops in zip(generated_images, subject_crops_per_image):
            score = self.compute_iis(gen_img, crops, reference_images)
            scores.append(score)

        mean_iis = sum(scores) / len(scores)
        logger.info("Batch IIS = %.4f (over %d images)", mean_iis, len(scores))
        return mean_iis
