"""Background augmentation for Context-Contrastive Decomposition (CCD) loss.

The CCD loss encourages identity-bearing LoRA blocks to focus on the subject
rather than background context by contrasting features from the original image
against features from background-replaced variants.  This module provides the
augmentation pipeline that produces those variants.

Current implementation uses OpenCV GrabCut for segmentation with a fallback
center-biased elliptical mask.  A hook is provided for replacing the
segmentation backend with SAM-2 for higher-quality masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from PIL import Image, ImageFilter

from modularbooth.data.dataset import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation backend protocol (for SAM-2 integration)
# ---------------------------------------------------------------------------

@runtime_checkable
class SegmentationBackend(Protocol):
    """Protocol that any pluggable segmentation model must satisfy.

    To integrate SAM-2 or another foundation segmentation model, implement
    a class that conforms to this protocol and pass it to
    :class:`BackgroundAugmentor` via the *segmentation_backend* argument.
    """

    def segment(self, image: Image.Image) -> np.ndarray:
        """Return a binary mask (uint8, 0/255) where 255 = foreground subject.

        Args:
            image: RGB PIL image.

        Returns:
            ``np.ndarray`` of shape ``(H, W)`` with dtype ``uint8``.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Background replacement strategies
# ---------------------------------------------------------------------------

def _random_solid_background(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace background with a random solid colour.

    Args:
        image: Source image, ``(H, W, 3)`` uint8 BGR (OpenCV convention).
        mask: Binary mask, ``(H, W)`` uint8, 255 = foreground.
        rng: NumPy random generator for reproducibility.

    Returns:
        Composite image as ``(H, W, 3)`` uint8 BGR.
    """
    colour = rng.integers(0, 256, size=3).astype(np.uint8)
    canvas = np.full_like(image, colour)
    fg_mask = (mask > 127).astype(np.uint8)
    fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1)
    composite = image * fg_mask_3ch + canvas * (1 - fg_mask_3ch)
    return composite.astype(np.uint8)


def _gaussian_noise_background(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace background with Gaussian noise.

    Args:
        image: Source image, ``(H, W, 3)`` uint8 BGR.
        mask: Binary mask, ``(H, W)`` uint8, 255 = foreground.
        rng: NumPy random generator.

    Returns:
        Composite image as ``(H, W, 3)`` uint8 BGR.
    """
    noise = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    fg_mask = (mask > 127).astype(np.uint8)
    fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1)
    composite = image * fg_mask_3ch + noise * (1 - fg_mask_3ch)
    return composite.astype(np.uint8)


def _colour_jitter_background(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Jitter the hue/saturation/brightness of the original background.

    The foreground subject is preserved exactly; only background pixels are
    modified in HSV space.

    Args:
        image: Source image, ``(H, W, 3)`` uint8 BGR.
        mask: Binary mask, ``(H, W)`` uint8, 255 = foreground.
        rng: NumPy random generator.

    Returns:
        Composite image as ``(H, W, 3)`` uint8 BGR.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    hue_shift = int(rng.integers(-30, 31))
    sat_scale = rng.uniform(0.5, 1.5)
    val_shift = int(rng.integers(-40, 41))

    bg_mask = (mask <= 127)
    hsv[:, :, 0][bg_mask] = (hsv[:, :, 0][bg_mask] + hue_shift) % 180
    hsv[:, :, 1][bg_mask] = np.clip(hsv[:, :, 1][bg_mask] * sat_scale, 0, 255)
    hsv[:, :, 2][bg_mask] = np.clip(hsv[:, :, 2][bg_mask] + val_shift, 0, 255)

    jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Restore original foreground to avoid any rounding artefacts.
    fg_mask_3ch = np.stack([(mask > 127).astype(np.uint8)] * 3, axis=-1)
    composite = image * fg_mask_3ch + jittered * (1 - fg_mask_3ch)
    return composite.astype(np.uint8)


def _blurred_background(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Heavily blur the background while keeping the subject sharp.

    Args:
        image: Source image, ``(H, W, 3)`` uint8 BGR.
        mask: Binary mask, ``(H, W)`` uint8, 255 = foreground.
        rng: NumPy random generator (unused, kept for API consistency).

    Returns:
        Composite image as ``(H, W, 3)`` uint8 BGR.
    """
    ksize = 51
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=20)
    fg_mask_3ch = np.stack([(mask > 127).astype(np.uint8)] * 3, axis=-1)
    composite = image * fg_mask_3ch + blurred * (1 - fg_mask_3ch)
    return composite.astype(np.uint8)


# All available replacement strategies, cycled through for variety.
_REPLACEMENT_STRATEGIES = [
    _random_solid_background,
    _gaussian_noise_background,
    _colour_jitter_background,
    _blurred_background,
]


# ---------------------------------------------------------------------------
# Core augmentor
# ---------------------------------------------------------------------------

class BackgroundAugmentor:
    """Generate background-replaced variants of subject images for CCD loss.

    The pipeline has two stages:

    1. **Segmentation** -- Produce a binary foreground mask of the subject.
    2. **Replacement** -- Composite the masked subject onto diverse synthetic
       backgrounds.

    The segmentation backend defaults to OpenCV GrabCut but can be replaced
    with any object implementing :class:`SegmentationBackend` (e.g. SAM-2).

    Args:
        segmentation_backend: Optional pluggable segmentation model.  When
            ``None`` the built-in GrabCut / centre-ellipse fallback is used.
        grabcut_iterations: Number of GrabCut refinement iterations.
        seed: Random seed for reproducible augmentations.

    Example::

        augmentor = BackgroundAugmentor(seed=42)
        augmentor.augment_subject("./data/subject", "./data/augmented", num_variants=5)
    """

    def __init__(
        self,
        segmentation_backend: SegmentationBackend | None = None,
        grabcut_iterations: int = 5,
        seed: int = 42,
    ) -> None:
        self._backend = segmentation_backend
        self._grabcut_iters = grabcut_iterations
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_subject(self, image: Image.Image) -> np.ndarray:
        """Produce a binary foreground mask for the main subject.

        If a pluggable :class:`SegmentationBackend` was provided at
        construction time it is used directly.  Otherwise the method
        attempts OpenCV GrabCut and, if that fails, falls back to a
        centre-biased elliptical mask.

        Args:
            image: RGB PIL image.

        Returns:
            Binary mask as ``np.ndarray`` of shape ``(H, W)`` with dtype
            ``uint8``.  Foreground pixels are 255, background pixels are 0.
        """
        if self._backend is not None:
            return self._backend.segment(image)

        return self._grabcut_segment(image)

    def _grabcut_segment(self, image: Image.Image) -> np.ndarray:
        """Segment using OpenCV GrabCut with a centre-rectangle initialisation.

        Falls back to a centre-biased ellipse if GrabCut raises an exception
        (e.g. on very small images).

        Args:
            image: RGB PIL image.

        Returns:
            Binary mask ``(H, W)`` uint8, 255 = foreground.
        """
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Define a centre rectangle covering ~60% of the image area.
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                img_bgr, mask, rect, bg_model, fg_model,
                self._grabcut_iters, cv2.GC_INIT_WITH_RECT,
            )
            # GrabCut labels: 0 = BG, 1 = FG, 2 = probable BG, 3 = probable FG
            binary_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except cv2.error:
            logger.warning("GrabCut failed, falling back to centre-ellipse mask.")
            binary_mask = self._centre_ellipse_mask(h, w)

        return binary_mask

    @staticmethod
    def _centre_ellipse_mask(h: int, w: int) -> np.ndarray:
        """Generate a centre-biased elliptical mask as a last-resort fallback.

        The ellipse axes cover ~70% of each dimension, producing a soft
        approximation of a centred subject.

        Args:
            h: Image height in pixels.
            w: Image width in pixels.

        Returns:
            Binary mask ``(H, W)`` uint8, 255 = foreground.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        centre = (w // 2, h // 2)
        axes = (int(w * 0.35), int(h * 0.35))
        cv2.ellipse(mask, centre, axes, angle=0, startAngle=0, endAngle=360,
                     color=255, thickness=-1)
        return mask

    # ------------------------------------------------------------------
    # SAM-2 integration hook
    # ------------------------------------------------------------------

    @staticmethod
    def create_sam2_backend(
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        confidence_threshold: float = 0.8,
    ) -> SegmentationBackend:
        """Factory for a SAM-2 based segmentation backend.

        This is a placeholder that documents the expected integration
        surface.  Replace the body with actual SAM-2 loading logic when
        the dependency is available.

        Args:
            model_id: HuggingFace model identifier for SAM-2.
            device: Torch device string.
            confidence_threshold: Minimum confidence for accepting a mask.

        Returns:
            An object satisfying :class:`SegmentationBackend`.

        Raises:
            NotImplementedError: Always, until SAM-2 integration is complete.
        """
        raise NotImplementedError(
            f"SAM-2 backend ({model_id}) is not yet implemented. "
            "Install segment-anything-2 and provide the integration here."
        )

    # ------------------------------------------------------------------
    # Background replacement
    # ------------------------------------------------------------------

    def replace_background(
        self,
        image: Image.Image,
        mask: np.ndarray,
        num_variants: int = 5,
    ) -> list[Image.Image]:
        """Generate background-replaced variants of a subject image.

        The method cycles through the available replacement strategies
        (solid colour, Gaussian noise, colour jitter, heavy blur) to
        produce *num_variants* diverse composites.

        Args:
            image: RGB PIL image of the subject.
            mask: Binary foreground mask ``(H, W)`` uint8, 255 = foreground.
            num_variants: Number of variants to produce.

        Returns:
            List of RGB PIL images with the subject composited onto new
            backgrounds.
        """
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        num_strategies = len(_REPLACEMENT_STRATEGIES)

        variants: list[Image.Image] = []
        for i in range(num_variants):
            strategy = _REPLACEMENT_STRATEGIES[i % num_strategies]
            composite_bgr = strategy(img_bgr, mask, self._rng)
            composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
            variants.append(Image.fromarray(composite_rgb))

        return variants

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def augment_subject(
        self,
        subject_images_dir: str | Path,
        output_dir: str | Path,
        num_variants: int = 5,
    ) -> Path:
        """Process all images in a subject directory and write augmented variants.

        Output layout::

            output_dir/
                <image_stem>/
                    variant_000.png
                    variant_001.png
                    ...

        This layout is directly consumable by :class:`DreamBoothDataset` via
        the *augmented_images_dir* parameter.

        Args:
            subject_images_dir: Directory containing the original subject images.
            output_dir: Root directory for augmented output.
            num_variants: Number of background-replaced variants per image.

        Returns:
            The resolved *output_dir* path.

        Raises:
            FileNotFoundError: If *subject_images_dir* does not exist or
                contains no images.
        """
        src_dir = Path(subject_images_dir)
        out_dir = Path(output_dir)

        if not src_dir.is_dir():
            raise FileNotFoundError(f"Subject images directory not found: {src_dir}")

        image_paths = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {src_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            logger.info("Augmenting %s", img_path.name)
            image = Image.open(img_path).convert("RGB")
            mask = self.segment_subject(image)
            variants = self.replace_background(image, mask, num_variants=num_variants)

            variant_dir = out_dir / img_path.stem
            variant_dir.mkdir(parents=True, exist_ok=True)

            for vi, variant in enumerate(variants):
                save_path = variant_dir / f"variant_{vi:03d}.png"
                variant.save(save_path)
                logger.debug("  Saved %s", save_path)

        logger.info(
            "Augmentation complete: %d images x %d variants -> %s",
            len(image_paths),
            num_variants,
            out_dir,
        )
        return out_dir.resolve()
