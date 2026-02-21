"""DreamBooth dataset with interleaved subject and class images.

Supports prior-preservation loss (PPL) through class images and
context-contrastive decomposition (CCD) loss through augmented images
with replaced backgrounds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Supported image extensions for directory scanning.
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _collect_image_paths(directory: Path) -> list[Path]:
    """Collect and sort image paths from a directory.

    Args:
        directory: Path to a directory containing image files.

    Returns:
        Sorted list of paths to image files found in the directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")
    return paths


def _build_image_transform(resolution: int) -> transforms.Compose:
    """Build the standard image preprocessing pipeline.

    Images are resized so the shorter edge matches *resolution*, then
    center-cropped to a square, converted to a tensor, and normalized
    to the [-1, 1] range expected by diffusion models.

    Args:
        resolution: Target spatial resolution (height = width).

    Returns:
        A composed torchvision transform.
    """
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class DreamBoothDataset(torch.utils.data.Dataset):
    """Dataset that interleaves subject images and class images for DreamBooth training.

    Each sample is a dictionary containing:

    * ``pixel_values`` -- Image tensor normalised to [-1, 1], shape ``(3, H, W)``.
    * ``input_ids`` -- The text prompt string (tokenisation is deferred to the
      training loop so the dataset stays tokenizer-agnostic).
    * ``is_class_image`` -- Boolean flag indicating whether this sample is a
      prior-preservation class image.

    When *augmented_images_dir* is provided (for CCD loss), subject-image
    samples additionally contain:

    * ``augmented_pixel_values`` -- A list of augmented-image tensors where the
      subject background has been replaced.

    Subject and class images are interleaved so that each training batch
    naturally contains both kinds of samples when using a sequential sampler.

    Args:
        subject_images_dir: Path to directory with 3-5 subject images.
        class_images_dir: Path to directory with prior-preservation class images.
        token: Rare identifier token, e.g. ``"[V]"``.
        class_noun: Natural-language class noun, e.g. ``"dog"``.
        captions: Optional list of informative captions aligned 1-to-1 with the
            subject images (sorted alphabetically by filename).  When ``None``,
            a simple ``"a {token} {class_noun}"`` template is used.
        resolution: Spatial resolution for the output tensors.
        augmented_images_dir: Optional directory containing CCD background-
            augmented variants.  Expected layout is one sub-directory per
            source image (named after the source filename stem) each holding
            the augmented variants.
    """

    def __init__(
        self,
        subject_images_dir: str | Path,
        class_images_dir: str | Path,
        token: str,
        class_noun: str,
        captions: list[str] | None = None,
        resolution: int = 1024,
        augmented_images_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        self.token = token
        self.class_noun = class_noun
        self.resolution = resolution
        self.transform = _build_image_transform(resolution)

        # ---- Subject images ------------------------------------------------
        self.subject_paths = _collect_image_paths(Path(subject_images_dir))
        num_subjects = len(self.subject_paths)
        logger.info("Loaded %d subject images from %s", num_subjects, subject_images_dir)

        # ---- Captions -------------------------------------------------------
        if captions is not None:
            if len(captions) != num_subjects:
                raise ValueError(
                    f"Number of captions ({len(captions)}) does not match "
                    f"the number of subject images ({num_subjects})."
                )
            self.subject_captions = list(captions)
        else:
            default_caption = f"a {self.token} {self.class_noun}"
            self.subject_captions = [default_caption] * num_subjects

        # ---- Class images (prior preservation) ------------------------------
        self.class_paths = _collect_image_paths(Path(class_images_dir))
        self.class_caption = f"a {self.class_noun}"
        logger.info("Loaded %d class images from %s", len(self.class_paths), class_images_dir)

        # ---- Augmented images (CCD loss) ------------------------------------
        self.augmented_images_dir: Path | None = None
        self._augmented_map: dict[str, list[Path]] = {}
        if augmented_images_dir is not None:
            aug_dir = Path(augmented_images_dir)
            if aug_dir.is_dir():
                self.augmented_images_dir = aug_dir
                self._augmented_map = self._build_augmented_map(aug_dir)
                logger.info(
                    "Found augmented images for %d subjects in %s",
                    len(self._augmented_map),
                    aug_dir,
                )
            else:
                logger.warning(
                    "Augmented images directory does not exist, CCD augmentation disabled: %s",
                    aug_dir,
                )

        # ---- Build interleaved index ---------------------------------------
        # We interleave subject and class images so that sequential iteration
        # yields both kinds in close succession.  Class images are cycled to
        # match the number of subject images if there are fewer class images.
        self._samples: list[dict[str, Any]] = self._build_interleaved_samples()
        logger.info("Dataset size (interleaved): %d samples", len(self._samples))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_augmented_map(self, aug_dir: Path) -> dict[str, list[Path]]:
        """Map each subject image stem to its list of augmented variant paths.

        Expected layout::

            augmented_images_dir/
                subject_001/
                    variant_0.png
                    variant_1.png
                    ...
                subject_002/
                    ...

        Args:
            aug_dir: Root directory of augmented images.

        Returns:
            Mapping from subject filename stem to sorted list of variant paths.
        """
        mapping: dict[str, list[Path]] = {}
        for subject_path in self.subject_paths:
            stem = subject_path.stem
            variant_dir = aug_dir / stem
            if variant_dir.is_dir():
                variants = sorted(
                    p for p in variant_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
                )
                if variants:
                    mapping[stem] = variants
        return mapping

    def _build_interleaved_samples(self) -> list[dict[str, Any]]:
        """Build the flat interleaved list of sample metadata.

        For every subject image we insert one subject entry followed by one
        class entry (cycling class images as needed).  This keeps the ratio
        roughly 1:1 which is the standard DreamBooth PPL balance.

        Returns:
            List of sample metadata dicts (paths and captions, not yet loaded).
        """
        samples: list[dict[str, Any]] = []
        num_class = len(self.class_paths)

        for idx, subject_path in enumerate(self.subject_paths):
            # Subject sample
            aug_key = subject_path.stem
            samples.append({
                "image_path": subject_path,
                "caption": self.subject_captions[idx],
                "is_class_image": False,
                "augmented_variants": self._augmented_map.get(aug_key, []),
            })
            # Paired class sample (cycled)
            class_path = self.class_paths[idx % num_class]
            samples.append({
                "image_path": class_path,
                "caption": self.class_caption,
                "is_class_image": True,
                "augmented_variants": [],
            })

        # Append remaining class images so they are not wasted.
        if num_class > len(self.subject_paths):
            for ci in range(len(self.subject_paths), num_class):
                samples.append({
                    "image_path": self.class_paths[ci],
                    "caption": self.class_caption,
                    "is_class_image": True,
                    "augmented_variants": [],
                })

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and transform a single sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with at minimum ``pixel_values``, ``input_ids``, and
            ``is_class_image``.  Subject samples that have augmented variants
            also include ``augmented_pixel_values``.
        """
        meta = self._samples[index]

        image = Image.open(meta["image_path"]).convert("RGB")
        pixel_values: torch.Tensor = self.transform(image)

        sample: dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": meta["caption"],
            "is_class_image": meta["is_class_image"],
        }

        # Load augmented variants when available (subject images only).
        if meta["augmented_variants"]:
            augmented_tensors: list[torch.Tensor] = []
            for variant_path in meta["augmented_variants"]:
                variant_image = Image.open(variant_path).convert("RGB")
                augmented_tensors.append(self.transform(variant_image))
            sample["augmented_pixel_values"] = augmented_tensors

        return sample

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        subject_images_dir: str | Path,
        class_images_dir: str | Path,
        cfg: Any,
        captions: list[str] | None = None,
        augmented_images_dir: str | Path | None = None,
    ) -> "DreamBoothDataset":
        """Create a dataset from an OmegaConf config object.

        Reads ``cfg.subject.token``, ``cfg.subject.class_noun``, and
        ``cfg.inference.resolution`` (falling back to 1024).

        Args:
            subject_images_dir: Path to subject images.
            class_images_dir: Path to class images.
            cfg: OmegaConf DictConfig (see ``configs/default.yaml``).
            captions: Optional pre-computed captions.
            augmented_images_dir: Optional CCD augmented images path.

        Returns:
            A configured ``DreamBoothDataset`` instance.
        """
        return cls(
            subject_images_dir=subject_images_dir,
            class_images_dir=class_images_dir,
            token=cfg.subject.token,
            class_noun=cfg.subject.class_noun,
            captions=captions,
            resolution=getattr(cfg.inference, "resolution", 1024),
            augmented_images_dir=augmented_images_dir,
        )
