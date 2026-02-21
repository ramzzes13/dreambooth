"""Subject captioning for DreamBooth training prompts.

Diverse, informative captions improve DreamBooth fine-tuning by exposing the
model to the rare token in varied linguistic contexts.  This module provides
a template-based captioner for immediate use and a placeholder hook for
LLM-based captioning (e.g. LLaVA, GPT-4V) when richer descriptions are
needed.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from PIL import Image

from modularbooth.data.dataset import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption templates
# ---------------------------------------------------------------------------

# Templates are ordered roughly from simple to complex.  The captioner cycles
# through them so that every subject image receives a distinct prompt style.
_CAPTION_TEMPLATES: list[str] = [
    "a {token} {class_noun}",
    "a photo of a {token} {class_noun}",
    "a high-resolution photo of a {token} {class_noun}",
    "a close-up photo of a {token} {class_noun}",
    "a bright photo of a {token} {class_noun}",
    "a dark photo of a {token} {class_noun}",
    "a cropped photo of a {token} {class_noun}",
    "a good photo of a {token} {class_noun}",
    "a rendition of a {token} {class_noun}",
    "a depiction of a {token} {class_noun}",
    "a {token} {class_noun} in high resolution",
    "a {token} {class_noun}, detailed, high quality",
    "a professional photograph of a {token} {class_noun}",
    "a detailed image of a {token} {class_noun}",
    "an image of a {token} {class_noun}",
    "a {token} {class_noun}, sharp focus",
]


class SubjectCaptioner:
    """Generate diverse text prompts for DreamBooth subject images.

    The default mode is *template-based*: each image receives a caption drawn
    from a pool of varied templates that embed the rare token and class noun.
    An LLM-based mode is provided as a placeholder for future integration.

    Args:
        templates: Optional custom list of caption templates.  Each template
            must contain ``{token}`` and ``{class_noun}`` placeholders.
        seed: Random seed for reproducible template selection.

    Example::

        captioner = SubjectCaptioner()
        captions = captioner.caption_dataset("./data/subject", "[V]", "dog")
        # ["a [V] dog", "a photo of a [V] dog", ...]
    """

    def __init__(
        self,
        templates: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.templates = list(templates) if templates is not None else list(_CAPTION_TEMPLATES)
        self._rng = random.Random(seed)

        # Validate templates contain required placeholders.
        for tmpl in self.templates:
            if "{token}" not in tmpl or "{class_noun}" not in tmpl:
                raise ValueError(
                    f"Every template must contain '{{token}}' and '{{class_noun}}' "
                    f"placeholders.  Invalid template: {tmpl!r}"
                )

    # ------------------------------------------------------------------
    # Template-based captioning
    # ------------------------------------------------------------------

    def generate_caption(
        self,
        image_path: str | Path,
        token: str,
        class_noun: str,
    ) -> str:
        """Generate a single caption for an image using a randomly selected template.

        The image itself is not inspected in template mode; the template is
        chosen pseudo-randomly for diversity.  In LLM mode (future), the
        image content would inform the caption.

        Args:
            image_path: Path to the image file (used for logging; the image is
                not opened in template mode).
            token: Rare identifier token, e.g. ``"[V]"``.
            class_noun: Natural-language class noun, e.g. ``"dog"``.

        Returns:
            A formatted caption string, e.g. ``"a photo of a [V] dog"``.
        """
        template = self._rng.choice(self.templates)
        caption = template.format(token=token, class_noun=class_noun)
        logger.debug("Caption for %s: %s", Path(image_path).name, caption)
        return caption

    def generate_deterministic_caption(
        self,
        image_index: int,
        token: str,
        class_noun: str,
    ) -> str:
        """Generate a caption by cycling through templates deterministically.

        Useful when you want each image in a small set (3-5 images) to get a
        distinct template without random overlap.

        Args:
            image_index: Zero-based index of the image in the dataset.
            token: Rare identifier token.
            class_noun: Class noun.

        Returns:
            Formatted caption string.
        """
        template = self.templates[image_index % len(self.templates)]
        return template.format(token=token, class_noun=class_noun)

    # ------------------------------------------------------------------
    # LLM-based captioning (placeholder)
    # ------------------------------------------------------------------

    def generate_llm_caption(
        self,
        image_path: str | Path,
        token: str,
        class_noun: str,
        model: str = "llava",
        prompt_template: str | None = None,
    ) -> str:
        """Generate an informative caption using a vision-language model.

        This is a placeholder for future integration with models such as
        LLaVA, GPT-4V, or Qwen-VL.  The expected workflow is:

        1. Load the image from *image_path*.
        2. Send it to the VLM with a system prompt that instructs the model to
           describe the main subject, referencing it as *token* *class_noun*.
        3. Return the generated description.

        The *prompt_template* parameter can customise the system instruction;
        it should contain ``{token}`` and ``{class_noun}`` placeholders.

        Args:
            image_path: Path to the image to caption.
            token: Rare identifier token.
            class_noun: Class noun.
            model: VLM model identifier (e.g. ``"llava"``, ``"gpt-4v"``).
            prompt_template: Optional custom instruction template.

        Returns:
            Generated caption string.

        Raises:
            NotImplementedError: Always, until VLM integration is complete.
        """
        if prompt_template is None:
            prompt_template = (
                "Describe this image in detail, focusing on the main subject. "
                "The subject is referred to as {token} {class_noun}."
            )
        instruction = prompt_template.format(token=token, class_noun=class_noun)

        raise NotImplementedError(
            f"LLM captioning with model={model!r} is not yet implemented. "
            f"Planned instruction: {instruction!r}. "
            "Install the required VLM package and provide the integration here."
        )

    # ------------------------------------------------------------------
    # Batch captioning
    # ------------------------------------------------------------------

    def caption_dataset(
        self,
        images_dir: str | Path,
        token: str,
        class_noun: str,
        deterministic: bool = True,
    ) -> list[str]:
        """Generate captions for all images in a directory.

        Images are sorted alphabetically by filename to ensure a stable
        ordering that matches :class:`DreamBoothDataset`.

        Args:
            images_dir: Directory containing subject images.
            token: Rare identifier token.
            class_noun: Class noun.
            deterministic: If ``True`` (default), cycle templates in order
                so every image gets a distinct style.  If ``False``, select
                templates randomly.

        Returns:
            List of caption strings, one per image, in filename-sorted order.

        Raises:
            FileNotFoundError: If *images_dir* does not exist or contains no
                images.
        """
        src_dir = Path(images_dir)
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {src_dir}")

        image_paths = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {src_dir}")

        captions: list[str] = []
        for idx, img_path in enumerate(image_paths):
            if deterministic:
                caption = self.generate_deterministic_caption(idx, token, class_noun)
            else:
                caption = self.generate_caption(img_path, token, class_noun)
            captions.append(caption)

        logger.info(
            "Generated %d captions for images in %s (deterministic=%s)",
            len(captions),
            src_dir,
            deterministic,
        )
        return captions

    # ------------------------------------------------------------------
    # Config-driven factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> "SubjectCaptioner":
        """Create a captioner from an OmegaConf config object.

        Reads ``cfg.training.seed`` for reproducibility.

        Args:
            cfg: OmegaConf DictConfig (see ``configs/default.yaml``).

        Returns:
            Configured ``SubjectCaptioner`` instance.
        """
        seed = getattr(cfg.training, "seed", 42)
        return cls(seed=seed)
