"""VQA-based prompt alignment metric.

Evaluates how well a generated image matches its text prompt by decomposing the
prompt into a set of binary yes/no questions and (eventually) using a
Vision-Language Model (VLM) to answer them.

Currently the VLM integration is a placeholder -- :meth:`compute_alignment`
always returns ``0.0``.  The question-generation logic is fully functional
and can be used to prepare inputs for an external VLM evaluation pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Union

from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image]

# Prepositions and articles to strip when extracting noun phrases.
_STOPWORDS: set[str] = {
    "a", "an", "the", "in", "on", "at", "with", "by", "of", "to", "and",
    "is", "are", "was", "were", "it", "its", "this", "that", "from",
    "for", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "over", "some", "each", "every", "both",
    "few", "many", "much", "very", "quite", "really", "just", "also",
    "there", "here", "where", "when", "while", "then", "so", "but", "or",
    "if", "not", "no", "all", "any", "only", "own", "same", "other",
}

# Simple colour words for more targeted questions.
_COLOUR_WORDS: set[str] = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
    "black", "brown", "gray", "grey", "golden", "silver",
}

# Common scene / location keywords for context questions.
_SCENE_KEYWORDS: set[str] = {
    "beach", "forest", "mountain", "city", "street", "garden", "park",
    "room", "kitchen", "studio", "desert", "ocean", "lake", "field",
    "snow", "rain", "sunset", "sunrise", "night", "underwater",
    "space", "sky", "castle", "library", "museum", "cafe", "restaurant",
}

# Style descriptors.
_STYLE_KEYWORDS: set[str] = {
    "realistic", "cartoon", "anime", "watercolor", "oil painting",
    "sketch", "photograph", "cinematic", "dramatic", "minimalist",
    "abstract", "surreal", "vintage", "retro", "modern", "futuristic",
    "3d render", "pixel art", "digital art", "pencil drawing",
}


class VQAAlignment:
    """VQA-based prompt alignment scorer.

    Decomposes a text prompt into binary yes/no questions and (in the future)
    queries a VLM to verify whether each element is present in the generated
    image.

    .. note::

        The VLM inference step is currently a **placeholder**.  Call
        :meth:`generate_questions` to obtain the question list and feed it to
        your own VQA pipeline externally.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_noun_phrases(prompt: str) -> list[str]:
        """Extract candidate noun phrases from a prompt using simple heuristics.

        This is *not* a full NLP parser -- it splits on common delimiters and
        filters stop-words to produce reasonable keyword groups.

        Args:
            prompt: The text prompt.

        Returns:
            Deduplicated list of extracted noun phrases (lower-cased).
        """
        # Normalise whitespace and lowercase.
        text = re.sub(r"\s+", " ", prompt.lower().strip())

        # Split on commas and common conjunctions to get clauses.
        clauses = re.split(r",\s*|\band\b|\bwith\b|\bin\b|\bon\b|\bat\b", text)

        phrases: list[str] = []
        seen: set[str] = set()

        for clause in clauses:
            # Remove leading/trailing stop-words and whitespace.
            words = clause.strip().split()
            words = [w for w in words if w not in _STOPWORDS]
            if words:
                phrase = " ".join(words)
                if phrase not in seen:
                    phrases.append(phrase)
                    seen.add(phrase)

        return phrases

    @staticmethod
    def _detect_colours(prompt: str) -> list[tuple[str, str]]:
        """Detect colour-noun associations in the prompt.

        Looks for patterns like "red hat", "blue car", etc.

        Args:
            prompt: The text prompt (will be lower-cased internally).

        Returns:
            List of ``(colour, noun)`` tuples found.
        """
        text = prompt.lower()
        pairs: list[tuple[str, str]] = []
        for colour in _COLOUR_WORDS:
            pattern = rf"\b{colour}\s+(\w+)"
            matches = re.findall(pattern, text)
            for noun in matches:
                if noun not in _STOPWORDS:
                    pairs.append((colour, noun))
        return pairs

    @staticmethod
    def _detect_scenes(prompt: str) -> list[str]:
        """Detect scene / location keywords in the prompt.

        Args:
            prompt: The text prompt.

        Returns:
            List of detected scene keywords.
        """
        text = prompt.lower()
        return [kw for kw in _SCENE_KEYWORDS if kw in text]

    @staticmethod
    def _detect_styles(prompt: str) -> list[str]:
        """Detect artistic style descriptors in the prompt.

        Args:
            prompt: The text prompt.

        Returns:
            List of detected style keywords.
        """
        text = prompt.lower()
        return [s for s in _STYLE_KEYWORDS if s in text]

    def generate_questions(self, prompt: str) -> list[str]:
        """Parse a text prompt into a list of binary yes/no questions.

        Each question targets a single visual element that should be verifiable
        by a VLM.

        Examples::

            >>> vqa = VQAAlignment()
            >>> vqa.generate_questions("a dog on a beach wearing a red hat")
            ['Is there a dog?',
             'Is there a beach?',
             'Is there a red hat?',
             'Is the hat red?',
             'Does the image show a beach scene?']

        Args:
            prompt: The text prompt to decompose.

        Returns:
            Deduplicated list of yes/no question strings.
        """
        questions: list[str] = []
        seen: set[str] = set()

        def _add(q: str) -> None:
            q_lower = q.lower()
            if q_lower not in seen:
                questions.append(q)
                seen.add(q_lower)

        # 1. Noun-phrase presence questions.
        for phrase in self._extract_noun_phrases(prompt):
            _add(f"Is there a {phrase}?")

        # 2. Colour-attribute questions.
        for colour, noun in self._detect_colours(prompt):
            _add(f"Is there a {colour} {noun}?")
            _add(f"Is the {noun} {colour}?")

        # 3. Scene / location questions.
        for scene in self._detect_scenes(prompt):
            _add(f"Does the image show a {scene} scene?")

        # 4. Style questions.
        for style in self._detect_styles(prompt):
            _add(f"Is the image in a {style} style?")

        if not questions:
            # Fallback: ask about the prompt as a whole.
            _add(f'Does the image match the description "{prompt}"?')

        return questions

    # ------------------------------------------------------------------
    # Alignment scoring (placeholder)
    # ------------------------------------------------------------------

    def compute_alignment(self, image: ImageInput, prompt: str) -> float:
        """Compute VQA-based alignment between an image and a prompt.

        .. warning::

            This method is a **placeholder**.  It generates the binary
            questions but does not yet run a VLM to answer them.  VLM
            integration is planned for a future release.  The method
            always returns ``0.0``.

        To use this metric today, call :meth:`generate_questions` and feed
        the resulting question list to your own VLM pipeline, then compute the
        fraction of "yes" answers.

        Args:
            image: The generated image (currently unused).
            prompt: The text prompt used to generate the image.

        Returns:
            Alignment score in ``[0, 1]`` -- currently always ``0.0``.
        """
        questions = self.generate_questions(prompt)
        logger.warning(
            "VQAAlignment.compute_alignment is a placeholder and always returns 0.0. "
            "Generated %d questions for prompt: '%s'. "
            "Integrate a VLM (e.g. LLaVA, InstructBLIP) to answer these questions.",
            len(questions),
            prompt[:80],
        )
        # TODO: Integrate a VLM to answer the generated questions and compute
        # the fraction of affirmative responses.
        return 0.0

    def compute_batch_alignment(
        self,
        images: list[ImageInput],
        prompts: list[str],
    ) -> float:
        """Compute mean VQA alignment across a batch (placeholder).

        Args:
            images: List of generated images.
            prompts: Corresponding text prompts (same length as ``images``).

        Returns:
            Mean alignment score -- currently always ``0.0``.

        Raises:
            ValueError: If the number of images and prompts do not match.
        """
        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match "
                f"the number of prompts ({len(prompts)})."
            )

        scores = [self.compute_alignment(img, p) for img, p in zip(images, prompts)]
        return sum(scores) / len(scores) if scores else 0.0
