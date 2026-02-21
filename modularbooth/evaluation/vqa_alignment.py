"""VQA-based prompt alignment metric.

Evaluates how well a generated image matches its text prompt by decomposing the
prompt into a set of binary yes/no questions and using CLIP text-image similarity
as a proxy for VQA answering.  Each question is converted to a CLIP text query
and matched against the image; the alignment score is the fraction of questions
whose CLIP similarity exceeds a threshold.

The question-generation logic decomposes prompts into element-level questions
that can also be fed to an external VLM evaluation pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Union

import torch
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

    Decomposes a text prompt into binary yes/no questions and uses CLIP
    text-image similarity as a proxy to verify whether each element is
    present in the generated image.  The alignment score is the fraction
    of questions whose CLIP similarity exceeds a configurable threshold.

    Args:
        clip_model: OpenCLIP model name (default: ``"ViT-B-32"``).
        pretrained: OpenCLIP pretrained weights (default: ``"laion2b_s34b_b79k"``).
        device: Torch device string.
        threshold: CLIP cosine similarity threshold above which a question
            is considered "answered yes".
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        threshold: float = 0.22,
    ) -> None:
        self._clip_model_name = clip_model
        self._pretrained = pretrained
        self.device = device
        self.threshold = threshold
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        """Lazily load the CLIP model on first use."""
        if self._model is not None:
            return

        import open_clip

        logger.info("Loading OpenCLIP model %s (%s)", self._clip_model_name, self._pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name, pretrained=self._pretrained
        )
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._clip_model_name)

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
    # Alignment scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_alignment(self, image: ImageInput, prompt: str) -> float:
        """Compute VQA-based alignment between an image and a prompt.

        Decomposes the prompt into binary questions, then uses CLIP text-image
        similarity as a proxy for VQA.  Each question is converted to a
        statement (e.g. "Is there a dog?" -> "a photo of a dog") and compared
        against the image.  The score is the fraction of questions whose
        similarity exceeds ``self.threshold``.

        Args:
            image: The generated image.
            prompt: The text prompt used to generate the image.

        Returns:
            Alignment score in ``[0, 1]``.
        """
        self._ensure_model()

        questions = self.generate_questions(prompt)
        if not questions:
            return 0.0

        # Convert questions to CLIP-friendly statements.
        statements = [self._question_to_statement(q) for q in questions]

        # Encode image.
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        image_features = self._model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode text statements.
        tokens = self._tokenizer(statements).to(self.device)
        text_features = self._model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarities.
        similarities = (image_features @ text_features.T).squeeze(0)

        # Score = fraction of questions above threshold.
        num_yes = (similarities >= self.threshold).sum().item()
        score = num_yes / len(questions)

        logger.debug(
            "VQA alignment for '%s': %.3f (%d/%d questions above threshold %.2f)",
            prompt[:60], score, int(num_yes), len(questions), self.threshold,
        )
        return score

    @staticmethod
    def _question_to_statement(question: str) -> str:
        """Convert a yes/no question to a CLIP-friendly statement.

        Examples:
            "Is there a dog?" -> "a photo with a dog"
            "Is the hat red?" -> "a photo where the hat is red"
            "Does the image show a beach scene?" -> "a photo of a beach scene"
        """
        q = question.rstrip("?").strip()

        if q.lower().startswith("is there "):
            return "a photo with " + q[9:]
        if q.lower().startswith("does the image show "):
            return "a photo of " + q[20:]
        if q.lower().startswith("is the image in "):
            return "a photo in " + q[16:]
        if q.lower().startswith("is the "):
            return "a photo where the " + q[7:]
        if q.lower().startswith("is "):
            return "a photo where " + q[3:]
        if q.lower().startswith("does "):
            return "a photo where " + q[5:]

        return "a photo of " + q

    def compute_batch_alignment(
        self,
        images: list[ImageInput],
        prompts: list[str],
    ) -> float:
        """Compute mean VQA alignment across a batch.

        Args:
            images: List of generated images.
            prompts: Corresponding text prompts (same length as ``images``).

        Returns:
            Mean alignment score in ``[0, 1]``.

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
