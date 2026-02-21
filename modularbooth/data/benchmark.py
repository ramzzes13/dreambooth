"""DreamBooth evaluation benchmarks for single- and multi-subject personalisation.

This module provides structured access to the standard DreamBooth benchmark
(30 subjects, 25 prompt templates) as well as extensions for multi-subject
evaluation and context-appearance entanglement probing.

Reference:
    Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models
    for Subject-Driven Generation", CVPR 2023.
"""

from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard DreamBooth benchmark prompt templates (25 prompts)
# ---------------------------------------------------------------------------

# These templates follow the original DreamBooth paper's evaluation protocol.
# Each contains a ``{subject}`` placeholder that is replaced with
# ``"a {token} {class_noun}"`` at evaluation time.
_DREAMBOOTH_PROMPT_TEMPLATES: list[str] = [
    "{subject} in the jungle",
    "{subject} in the snow",
    "{subject} on the beach",
    "{subject} on a cobblestone street",
    "{subject} on top of pink fabric",
    "{subject} on top of a wooden floor",
    "{subject} with a city in the background",
    "{subject} with a mountain in the background",
    "{subject} with a blue house in the background",
    "{subject} on top of a purple rug in a forest",
    "{subject} wearing a red hat",
    "{subject} wearing a santa hat",
    "{subject} wearing a rainbow scarf",
    "{subject} wearing a black top hat and a monocle",
    "{subject} in a chef outfit",
    "{subject} in a firefighter outfit",
    "{subject} in a police outfit",
    "{subject} wearing pink glasses",
    "{subject} wearing a yellow shirt",
    "{subject} in a purple wizard outfit",
    "a painting of {subject} in the style of Monet",
    "a painting of {subject} in the style of Van Gogh",
    "a statue of {subject}",
    "a watercolour painting of {subject}",
    "an oil painting of {subject}",
]

# Standard DreamBooth benchmark subjects (30 subjects: 21 live + 9 object).
# Each entry: (subject_id, class_noun, category).
_DREAMBOOTH_SUBJECTS: list[tuple[str, str, str]] = [
    # Live subjects
    ("backpack", "backpack", "object"),
    ("backpack_dog", "dog", "live"),
    ("bear_plushie", "stuffed animal", "object"),
    ("berry_bowl", "bowl", "object"),
    ("can", "can", "object"),
    ("candle", "candle", "object"),
    ("cat", "cat", "live"),
    ("cat2", "cat", "live"),
    ("clock", "clock", "object"),
    ("colorful_sneaker", "sneaker", "object"),
    ("dog", "dog", "live"),
    ("dog2", "dog", "live"),
    ("dog3", "dog", "live"),
    ("dog5", "dog", "live"),
    ("dog6", "dog", "live"),
    ("dog7", "dog", "live"),
    ("dog8", "dog", "live"),
    ("duck_toy", "toy", "object"),
    ("fancy_boot", "boot", "object"),
    ("grey_sloth_plushie", "stuffed animal", "object"),
    ("monster_toy", "toy", "object"),
    ("pink_sunglasses", "glasses", "object"),
    ("poop_emoji", "toy", "object"),
    ("rc_car", "toy", "object"),
    ("red_cartoon", "cartoon", "object"),
    ("robot_toy", "toy", "object"),
    ("shiny_sneaker", "sneaker", "object"),
    ("teapot", "teapot", "object"),
    ("vase", "vase", "object"),
    ("wolf_plushie", "stuffed animal", "object"),
]


# ---------------------------------------------------------------------------
# Entanglement probing prompts
# ---------------------------------------------------------------------------

# These prompts are specifically designed to trigger context-appearance
# entanglement -- the failure mode where a fine-tuned model leaks training
# background / context into generated images even when the prompt specifies
# a novel context.  High visual similarity between outputs of these prompts
# and the training images' *backgrounds* signals entanglement.
_ENTANGLEMENT_PROBE_TEMPLATES: list[str] = [
    "{subject} in a completely white room",
    "{subject} floating in outer space",
    "{subject} on a bright red surface",
    "{subject} underwater surrounded by fish",
    "{subject} on the surface of the moon",
    "{subject} inside a medieval castle",
    "{subject} in a neon-lit cyberpunk city",
    "{subject} in a dense rainforest",
    "{subject} on an ice rink",
    "{subject} in front of a plain grey wall",
    "{subject} in an empty desert at sunset",
    "{subject} on a kitchen counter",
    "{subject} in a children's playroom with colourful toys",
    "{subject} inside an art gallery with white walls",
    "{subject} on a bed of autumn leaves",
]


# ---------------------------------------------------------------------------
# Multi-subject prompt templates
# ---------------------------------------------------------------------------

_MULTI_SUBJECT_PAIR_TEMPLATES: list[str] = [
    "{subject_a} and {subject_b} sitting together",
    "{subject_a} next to {subject_b} on a table",
    "{subject_a} and {subject_b} in the park",
    "{subject_a} and {subject_b} on the beach at sunset",
    "{subject_a} with {subject_b} in front of a fireplace",
    "a photo of {subject_a} and {subject_b}",
    "{subject_a} and {subject_b} in a garden",
    "{subject_a} beside {subject_b} on a white background",
    "{subject_a} and {subject_b} wearing party hats",
    "{subject_a} and {subject_b} in a studio photo",
]

_MULTI_SUBJECT_TRIPLE_TEMPLATES: list[str] = [
    "{subject_a}, {subject_b}, and {subject_c} together",
    "{subject_a}, {subject_b}, and {subject_c} in the park",
    "a photo of {subject_a}, {subject_b}, and {subject_c}",
    "{subject_a}, {subject_b}, and {subject_c} on a table",
    "{subject_a} with {subject_b} and {subject_c} in a studio",
    "{subject_a}, {subject_b}, and {subject_c} at the beach",
]


# ===================================================================
# DreamBoothBenchmark
# ===================================================================

class DreamBoothBenchmark:
    """Access to the standard DreamBooth evaluation benchmark.

    Provides the 30 canonical subjects and 25 prompt templates from the
    DreamBooth paper, plus entanglement-probing prompts designed to detect
    context-appearance leakage after fine-tuning.

    Args:
        dataset_root: Optional path to the DreamBooth dataset on disk.
            When provided, :meth:`get_subject_images` can return actual
            image paths.  The expected layout is::

                dataset_root/
                    backpack/
                        00.jpg
                        01.jpg
                        ...
                    cat/
                        ...

        seed: Random seed for any stochastic prompt sampling.

    Example::

        bench = DreamBoothBenchmark("./dreambooth_dataset")
        prompts = bench.get_evaluation_prompts({"[V]": "dog"})
    """

    # Class-level constants exposed for external use.
    SUBJECTS = _DREAMBOOTH_SUBJECTS
    PROMPT_TEMPLATES = _DREAMBOOTH_PROMPT_TEMPLATES

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self._rng = random.Random(seed)

        # Build look-up structures.
        self._subject_map: dict[str, tuple[str, str]] = {
            sid: (class_noun, category)
            for sid, class_noun, category in _DREAMBOOTH_SUBJECTS
        }

    # ------------------------------------------------------------------
    # Subject queries
    # ------------------------------------------------------------------

    def list_subjects(
        self,
        category: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """List benchmark subjects, optionally filtered by category.

        Args:
            category: ``"live"``, ``"object"``, or ``None`` for all.

        Returns:
            List of ``(subject_id, class_noun, category)`` tuples.
        """
        if category is None:
            return list(_DREAMBOOTH_SUBJECTS)
        return [
            (sid, cn, cat) for sid, cn, cat in _DREAMBOOTH_SUBJECTS
            if cat == category
        ]

    def get_subject_images(self, subject_id: str) -> list[Path]:
        """Return image paths for a benchmark subject.

        Requires *dataset_root* to be set at construction time.

        Args:
            subject_id: One of the 30 canonical subject identifiers.

        Returns:
            Sorted list of image paths.

        Raises:
            ValueError: If *dataset_root* was not provided.
            FileNotFoundError: If the subject directory does not exist.
        """
        if self.dataset_root is None:
            raise ValueError(
                "dataset_root was not provided; cannot resolve image paths."
            )
        subject_dir = self.dataset_root / subject_id
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        from modularbooth.data.dataset import IMAGE_EXTENSIONS
        return sorted(
            p for p in subject_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )

    # ------------------------------------------------------------------
    # Evaluation prompts
    # ------------------------------------------------------------------

    def get_evaluation_prompts(
        self,
        subject_tokens: dict[str, str],
        num_subjects: int = 1,
    ) -> list[str]:
        """Generate evaluation prompts for one or more subjects.

        For single-subject evaluation (``num_subjects=1``), each subject is
        paired with all 25 standard prompt templates.  When multiple
        subject tokens are provided and ``num_subjects > 1``, this delegates
        to multi-subject prompt generation (see :class:`MultiSubjectBenchmark`).

        Args:
            subject_tokens: Mapping from rare token to class noun, e.g.
                ``{"[V]": "dog"}`` or ``{"[V1]": "dog", "[V2]": "cat"}``.
            num_subjects: Number of subjects to compose per prompt.
                1 = standard single-subject evaluation.

        Returns:
            List of fully-formatted prompt strings.
        """
        if num_subjects == 1:
            return self._single_subject_prompts(subject_tokens)

        # Multi-subject: delegate to MultiSubjectBenchmark.
        multi = MultiSubjectBenchmark(seed=self._rng.randint(0, 2**31))
        return multi.generate_prompts(subject_tokens, num_subjects=num_subjects)

    def _single_subject_prompts(
        self,
        subject_tokens: dict[str, str],
    ) -> list[str]:
        """Generate all 25 prompts for each subject token.

        Args:
            subject_tokens: Token-to-class_noun mapping.

        Returns:
            List of formatted prompt strings.
        """
        prompts: list[str] = []
        for token, class_noun in subject_tokens.items():
            subject_str = f"a {token} {class_noun}"
            for template in _DREAMBOOTH_PROMPT_TEMPLATES:
                prompts.append(template.format(subject=subject_str))
        return prompts

    # ------------------------------------------------------------------
    # Entanglement probing
    # ------------------------------------------------------------------

    def get_entanglement_probing_prompts(
        self,
        token: str,
        class_noun: str,
    ) -> list[str]:
        """Generate prompts that probe for context-appearance entanglement.

        These prompts place the subject in contexts that are maximally
        different from typical training backgrounds.  If the generated
        images still resemble training backgrounds, the model suffers from
        context-appearance entanglement.

        Args:
            token: Rare identifier token, e.g. ``"[V]"``.
            class_noun: Class noun, e.g. ``"dog"``.

        Returns:
            List of probing prompt strings.
        """
        subject_str = f"a {token} {class_noun}"
        return [
            template.format(subject=subject_str)
            for template in _ENTANGLEMENT_PROBE_TEMPLATES
        ]


# ===================================================================
# MultiSubjectBenchmark
# ===================================================================

class MultiSubjectBenchmark:
    """Generate evaluation prompts for multi-subject personalisation.

    Given a set of subject tokens, this class constructs prompts that
    reference pairs or triples of subjects in the same scene -- the core
    evaluation scenario for ModularBooth's modular LoRA approach.

    Args:
        pair_templates: Custom templates for 2-subject prompts.  Must
            contain ``{subject_a}`` and ``{subject_b}`` placeholders.
        triple_templates: Custom templates for 3-subject prompts.  Must
            contain ``{subject_a}``, ``{subject_b}``, and ``{subject_c}``.
        seed: Random seed for combinatorial sampling.

    Example::

        bench = MultiSubjectBenchmark()
        tokens = {"[V1]": "dog", "[V2]": "cat", "[V3]": "backpack"}
        prompts = bench.generate_prompts(tokens, num_subjects=2)
    """

    def __init__(
        self,
        pair_templates: list[str] | None = None,
        triple_templates: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.pair_templates = (
            list(pair_templates) if pair_templates is not None
            else list(_MULTI_SUBJECT_PAIR_TEMPLATES)
        )
        self.triple_templates = (
            list(triple_templates) if triple_templates is not None
            else list(_MULTI_SUBJECT_TRIPLE_TEMPLATES)
        )
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def generate_prompts(
        self,
        subject_tokens: dict[str, str],
        num_subjects: int = 2,
        max_combinations: int | None = None,
    ) -> list[str]:
        """Generate multi-subject prompts from all combinations of subjects.

        Args:
            subject_tokens: Mapping from rare token to class noun, e.g.
                ``{"[V1]": "dog", "[V2]": "cat"}``.
            num_subjects: Number of subjects per prompt (2 or 3).
            max_combinations: If set, randomly sample at most this many
                subject combinations (useful when the number of subjects is
                large).

        Returns:
            List of formatted prompt strings.

        Raises:
            ValueError: If *num_subjects* is not 2 or 3, or if there are
                fewer subjects than *num_subjects*.
        """
        if num_subjects not in (2, 3):
            raise ValueError(
                f"num_subjects must be 2 or 3, got {num_subjects}."
            )

        tokens_list = list(subject_tokens.items())  # [(token, class_noun), ...]
        if len(tokens_list) < num_subjects:
            raise ValueError(
                f"Need at least {num_subjects} subjects, "
                f"got {len(tokens_list)}."
            )

        combinations = list(itertools.combinations(tokens_list, num_subjects))
        if max_combinations is not None and len(combinations) > max_combinations:
            combinations = self._rng.sample(combinations, max_combinations)

        if num_subjects == 2:
            return self._pair_prompts(combinations)
        return self._triple_prompts(combinations)

    def _pair_prompts(
        self,
        combinations: list[tuple[tuple[str, str], ...]],
    ) -> list[str]:
        """Generate prompts for all subject pairs across all templates.

        Args:
            combinations: List of 2-element tuples of ``(token, class_noun)``.

        Returns:
            Formatted prompt strings.
        """
        prompts: list[str] = []
        for combo in combinations:
            (tok_a, noun_a), (tok_b, noun_b) = combo[0], combo[1]
            subj_a = f"a {tok_a} {noun_a}"
            subj_b = f"a {tok_b} {noun_b}"
            for template in self.pair_templates:
                prompts.append(
                    template.format(subject_a=subj_a, subject_b=subj_b)
                )
        return prompts

    def _triple_prompts(
        self,
        combinations: list[tuple[tuple[str, str], ...]],
    ) -> list[str]:
        """Generate prompts for all subject triples across all templates.

        Args:
            combinations: List of 3-element tuples of ``(token, class_noun)``.

        Returns:
            Formatted prompt strings.
        """
        prompts: list[str] = []
        for combo in combinations:
            (tok_a, noun_a) = combo[0]
            (tok_b, noun_b) = combo[1]
            (tok_c, noun_c) = combo[2]
            subj_a = f"a {tok_a} {noun_a}"
            subj_b = f"a {tok_b} {noun_b}"
            subj_c = f"a {tok_c} {noun_c}"
            for template in self.triple_templates:
                prompts.append(
                    template.format(
                        subject_a=subj_a,
                        subject_b=subj_b,
                        subject_c=subj_c,
                    )
                )
        return prompts

    # ------------------------------------------------------------------
    # Entanglement cross-subject probing
    # ------------------------------------------------------------------

    def get_cross_subject_entanglement_prompts(
        self,
        subject_tokens: dict[str, str],
    ) -> list[str]:
        """Generate prompts that probe cross-subject entanglement.

        In multi-subject personalisation, a common failure is *identity
        leakage* where attributes of one subject bleed into another.
        These prompts request only one subject at a time but in contexts
        associated with the *other* subjects to test whether leakage occurs.

        Args:
            subject_tokens: Mapping from token to class noun.

        Returns:
            List of probing prompt strings.
        """
        prompts: list[str] = []
        tokens_list = list(subject_tokens.items())

        # For each subject, generate prompts that mention it alone but in
        # contexts that reference the *class noun* of other subjects
        # (without the rare token).
        for i, (token, class_noun) in enumerate(tokens_list):
            subject_str = f"a {token} {class_noun}"
            for j, (_, other_noun) in enumerate(tokens_list):
                if i == j:
                    continue
                prompts.append(
                    f"{subject_str} that looks nothing like a {other_noun}"
                )
                prompts.append(
                    f"{subject_str} in a scene with a regular {other_noun}"
                )

        return prompts
