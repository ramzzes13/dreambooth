"""End-to-end evaluation pipeline for ModularBooth.

Orchestrates all evaluation metrics -- DINO subject-fidelity, CLIP-I / CLIP-T
alignment, LPIPS diversity, Context-Appearance Entanglement (CAE), Identity
Isolation Score (IIS), and VQA alignment -- into a single pipeline that can
evaluate both single-subject and multi-subject personalisation results.

Typical usage::

    from omegaconf import OmegaConf
    from modularbooth.evaluation.run_evaluation import EvaluationPipeline

    cfg = OmegaConf.load("configs/default.yaml")
    pipeline = EvaluationPipeline(cfg, device="cuda")
    report = pipeline.run_full_evaluation("outputs/experiment_01")
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from modularbooth.evaluation.clip_score import CLIPScore
from modularbooth.evaluation.dino_score import DINOScore, DINOv2Score
from modularbooth.evaluation.diversity import LPIPSDiversity
from modularbooth.evaluation.entanglement import ContextAppearanceEntanglement
from modularbooth.evaluation.identity_isolation import IdentityIsolationScore
from modularbooth.evaluation.vqa_alignment import VQAAlignment

logger = logging.getLogger(__name__)

# Supported image extensions (mirrors the dataset module).
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

ImageInput = Union[Image.Image]

# Type alias for bounding-box coordinates: (x1, y1, x2, y2).
BBox = tuple[int, int, int, int]


def _collect_images(directory: Path) -> list[Image.Image]:
    """Load all images from a directory in sorted order.

    Args:
        directory: Path to a directory containing image files.

    Returns:
        List of PIL Images (RGB).

    Raises:
        FileNotFoundError: If the directory does not exist or contains no images.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")
    return [Image.open(p).convert("RGB") for p in paths]


def _config_hash(cfg: DictConfig) -> str:
    """Compute a short SHA-256 hash of the config for identification.

    Args:
        cfg: OmegaConf DictConfig.

    Returns:
        First 12 hex characters of the SHA-256 hash.
    """
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(yaml_str.encode()).hexdigest()[:12]


class EvaluationPipeline:
    """End-to-end evaluation pipeline for ModularBooth experiments.

    All metric instances are created lazily on first use so that construction
    is cheap and metrics that are not needed are never loaded.

    Args:
        config: An OmegaConf ``DictConfig`` (typically loaded from
            ``configs/default.yaml`` merged with experiment overrides).
        device: Torch device string.
    """

    def __init__(self, config: DictConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device

        # Lazily initialised metric instances.
        self._dino_scorer: DINOScore | None = None
        self._dinov2_scorer: DINOv2Score | None = None
        self._clip_scorer: CLIPScore | None = None
        self._lpips_diversity: LPIPSDiversity | None = None
        self._cae_scorer: ContextAppearanceEntanglement | None = None
        self._iis_scorer: IdentityIsolationScore | None = None
        self._vqa_scorer: VQAAlignment | None = None

    # ------------------------------------------------------------------
    # Lazy metric accessors
    # ------------------------------------------------------------------

    @property
    def dino_scorer(self) -> DINOScore:
        """Lazily initialised DINO scorer."""
        if self._dino_scorer is None:
            model_name = OmegaConf.select(
                self.config, "evaluation.dino_model", default="facebook/dino-vits16"
            )
            self._dino_scorer = DINOScore(model_name=model_name, device=self.device)
        return self._dino_scorer

    @property
    def dinov2_scorer(self) -> DINOv2Score:
        """Lazily initialised DINOv2 scorer."""
        if self._dinov2_scorer is None:
            model_name = OmegaConf.select(
                self.config, "evaluation.dinov2_model", default="facebook/dinov2-vitb14"
            )
            self._dinov2_scorer = DINOv2Score(model_name=model_name, device=self.device)
        return self._dinov2_scorer

    @property
    def clip_scorer(self) -> CLIPScore:
        """Lazily initialised CLIP scorer."""
        if self._clip_scorer is None:
            clip_cfg = OmegaConf.select(
                self.config, "evaluation.clip_model", default="openai/clip-vit-large-patch14"
            )
            # Parse "org/model" into open_clip's model_name + pretrained.
            if "/" in str(clip_cfg):
                pretrained = str(clip_cfg).split("/")[0]
                model_arch = "ViT-L-14"  # default for openai/clip-vit-large-patch14
            else:
                pretrained = "openai"
                model_arch = str(clip_cfg)

            self._clip_scorer = CLIPScore(
                model_name=model_arch, pretrained=pretrained, device=self.device
            )
        return self._clip_scorer

    @property
    def lpips_diversity(self) -> LPIPSDiversity:
        """Lazily initialised LPIPS diversity scorer."""
        if self._lpips_diversity is None:
            self._lpips_diversity = LPIPSDiversity(net="alex", device=self.device)
        return self._lpips_diversity

    @property
    def cae_scorer(self) -> ContextAppearanceEntanglement:
        """Lazily initialised Context-Appearance Entanglement scorer."""
        if self._cae_scorer is None:
            self._cae_scorer = ContextAppearanceEntanglement(
                dino_scorer=self.dino_scorer
            )
        return self._cae_scorer

    @property
    def iis_scorer(self) -> IdentityIsolationScore:
        """Lazily initialised Identity Isolation Score scorer."""
        if self._iis_scorer is None:
            self._iis_scorer = IdentityIsolationScore(
                dino_scorer=self.dino_scorer, device=self.device
            )
        return self._iis_scorer

    @property
    def vqa_scorer(self) -> VQAAlignment:
        """Lazily initialised VQA alignment scorer."""
        if self._vqa_scorer is None:
            self._vqa_scorer = VQAAlignment()
        return self._vqa_scorer

    # ------------------------------------------------------------------
    # Single-subject evaluation
    # ------------------------------------------------------------------

    def evaluate_single_subject(
        self,
        generated_dir: str | Path,
        reference_dir: str | Path,
        prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all single-subject metrics on generated images.

        Computes:

        * **dino_score** -- DINO subject-fidelity.
        * **clip_i_score** -- CLIP image-image similarity.
        * **clip_t_score** -- CLIP text-image alignment (if prompts given).
        * **lpips_diversity** -- Output diversity via LPIPS.
        * **cae** -- Context-Appearance Entanglement.
        * **vqa_alignment** -- VQA prompt alignment (placeholder).

        Args:
            generated_dir: Directory of generated images.
            reference_dir: Directory of reference subject images.
            prompts: Optional list of prompts aligned with generated images.
                Required for CLIP-T and VQA scores.

        Returns:
            Dictionary of metric name to score.
        """
        gen_dir = Path(generated_dir)
        ref_dir = Path(reference_dir)

        generated_images = _collect_images(gen_dir)
        reference_images = _collect_images(ref_dir)

        logger.info(
            "Evaluating single subject: %d generated, %d reference images",
            len(generated_images), len(reference_images),
        )

        results: dict[str, Any] = {}

        # DINO subject fidelity.
        results["dino_score"] = self.dino_scorer.compute_score(
            generated_images, reference_images
        )

        # CLIP-I (image-image similarity).
        results["clip_i_score"] = self.clip_scorer.clip_i_score(
            generated_images, reference_images
        )

        # CLIP-T (text-image alignment), requires prompts.
        if prompts is not None:
            if len(prompts) == 1 and len(generated_images) > 1:
                # Single prompt for all images -- expand.
                prompts = prompts * len(generated_images)
            if len(prompts) == len(generated_images):
                results["clip_t_score"] = self.clip_scorer.clip_t_score(
                    generated_images, prompts
                )
            else:
                logger.warning(
                    "Skipping CLIP-T: %d prompts vs %d images",
                    len(prompts), len(generated_images),
                )
                results["clip_t_score"] = None
        else:
            results["clip_t_score"] = None

        # LPIPS diversity (all generated images should be from the same prompt).
        if len(generated_images) >= 2:
            results["lpips_diversity"] = self.lpips_diversity.compute_diversity(
                generated_images
            )
        else:
            results["lpips_diversity"] = None

        # CAE (same subject across contexts -- we treat generated images as
        # different contexts of the same subject).
        if len(generated_images) >= 2:
            results["cae"] = self.cae_scorer.compute_cae(generated_images)
        else:
            results["cae"] = None

        # VQA alignment (placeholder).
        if prompts is not None and len(prompts) == len(generated_images):
            results["vqa_alignment"] = self.vqa_scorer.compute_batch_alignment(
                generated_images, prompts
            )
        else:
            results["vqa_alignment"] = None

        logger.info("Single-subject results: %s", results)
        return results

    # ------------------------------------------------------------------
    # Multi-subject evaluation
    # ------------------------------------------------------------------

    def evaluate_multi_subject(
        self,
        generated_dir: str | Path,
        reference_dirs: dict[int, str | Path],
        subject_crops: list[list[BBox]] | None = None,
        prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all multi-subject metrics on generated images.

        In addition to the single-subject metrics (run on the full generated
        images against all reference images combined), this also computes:

        * **iis** -- Identity Isolation Score (requires bounding boxes).
        * Per-subject DINO scores.

        Args:
            generated_dir: Directory of generated multi-subject images.
            reference_dirs: Mapping from subject ID to the directory of
                reference images for that subject.
            subject_crops: Per-image list of bounding boxes for each subject.
                Outer list is aligned with generated images; inner list is
                aligned with subject IDs.  Required for IIS.
            prompts: Optional prompts aligned with generated images.

        Returns:
            Dictionary of metric name to score.
        """
        gen_dir = Path(generated_dir)
        generated_images = _collect_images(gen_dir)

        # Load reference images for each subject.
        ref_images: dict[int, list[Image.Image]] = {}
        all_ref_images: list[Image.Image] = []
        for subject_id, ref_dir in reference_dirs.items():
            refs = _collect_images(Path(ref_dir))
            ref_images[subject_id] = refs
            all_ref_images.extend(refs)

        logger.info(
            "Evaluating multi-subject: %d generated images, %d subjects",
            len(generated_images), len(reference_dirs),
        )

        results: dict[str, Any] = {}

        # Global DINO score (all generated vs all references pooled).
        results["dino_score"] = self.dino_scorer.compute_score(
            generated_images, all_ref_images
        )

        # Per-subject DINO scores.
        per_subject_dino: dict[int, float] = {}
        for subject_id, refs in ref_images.items():
            per_subject_dino[subject_id] = self.dino_scorer.compute_score(
                generated_images, refs
            )
        results["per_subject_dino"] = per_subject_dino

        # CLIP-I against pooled references.
        results["clip_i_score"] = self.clip_scorer.clip_i_score(
            generated_images, all_ref_images
        )

        # CLIP-T.
        if prompts is not None:
            if len(prompts) == 1 and len(generated_images) > 1:
                prompts = prompts * len(generated_images)
            if len(prompts) == len(generated_images):
                results["clip_t_score"] = self.clip_scorer.clip_t_score(
                    generated_images, prompts
                )
            else:
                results["clip_t_score"] = None
        else:
            results["clip_t_score"] = None

        # LPIPS diversity.
        if len(generated_images) >= 2:
            results["lpips_diversity"] = self.lpips_diversity.compute_diversity(
                generated_images
            )
        else:
            results["lpips_diversity"] = None

        # Identity Isolation Score (requires bounding boxes).
        if subject_crops is not None and len(subject_crops) == len(generated_images):
            results["iis"] = self.iis_scorer.compute_batch_iis(
                generated_images, subject_crops, ref_images
            )
        else:
            if subject_crops is None:
                logger.info(
                    "Skipping IIS: no subject bounding boxes provided."
                )
            else:
                logger.warning(
                    "Skipping IIS: %d crop lists vs %d images",
                    len(subject_crops), len(generated_images),
                )
            results["iis"] = None

        # CAE per subject (requires crops for each subject across images).
        if subject_crops is not None and len(subject_crops) == len(generated_images):
            subject_ids = sorted(ref_images.keys())
            subjects_across_contexts: list[list[Image.Image]] = [
                [] for _ in subject_ids
            ]
            for img, crops in zip(generated_images, subject_crops):
                for j, bbox in enumerate(crops):
                    if j < len(subject_ids):
                        x1, y1, x2, y2 = bbox
                        crop = img.crop((x1, y1, x2, y2))
                        subjects_across_contexts[j].append(crop)

            cae_results = self.cae_scorer.compute_batch_cae(subjects_across_contexts)
            results["cae"] = cae_results["mean_cae"]
            results["per_subject_cae"] = dict(
                zip(subject_ids, cae_results["per_subject"])
            )
        else:
            results["cae"] = None
            results["per_subject_cae"] = None

        # VQA alignment (placeholder).
        if prompts is not None and len(prompts) == len(generated_images):
            results["vqa_alignment"] = self.vqa_scorer.compute_batch_alignment(
                generated_images, prompts
            )
        else:
            results["vqa_alignment"] = None

        logger.info("Multi-subject results: %s", results)
        return results

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def run_full_evaluation(self, results_dir: str | Path) -> dict[str, Any]:
        """Run all evaluations on an experiment output directory.

        Expected directory layout::

            results_dir/
                generated/          # generated images
                reference/          # reference subject images (single-subject)
                  OR
                reference_0/        # per-subject reference dirs (multi-subject)
                reference_1/
                ...
                prompts.txt         # one prompt per line (optional)
                subject_crops.json  # bounding boxes (optional, multi-subject)

        Args:
            results_dir: Path to the experiment output directory.

        Returns:
            Dictionary containing all metric scores plus metadata.
        """
        root = Path(results_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Results directory not found: {root}")

        generated_dir = root / "generated"
        if not generated_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'generated/' subdirectory in {root}"
            )

        # Load prompts if available.
        prompts: list[str] | None = None
        prompts_file = root / "prompts.txt"
        if prompts_file.is_file():
            prompts = [
                line.strip()
                for line in prompts_file.read_text().splitlines()
                if line.strip()
            ]
            logger.info("Loaded %d prompts from %s", len(prompts), prompts_file)

        # Detect single-subject vs multi-subject layout.
        single_ref = root / "reference"
        multi_ref_dirs: dict[int, Path] = {}
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith("reference_"):
                try:
                    subject_id = int(d.name.split("_", 1)[1])
                    multi_ref_dirs[subject_id] = d
                except ValueError:
                    pass

        # Load subject crops if available.
        subject_crops: list[list[BBox]] | None = None
        crops_file = root / "subject_crops.json"
        if crops_file.is_file():
            with open(crops_file) as f:
                raw_crops = json.load(f)
            subject_crops = [
                [tuple(bbox) for bbox in image_crops]
                for image_crops in raw_crops
            ]
            logger.info("Loaded subject crops from %s", crops_file)

        # Run evaluation.
        if multi_ref_dirs:
            logger.info(
                "Detected multi-subject layout with %d subjects",
                len(multi_ref_dirs),
            )
            results = self.evaluate_multi_subject(
                generated_dir=generated_dir,
                reference_dirs={sid: str(d) for sid, d in multi_ref_dirs.items()},
                subject_crops=subject_crops,
                prompts=prompts,
            )
        elif single_ref.is_dir():
            logger.info("Detected single-subject layout")
            results = self.evaluate_single_subject(
                generated_dir=generated_dir,
                reference_dir=single_ref,
                prompts=prompts,
            )
        else:
            raise FileNotFoundError(
                f"No reference directory found in {root}. "
                "Expected 'reference/' or 'reference_0/', 'reference_1/', ..."
            )

        # Attach metadata.
        results["_metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": _config_hash(self.config),
            "results_dir": str(root.resolve()),
            "device": self.device,
        }

        # Save report to the metrics output directory.
        metrics_dir = root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        report_path = metrics_dir / "evaluation_report.json"
        self.save_report(results, str(report_path))

        return results

    # ------------------------------------------------------------------
    # Report I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(results: dict[str, Any], output_path: str) -> None:
        """Save the evaluation report to a JSON file.

        Args:
            results: Dictionary of metric results (must be JSON-serialisable
                after conversion of nested dicts with int keys).
            output_path: Destination file path.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Convert any int-keyed dicts to string keys for JSON compatibility.
        def _sanitise(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {str(k): _sanitise(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitise(v) for v in obj]
            return obj

        with open(output, "w") as f:
            json.dump(_sanitise(results), f, indent=2, default=str)

        logger.info("Evaluation report saved to %s", output)
