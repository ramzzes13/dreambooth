"""ModularBooth evaluation metrics for multi-subject personalisation.

This package provides the following metric classes:

* :class:`DINOScore` / :class:`DINOv2Score` -- Subject fidelity via DINO embeddings.
* :class:`CLIPScore` -- CLIP-I (image-image) and CLIP-T (text-image) alignment.
* :class:`IdentityIsolationScore` -- Multi-subject identity isolation.
* :class:`ContextAppearanceEntanglement` -- Appearance stability across contexts.
* :class:`LPIPSDiversity` -- Output diversity via LPIPS perceptual distance.
* :class:`VQAAlignment` -- VQA-based prompt alignment (placeholder).
* :class:`EvaluationPipeline` -- End-to-end orchestration of all metrics.
"""

from modularbooth.evaluation.clip_score import CLIPScore
from modularbooth.evaluation.dino_score import DINOScore, DINOv2Score
from modularbooth.evaluation.diversity import LPIPSDiversity
from modularbooth.evaluation.entanglement import ContextAppearanceEntanglement
from modularbooth.evaluation.identity_isolation import IdentityIsolationScore
from modularbooth.evaluation.run_evaluation import EvaluationPipeline
from modularbooth.evaluation.vqa_alignment import VQAAlignment

__all__ = [
    "CLIPScore",
    "ContextAppearanceEntanglement",
    "DINOScore",
    "DINOv2Score",
    "EvaluationPipeline",
    "IdentityIsolationScore",
    "LPIPSDiversity",
    "VQAAlignment",
]
