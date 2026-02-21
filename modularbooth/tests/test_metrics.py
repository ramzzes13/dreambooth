"""Unit tests for ModularBooth evaluation metrics.

All tests run on CPU without downloading any models.  Heavy model
dependencies (DINO, CLIP, LPIPS) are mocked so that the tests exercise
the metric logic -- embedding aggregation, score computation,
question generation -- without requiring real weights.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_pil_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a small solid-colour RGB PIL image."""
    return Image.new("RGB", (width, height), color=(128, 128, 128))


# ===================================================================
# DINOScore tests
# ===================================================================


class TestDINOScoreIdenticalImages(unittest.TestCase):
    """When identical images are passed through DINO, their normalised
    embeddings should be equal and the cosine similarity score should
    be ~1.0.
    """

    @patch("modularbooth.evaluation.dino_score.torch.hub.load")
    def test_dino_score_identical_images(self, mock_hub_load: MagicMock) -> None:
        """Mock DINO to return a fixed embedding and verify score ~1.0."""
        # The mock model returns the same embedding for any input.
        fixed_embedding = torch.randn(1, 384)
        mock_model = MagicMock()
        mock_model.return_value = fixed_embedding
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_hub_load.return_value = mock_model

        from modularbooth.evaluation.dino_score import DINOScore

        scorer = DINOScore(device="cpu")

        img_a = _dummy_pil_image()
        img_b = _dummy_pil_image()

        # compute_embeddings calls the model with a batch.  Since the mock
        # always returns the same vector, all pairwise similarities are 1.
        # Override compute_embeddings so it returns normalised vectors.
        def _mock_compute_embeddings(images):
            emb = fixed_embedding.expand(len(images), -1)
            return F.normalize(emb, dim=1)

        scorer.compute_embeddings = _mock_compute_embeddings  # type: ignore[assignment]

        score = scorer.compute_score([img_a], [img_b])
        self.assertAlmostEqual(score, 1.0, places=4)


# ===================================================================
# CLIPScore tests
# ===================================================================


class TestCLIPScoreTextImageMismatch(unittest.TestCase):
    """When CLIP returns orthogonal embeddings for image and text the
    CLIP-T score should be ~0.
    """

    def test_clip_score_text_image_mismatch(self) -> None:
        from modularbooth.evaluation.clip_score import CLIPScore

        scorer = CLIPScore(device="cpu")

        # Provide pre-computed *orthogonal* embeddings so the paired
        # cosine similarity is zero.
        img_emb = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), dim=1)
        txt_emb = F.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), dim=1)

        scorer.compute_image_embeddings = MagicMock(return_value=img_emb)
        scorer.compute_text_embeddings = MagicMock(return_value=txt_emb)

        score = scorer.clip_t_score([_dummy_pil_image()], ["random text"])
        self.assertAlmostEqual(score, 0.0, places=4)


# ===================================================================
# IdentityIsolationScore tests
# ===================================================================


class TestIdentityIsolationPositive(unittest.TestCase):
    """When the crop matches the correct reference and does not match the
    wrong reference, the IIS should be positive.
    """

    def test_identity_isolation_positive(self) -> None:
        from modularbooth.evaluation.identity_isolation import IdentityIsolationScore

        mock_dino = MagicMock()

        # Subject-0 embedding (crop and reference) = [1, 0, 0, 0]
        # Subject-1 embedding (crop and reference) = [0, 1, 0, 0]
        emb_s0 = F.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
        emb_s1 = F.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=0)

        # compute_embedding returns the embedding for a crop.
        call_count = {"n": 0}

        def _compute_embedding(image):
            idx = call_count["n"]
            call_count["n"] += 1
            return emb_s0 if idx == 0 else emb_s1

        mock_dino.compute_embedding = _compute_embedding

        # compute_embeddings returns batched embeddings for references.
        ref_emb_s0 = emb_s0.unsqueeze(0)  # (1, 4)
        ref_emb_s1 = emb_s1.unsqueeze(0)  # (1, 4)

        def _compute_embeddings(images):
            # Determine which reference set this is by checking context.
            # We return the same embedding for all images in the set.
            # The caller alternates: first call for correct refs, second for incorrect,
            # etc.  Instead, we just return the correct embedding based on call pattern.
            # We'll track calls explicitly.
            return ref_emb_s0 if _compute_embeddings._calls % 2 == 0 else ref_emb_s1

        # Each call to _mean_similarity_to_refs calls compute_embeddings once.
        # For subject 0: pos calls compute_embeddings with refs[0] -> ref_emb_s0 (call 0)
        #                neg calls compute_embeddings with refs[1] -> ref_emb_s1 (call 1)
        # For subject 1: pos calls compute_embeddings with refs[1] -> ref_emb_s0 (call 2, wrong!)
        # Let's just mock the IIS computation at a higher level.

        iis = IdentityIsolationScore(dino_scorer=mock_dino)

        # Instead of dealing with the internal call flow, we mock
        # _mean_similarity_to_refs directly.
        call_log: list[tuple] = []

        def _mock_mean_sim(crop_emb, ref_images):
            call_log.append(("sim", len(ref_images)))
            # If crop_emb matches the first ref image embedding direction,
            # return high similarity; otherwise return low.
            # We can identify which ref set by the first ref image id.
            ref_id = id(ref_images)
            if ref_id == id(ref_images_0):
                # Similarity of crop to subject 0 references.
                sim = (crop_emb @ emb_s0).item()
            else:
                sim = (crop_emb @ emb_s1).item()
            return sim

        # We'll set up reference images as distinct objects so we can identify them.
        ref_images_0 = [_dummy_pil_image()]
        ref_images_1 = [_dummy_pil_image()]

        iis._mean_similarity_to_refs = _mock_mean_sim  # type: ignore[assignment]

        generated = Image.new("RGB", (256, 256), (200, 200, 200))
        crops = [(0, 0, 128, 256), (128, 0, 256, 256)]
        references = {0: ref_images_0, 1: ref_images_1}

        score = iis.compute_iis(generated, crops, references)
        # pos_sim - max_neg_sim should be positive for both subjects
        # since each crop embedding is orthogonal to the other.
        self.assertGreater(score, 0.0)


# ===================================================================
# ContextAppearanceEntanglement tests
# ===================================================================


class TestCAENoVariance(unittest.TestCase):
    """When all DINO embeddings are identical, the covariance trace (and
    hence CAE) should be 0.
    """

    def test_cae_no_variance(self) -> None:
        from modularbooth.evaluation.entanglement import ContextAppearanceEntanglement

        # _normalised_covariance_trace is a staticmethod -- test directly.
        identical_embs = torch.ones(5, 128)  # all rows equal
        cae = ContextAppearanceEntanglement._normalised_covariance_trace(identical_embs)
        self.assertAlmostEqual(cae, 0.0, places=6)


class TestCAEHighVariance(unittest.TestCase):
    """When embeddings vary substantially, CAE should be measurably > 0."""

    def test_cae_high_variance(self) -> None:
        from modularbooth.evaluation.entanglement import ContextAppearanceEntanglement

        # Use one-hot embeddings -- very high variance.
        embeddings = torch.eye(5)  # 5x5 identity matrix
        cae = ContextAppearanceEntanglement._normalised_covariance_trace(embeddings)
        # Covariance of one-hot rows is non-zero.
        self.assertGreater(cae, 0.01)


# ===================================================================
# LPIPSDiversity (LPIPS ~0 for identical images)
# ===================================================================


class TestLPIPSIdentical(unittest.TestCase):
    """Identical images should yield LPIPS distance ~0."""

    def test_lpips_identical(self) -> None:
        from torchvision import transforms

        from modularbooth.evaluation.diversity import LPIPSDiversity

        # Build a mock LPIPS model that always returns 0 distance.
        mock_lpips_fn = MagicMock()
        mock_lpips_fn.return_value = torch.tensor(0.0)

        scorer = LPIPSDiversity(device="cpu")

        # Directly inject the mock model and transform to bypass _ensure_model
        # (which does the local `import lpips` that's hard to patch).
        scorer._lpips_fn = mock_lpips_fn
        scorer._transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img = _dummy_pil_image()
        dist = scorer.compute_diversity([img, img])
        self.assertAlmostEqual(dist, 0.0, places=4)


# ===================================================================
# VQAAlignment question generation
# ===================================================================


class TestVQAQuestionGeneration(unittest.TestCase):
    """VQAAlignment.generate_questions should produce questions that
    reference the key nouns from the prompt.
    """

    def test_vqa_question_generation(self) -> None:
        from modularbooth.evaluation.vqa_alignment import VQAAlignment

        vqa = VQAAlignment()
        questions = vqa.generate_questions("a dog on a beach")

        # There should be at least one question mentioning "dog" and one
        # mentioning "beach".
        all_text = " ".join(questions).lower()
        self.assertIn("dog", all_text, "Expected a question about 'dog'")
        self.assertIn("beach", all_text, "Expected a question about 'beach'")

        # All generated strings should be questions (end with '?').
        for q in questions:
            self.assertTrue(q.endswith("?"), f"Not a question: {q!r}")


if __name__ == "__main__":
    unittest.main()
