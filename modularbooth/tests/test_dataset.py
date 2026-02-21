"""Unit tests for the ModularBooth data pipeline.

Covers DreamBoothDataset (length, item format, interleaving),
SubjectCaptioner (template-based captioning), and DreamBoothBenchmark
(evaluation prompt generation).

All tests create small dummy PNG images in temporary directories so
they run without any external data.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dummy_images(directory: Path, count: int, prefix: str = "img") -> list[Path]:
    """Create *count* tiny 32x32 RGB PNG images in *directory*.

    Returns the sorted list of created paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        img = Image.new("RGB", (32, 32), color=(i * 30 % 256, 100, 150))
        p = directory / f"{prefix}_{i:03d}.png"
        img.save(p)
        paths.append(p)
    return sorted(paths)


# ===================================================================
# DreamBoothDataset tests
# ===================================================================


class TestDreamBoothDatasetLength(unittest.TestCase):
    """Verify that dataset length matches the interleaving logic:
    - For each subject image we emit 1 subject sample + 1 class sample.
    - Remaining class images (if any) are appended.
    """

    def test_dreambooth_dataset_length(self) -> None:
        from modularbooth.data.dataset import DreamBoothDataset

        with tempfile.TemporaryDirectory() as tmp:
            subject_dir = Path(tmp) / "subjects"
            class_dir = Path(tmp) / "classes"
            _create_dummy_images(subject_dir, 3)
            _create_dummy_images(class_dir, 5)

            ds = DreamBoothDataset(
                subject_images_dir=subject_dir,
                class_images_dir=class_dir,
                token="[V]",
                class_noun="dog",
                resolution=64,
            )

            # 3 subject images -> 3 subject + 3 class (interleaved) = 6
            # 5 class images total, 3 used above, 2 appended = 6 + 2 = 8
            self.assertEqual(len(ds), 8)


class TestDreamBoothDatasetItem(unittest.TestCase):
    """Verify that __getitem__ returns a dict with the expected keys
    and tensor shapes.
    """

    def test_dreambooth_dataset_item(self) -> None:
        from modularbooth.data.dataset import DreamBoothDataset

        with tempfile.TemporaryDirectory() as tmp:
            subject_dir = Path(tmp) / "subjects"
            class_dir = Path(tmp) / "classes"
            _create_dummy_images(subject_dir, 2)
            _create_dummy_images(class_dir, 2)

            ds = DreamBoothDataset(
                subject_images_dir=subject_dir,
                class_images_dir=class_dir,
                token="[V]",
                class_noun="cat",
                resolution=64,
            )

            item = ds[0]

            # Required keys
            self.assertIn("pixel_values", item)
            self.assertIn("input_ids", item)
            self.assertIn("is_class_image", item)

            # pixel_values should be a 3-D tensor (C, H, W) in [-1, 1]
            pv = item["pixel_values"]
            self.assertIsInstance(pv, torch.Tensor)
            self.assertEqual(pv.shape, (3, 64, 64))
            self.assertGreaterEqual(pv.min().item(), -1.01)
            self.assertLessEqual(pv.max().item(), 1.01)

            # input_ids is a string caption
            self.assertIsInstance(item["input_ids"], str)

            # is_class_image is a bool
            self.assertIsInstance(item["is_class_image"], bool)


class TestDreamBoothDatasetInterleave(unittest.TestCase):
    """Subject and class images should alternate in the interleaved
    sample list: subject, class, subject, class, ...
    """

    def test_dreambooth_dataset_interleave(self) -> None:
        from modularbooth.data.dataset import DreamBoothDataset

        with tempfile.TemporaryDirectory() as tmp:
            subject_dir = Path(tmp) / "subjects"
            class_dir = Path(tmp) / "classes"
            _create_dummy_images(subject_dir, 3)
            _create_dummy_images(class_dir, 3)

            ds = DreamBoothDataset(
                subject_images_dir=subject_dir,
                class_images_dir=class_dir,
                token="[V]",
                class_noun="dog",
                resolution=64,
            )

            # The first 2*3=6 samples should alternate subject/class.
            for i in range(6):
                item = ds[i]
                expected_is_class = (i % 2 == 1)
                self.assertEqual(
                    item["is_class_image"],
                    expected_is_class,
                    f"Sample {i}: expected is_class_image={expected_is_class}, "
                    f"got {item['is_class_image']}",
                )


# ===================================================================
# SubjectCaptioner tests
# ===================================================================


class TestCaptionerTemplate(unittest.TestCase):
    """SubjectCaptioner.generate_deterministic_caption should produce
    captions that contain both the token and the class noun.
    """

    def test_captioner_template(self) -> None:
        from modularbooth.data.captioning import SubjectCaptioner

        captioner = SubjectCaptioner()
        token = "[V]"
        class_noun = "dog"

        for idx in range(5):
            caption = captioner.generate_deterministic_caption(idx, token, class_noun)
            self.assertIn(token, caption, f"Token not in caption: {caption!r}")
            self.assertIn(class_noun, caption, f"Class noun not in caption: {caption!r}")


# ===================================================================
# DreamBoothBenchmark tests
# ===================================================================


class TestBenchmarkPrompts(unittest.TestCase):
    """DreamBoothBenchmark.get_evaluation_prompts should return
    ``num_tokens * 25`` prompts for single-subject evaluation (25 is
    the number of standard DreamBooth prompt templates).
    """

    def test_benchmark_prompts(self) -> None:
        from modularbooth.data.benchmark import DreamBoothBenchmark

        bench = DreamBoothBenchmark()
        subject_tokens = {"[V]": "dog"}
        prompts = bench.get_evaluation_prompts(subject_tokens, num_subjects=1)

        # 1 subject * 25 templates = 25 prompts
        self.assertEqual(len(prompts), 25)

        # Each prompt should contain the subject string "a [V] dog"
        for p in prompts:
            self.assertIn("a [V] dog", p, f"Subject string missing in: {p!r}")


if __name__ == "__main__":
    unittest.main()
