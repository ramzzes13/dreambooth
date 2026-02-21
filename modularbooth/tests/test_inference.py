"""Unit tests for the ModularBooth inference / layout module.

All tests are pure-CPU and do not load any diffusion models.  They
exercise the LayoutGenerator's placement strategies, validation,
coordinate conversion, overlap computation, and visualisation.
"""

from __future__ import annotations

import unittest

import PIL.Image

from modularbooth.inference.layout import LayoutGenerator


# ===================================================================
# Layout generation tests
# ===================================================================


class TestLayoutGeneratorHorizontal(unittest.TestCase):
    """Horizontal layout for 2 subjects should produce two non-
    overlapping boxes arranged left-to-right.
    """

    def test_layout_generator_horizontal(self) -> None:
        gen = LayoutGenerator()
        layout = gen.generate_layout(num_subjects=2, strategy="horizontal")

        self.assertEqual(len(layout), 2)
        self.assertIn("V1", layout)
        self.assertIn("V2", layout)

        x1_a, y1_a, x2_a, y2_a = layout["V1"]
        x1_b, y1_b, x2_b, y2_b = layout["V2"]

        # V1 should be entirely to the left of V2.
        self.assertLess(x2_a, x1_b + 1e-6, "V1 right edge should be <= V2 left edge")

        # Both boxes should have the same vertical extent.
        self.assertAlmostEqual(y1_a, y1_b, places=4)
        self.assertAlmostEqual(y2_a, y2_b, places=4)


class TestLayoutGeneratorGrid(unittest.TestCase):
    """Grid layout for 4 subjects should produce a 2x2 arrangement."""

    def test_layout_generator_grid(self) -> None:
        gen = LayoutGenerator()
        layout = gen.generate_layout(num_subjects=4, strategy="grid")

        self.assertEqual(len(layout), 4)

        # All boxes should be within [0, 1].
        for sid, (x1, y1, x2, y2) in layout.items():
            self.assertGreaterEqual(x1, 0.0, f"{sid} x1 out of bounds")
            self.assertGreaterEqual(y1, 0.0, f"{sid} y1 out of bounds")
            self.assertLessEqual(x2, 1.0, f"{sid} x2 out of bounds")
            self.assertLessEqual(y2, 1.0, f"{sid} y2 out of bounds")
            self.assertLess(x1, x2, f"{sid} degenerate width")
            self.assertLess(y1, y2, f"{sid} degenerate height")

        # In a 2x2 grid the boxes should tile into quadrants.
        # V1 (row=0, col=0) should be top-left, V4 (row=1, col=1) bottom-right.
        # Check that V1's center is in the top-left quadrant.
        cx_v1 = (layout["V1"][0] + layout["V1"][2]) / 2
        cy_v1 = (layout["V1"][1] + layout["V1"][3]) / 2
        self.assertLess(cx_v1, 0.5, "V1 center should be in the left half")
        self.assertLess(cy_v1, 0.5, "V1 center should be in the top half")

        cx_v4 = (layout["V4"][0] + layout["V4"][2]) / 2
        cy_v4 = (layout["V4"][1] + layout["V4"][3]) / 2
        self.assertGreater(cx_v4, 0.5, "V4 center should be in the right half")
        self.assertGreater(cy_v4, 0.5, "V4 center should be in the bottom half")


# ===================================================================
# Validation tests
# ===================================================================


class TestLayoutValidate(unittest.TestCase):
    """validate_layout should accept valid layouts and reject
    overlapping ones.
    """

    def test_valid_layout_passes(self) -> None:
        gen = LayoutGenerator()
        valid = {
            "V1": (0.0, 0.0, 0.45, 0.45),
            "V2": (0.55, 0.55, 1.0, 1.0),
        }
        self.assertTrue(gen.validate_layout(valid, min_size=0.15))

    def test_overlapping_layout_fails(self) -> None:
        gen = LayoutGenerator()
        # Two boxes occupying the same region -> IoU = 1.0 > 0.3
        overlapping = {
            "V1": (0.1, 0.1, 0.9, 0.9),
            "V2": (0.1, 0.1, 0.9, 0.9),
        }
        self.assertFalse(gen.validate_layout(overlapping, min_size=0.15))


# ===================================================================
# Coordinate conversion tests
# ===================================================================


class TestLayoutNormalizeDenormalize(unittest.TestCase):
    """Round-tripping through normalize -> denormalize should recover
    the original pixel coordinates.
    """

    def test_layout_normalize_denormalize(self) -> None:
        gen = LayoutGenerator()
        image_size = (512, 256)

        pixel_layout = {
            "V1": (51.2, 25.6, 256.0, 128.0),
            "V2": (300.0, 100.0, 480.0, 230.0),
        }

        normalised = gen.normalize_layout(pixel_layout, image_size)

        # All normalised coords should be in [0, 1].
        for sid, (x1, y1, x2, y2) in normalised.items():
            self.assertGreaterEqual(x1, 0.0)
            self.assertLessEqual(x2, 1.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(y2, 1.0)

        recovered = gen.denormalize_layout(normalised, image_size)

        for sid in pixel_layout:
            for orig, rec in zip(pixel_layout[sid], recovered[sid]):
                self.assertAlmostEqual(orig, rec, places=4,
                                       msg=f"Round-trip mismatch for {sid}")


# ===================================================================
# Visualisation tests
# ===================================================================


class TestLayoutVisualize(unittest.TestCase):
    """visualize_layout should produce a PIL Image of the requested
    canvas size.
    """

    def test_layout_visualize(self) -> None:
        gen = LayoutGenerator()
        layout = gen.generate_layout(num_subjects=3, strategy="grid")
        image_size = (512, 512)

        canvas = gen.visualize_layout(layout, image_size=image_size)

        self.assertIsInstance(canvas, PIL.Image.Image)
        self.assertEqual(canvas.size, image_size)


# ===================================================================
# Overlap / IoU tests
# ===================================================================


class TestComputeOverlapNoIntersection(unittest.TestCase):
    """Non-overlapping boxes should have IoU = 0."""

    def test_compute_overlap_no_intersection(self) -> None:
        gen = LayoutGenerator()
        box_a = (0.0, 0.0, 0.4, 0.4)
        box_b = (0.6, 0.6, 1.0, 1.0)

        iou = gen.compute_overlap(box_a, box_b)
        self.assertAlmostEqual(iou, 0.0, places=6)


class TestComputeOverlapFull(unittest.TestCase):
    """Identical boxes should have IoU = 1."""

    def test_compute_overlap_full(self) -> None:
        gen = LayoutGenerator()
        box = (0.1, 0.2, 0.8, 0.9)

        iou = gen.compute_overlap(box, box)
        self.assertAlmostEqual(iou, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
