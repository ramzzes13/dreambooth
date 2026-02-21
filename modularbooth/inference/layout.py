"""Spatial layout generation for multi-subject scene composition.

This module provides utilities for creating, validating, and manipulating
bounding-box layouts that define where each personalized subject should appear
in a generated image.  Layouts are expressed in normalized [0, 1] coordinates
and can be converted to/from pixel coordinates for any target resolution.

Four placement strategies are available:

* **grid** -- arrange subjects in a regular grid.
* **horizontal** -- place subjects side by side in a single row.
* **vertical** -- stack subjects in a single column.
* **random** -- sample non-overlapping boxes with rejection sampling.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

logger = logging.getLogger(__name__)

# Type alias for a normalized bounding box (x1, y1, x2, y2) in [0, 1].
BBox = tuple[float, float, float, float]

VALID_STRATEGIES = {"grid", "horizontal", "vertical", "random"}


class LayoutGenerator:
    """Generate and manipulate spatial layouts for multi-subject scenes.

    All public methods are stateless (no constructor state is required), so the
    class can be instantiated once and reused across calls.
    """

    # ------------------------------------------------------------------
    # Layout generation
    # ------------------------------------------------------------------

    def generate_layout(
        self,
        num_subjects: int,
        image_size: tuple[int, int] = (1024, 1024),
        strategy: str = "grid",
    ) -> dict[str, BBox]:
        """Generate a spatial layout assigning a bounding box to each subject.

        Args:
            num_subjects: Number of subjects to place (must be >= 1).
            image_size: Target image dimensions ``(width, height)`` -- used only
                for aspect-ratio considerations; the returned coordinates are
                always in normalized [0, 1] space.
            strategy: Placement strategy.  One of ``"grid"``, ``"horizontal"``,
                ``"vertical"``, or ``"random"``.

        Returns:
            Dictionary mapping subject identifiers (``"V1"``, ``"V2"``, ...)
            to bounding boxes ``(x1, y1, x2, y2)`` in normalized coordinates.

        Raises:
            ValueError: If *num_subjects* < 1 or *strategy* is unknown.
        """
        if num_subjects < 1:
            raise ValueError(f"num_subjects must be >= 1, got {num_subjects}")
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Must be one of {sorted(VALID_STRATEGIES)}."
            )

        dispatch = {
            "grid": self._layout_grid,
            "horizontal": self._layout_horizontal,
            "vertical": self._layout_vertical,
            "random": self._layout_random,
        }
        boxes = dispatch[strategy](num_subjects, image_size)

        layout: dict[str, BBox] = {}
        for i, box in enumerate(boxes):
            subject_id = f"V{i + 1}"
            layout[subject_id] = box

        logger.debug(
            "Generated '%s' layout for %d subjects: %s",
            strategy,
            num_subjects,
            layout,
        )
        return layout

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_layout(
        self,
        layout: dict[str, BBox],
        min_size: float = 0.15,
    ) -> bool:
        """Check that a layout is geometrically valid.

        Validation criteria:

        1. Every box must have ``0 <= x1 < x2 <= 1`` and ``0 <= y1 < y2 <= 1``.
        2. Every box must have width and height >= *min_size*.
        3. No pair of boxes may have IoU > 0.3 (excessive overlap).

        Args:
            layout: Subject-id to bounding-box mapping.
            min_size: Minimum normalized width **and** height for each box.

        Returns:
            ``True`` if all checks pass, ``False`` otherwise.
        """
        boxes = list(layout.values())

        for sid, (x1, y1, x2, y2) in layout.items():
            # Bounds check
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                logger.warning("Box for %s is out of [0,1] bounds: %s", sid, (x1, y1, x2, y2))
                return False
            # Minimum size check
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                logger.warning(
                    "Box for %s is too small (%.3f x %.3f, min=%.3f).",
                    sid,
                    x2 - x1,
                    y2 - y1,
                    min_size,
                )
                return False

        # Pairwise overlap check
        ids = list(layout.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                iou = self.compute_overlap(layout[ids[i]], layout[ids[j]])
                if iou > 0.3:
                    logger.warning(
                        "Excessive overlap (IoU=%.3f) between %s and %s.",
                        iou,
                        ids[i],
                        ids[j],
                    )
                    return False

        return True

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def normalize_layout(
        self,
        layout: dict[str, BBox],
        image_size: tuple[int, int],
    ) -> dict[str, BBox]:
        """Convert pixel-coordinate bounding boxes to normalized [0, 1] coords.

        Args:
            layout: Subject-id to ``(x1, y1, x2, y2)`` in pixel coordinates.
            image_size: ``(width, height)`` of the target image.

        Returns:
            New dictionary with normalized bounding boxes.
        """
        w, h = image_size
        return {
            sid: (x1 / w, y1 / h, x2 / w, y2 / h)
            for sid, (x1, y1, x2, y2) in layout.items()
        }

    def denormalize_layout(
        self,
        layout: dict[str, BBox],
        image_size: tuple[int, int],
    ) -> dict[str, BBox]:
        """Convert normalized [0, 1] bounding boxes to pixel coordinates.

        Args:
            layout: Subject-id to ``(x1, y1, x2, y2)`` in normalized coords.
            image_size: ``(width, height)`` of the target image.

        Returns:
            New dictionary with pixel-coordinate bounding boxes (floats, not
            rounded -- callers may cast to ``int`` as needed).
        """
        w, h = image_size
        return {
            sid: (x1 * w, y1 * h, x2 * w, y2 * h)
            for sid, (x1, y1, x2, y2) in layout.items()
        }

    # ------------------------------------------------------------------
    # Overlap computation
    # ------------------------------------------------------------------

    def compute_overlap(self, box_a: BBox, box_b: BBox) -> float:
        """Compute Intersection-over-Union (IoU) between two bounding boxes.

        Both boxes should be in the same coordinate system (normalized or
        pixel -- it does not matter as long as they are consistent).

        Args:
            box_a: ``(x1, y1, x2, y2)`` for the first box.
            box_b: ``(x1, y1, x2, y2)`` for the second box.

        Returns:
            IoU value in [0, 1].  Returns 0.0 if the boxes do not intersect.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Intersection rectangle
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return float(inter_area / union_area)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_layout(
        self,
        layout: dict[str, BBox],
        image_size: tuple[int, int] = (1024, 1024),
    ) -> PIL.Image.Image:
        """Draw bounding boxes on a blank canvas for visual inspection.

        Each subject box is drawn in a distinct colour with its identifier
        label placed at the top-left corner.

        Args:
            layout: Subject-id to ``(x1, y1, x2, y2)`` in **normalized** coords.
            image_size: ``(width, height)`` for the output canvas.

        Returns:
            A PIL Image with the layout visualization.
        """
        w, h = image_size
        canvas = PIL.Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = PIL.ImageDraw.Draw(canvas)

        # Colour palette for up to 10 subjects; cycles if more are needed.
        colours = [
            (220, 50, 50),    # red
            (50, 120, 220),   # blue
            (50, 180, 50),    # green
            (220, 160, 0),    # orange
            (150, 50, 200),   # purple
            (0, 180, 180),    # teal
            (200, 100, 150),  # pink
            (120, 120, 50),   # olive
            (50, 50, 50),     # dark grey
            (100, 200, 200),  # light teal
        ]

        pixel_layout = self.denormalize_layout(layout, image_size)

        for idx, (sid, (x1, y1, x2, y2)) in enumerate(pixel_layout.items()):
            colour = colours[idx % len(colours)]
            # Draw rectangle with a 3-pixel border
            for offset in range(3):
                draw.rectangle(
                    [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                    outline=colour,
                )
            # Draw semi-transparent fill by compositing
            # (PIL doesn't support alpha directly on RGB, so use a thin fill)
            fill_colour = (*colour, 40)
            overlay = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
            overlay_draw = PIL.ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_colour)
            canvas = PIL.Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = PIL.ImageDraw.Draw(canvas)

            # Label
            label = f"[{sid}]"
            text_x = int(x1) + 6
            text_y = int(y1) + 6
            # Draw text shadow for readability
            draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0))
            draw.text((text_x, text_y), label, fill=colour)

        return canvas

    # ------------------------------------------------------------------
    # Private strategy implementations
    # ------------------------------------------------------------------

    def _layout_grid(
        self,
        num_subjects: int,
        image_size: tuple[int, int],
    ) -> list[BBox]:
        """Arrange subjects in a roughly square grid with padding.

        The grid dimensions are chosen so that ``cols * rows >= num_subjects``
        and the grid is as square as possible.  Each cell receives uniform
        padding so boxes do not touch the edges.
        """
        cols = math.ceil(math.sqrt(num_subjects))
        rows = math.ceil(num_subjects / cols)

        padding = 0.03  # normalized padding around each cell
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows

        boxes: list[BBox] = []
        for i in range(num_subjects):
            row = i // cols
            col = i % cols
            x1 = col * cell_w + padding
            y1 = row * cell_h + padding
            x2 = (col + 1) * cell_w - padding
            y2 = (row + 1) * cell_h - padding
            boxes.append((x1, y1, x2, y2))

        return boxes

    def _layout_horizontal(
        self,
        num_subjects: int,
        image_size: tuple[int, int],
    ) -> list[BBox]:
        """Place subjects side by side in a single row."""
        padding = 0.03
        cell_w = 1.0 / num_subjects
        # Vertically centred, occupying 60% of the height
        y1 = 0.20
        y2 = 0.80

        boxes: list[BBox] = []
        for i in range(num_subjects):
            x1 = i * cell_w + padding
            x2 = (i + 1) * cell_w - padding
            boxes.append((x1, y1, x2, y2))

        return boxes

    def _layout_vertical(
        self,
        num_subjects: int,
        image_size: tuple[int, int],
    ) -> list[BBox]:
        """Stack subjects vertically in a single column."""
        padding = 0.03
        cell_h = 1.0 / num_subjects
        # Horizontally centred, occupying 60% of the width
        x1 = 0.20
        x2 = 0.80

        boxes: list[BBox] = []
        for i in range(num_subjects):
            y1_box = i * cell_h + padding
            y2_box = (i + 1) * cell_h - padding
            boxes.append((x1, y1_box, x2, y2_box))

        return boxes

    def _layout_random(
        self,
        num_subjects: int,
        image_size: tuple[int, int],
        max_attempts: int = 500,
    ) -> list[BBox]:
        """Sample random non-overlapping boxes via rejection sampling.

        Each box has width and height uniformly sampled from [0.20, 0.45] in
        normalized coordinates.  A candidate is accepted only when its IoU with
        every previously placed box is below 0.10.

        Falls back to the grid strategy if placement fails after
        *max_attempts* total candidates.
        """
        min_dim = 0.20
        max_dim = 0.45
        max_iou = 0.10

        placed: list[BBox] = []
        attempts = 0

        while len(placed) < num_subjects and attempts < max_attempts:
            bw = random.uniform(min_dim, max_dim)
            bh = random.uniform(min_dim, max_dim)
            x1 = random.uniform(0.0, 1.0 - bw)
            y1 = random.uniform(0.0, 1.0 - bh)
            candidate = (x1, y1, x1 + bw, y1 + bh)

            # Check overlap with already-placed boxes
            ok = all(self.compute_overlap(candidate, p) < max_iou for p in placed)
            if ok:
                placed.append(candidate)

            attempts += 1

        if len(placed) < num_subjects:
            logger.warning(
                "Random layout failed after %d attempts for %d subjects; "
                "falling back to grid strategy.",
                max_attempts,
                num_subjects,
            )
            return self._layout_grid(num_subjects, image_size)

        return placed
