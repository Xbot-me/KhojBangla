"""
Layout segmentation stage: split a newspaper page into columns, then lines within
each column, in correct reading order - BEFORE any OCR happens.

This is the step that's usually missing in "throw the whole page at Tesseract/Paddle"
attempts, and is very likely the single biggest source of gibberish output on
multi-column newspaper scans, independent of the OCR engine's script quality.

Approach: vertical/horizontal projection profiles with gap detection. This is the
classic, well-understood technique for column/line segmentation and - critically -
requires NO model download, so it works fully offline in any environment.

For pages with genuinely irregular layouts (ads breaking up columns, images mid-column),
this will need tuning of the gap thresholds below, or eventually replacing with a
trained layout model (e.g. Kraken, LAREX) - left as a documented upgrade path rather
than built in now, since those need external model weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def _find_gaps(profile: np.ndarray, min_gap: int, threshold_ratio: float = 0.02) -> list[tuple[int, int]]:
    """
    Given a 1D projection profile (sum of dark pixels per row or column), find
    contiguous 'content' runs separated by near-zero 'gap' runs of at least min_gap.
    Returns list of (start, end) index ranges for content runs.
    """
    threshold = profile.max() * threshold_ratio
    is_content = profile > threshold

    runs = []
    start = None
    gap_len = 0
    for i, val in enumerate(is_content):
        if val:
            if start is None:
                start = i
            gap_len = 0
        else:
            if start is not None:
                gap_len += 1
                if gap_len >= min_gap:
                    runs.append((start, i - gap_len + 1))
                    start = None
                    gap_len = 0
    if start is not None:
        runs.append((start, len(is_content)))

    return [(s, e) for s, e in runs if e > s]


def detect_columns(gray: np.ndarray, min_gap_px: int = 25) -> list[BoundingBox]:
    """
    Detect vertical column boundaries by looking for tall, mostly-empty vertical
    strips (the whitespace gutters between newspaper columns).
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_profile = thresh.sum(axis=0).astype(np.float64)

    h = gray.shape[0]
    runs = _find_gaps(col_profile, min_gap=min_gap_px)

    boxes = []
    for x_start, x_end in runs:
        boxes.append(BoundingBox(x=x_start, y=0, w=x_end - x_start, h=h))
    return boxes


def detect_lines(gray: np.ndarray, min_gap_px: int = 4) -> list[BoundingBox]:
    """
    Within a single column crop, detect horizontal text line boundaries by looking
    for thin, mostly-empty horizontal strips (the gaps between lines of text).
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_profile = thresh.sum(axis=1).astype(np.float64)

    w = gray.shape[1]
    runs = _find_gaps(row_profile, min_gap=min_gap_px)

    boxes = []
    for y_start, y_end in runs:
        boxes.append(BoundingBox(x=0, y=y_start, w=w, h=y_end - y_start))
    return boxes


def segment_page(
    gray: np.ndarray,
    min_column_gap_px: int = 25,
    min_line_gap_px: int = 4,
    min_column_width_px: int = 40,
    min_line_height_px: int = 10,
) -> list[dict]:
    """
    Full segmentation: columns left-to-right, then lines top-to-bottom within
    each column. Returns ordered list of crop descriptors preserving reading order:

        [{"column_index": 0, "line_index": 0, "bbox": BoundingBox, "image": np.ndarray}, ...]
    """
    columns = detect_columns(gray, min_gap_px=min_column_gap_px)
    columns = [c for c in columns if c.w >= min_column_width_px]

    results = []
    for col_idx, col_box in enumerate(columns):
        col_crop = col_box.crop(gray)
        lines = detect_lines(col_crop, min_gap_px=min_line_gap_px)
        lines = [ln for ln in lines if ln.h >= min_line_height_px]

        for line_idx, line_box in enumerate(lines):
            abs_box = BoundingBox(
                x=col_box.x, y=col_box.y + line_box.y, w=col_box.w, h=line_box.h
            )
            results.append(
                {
                    "column_index": col_idx,
                    "line_index": line_idx,
                    "bbox": abs_box,
                    "image": abs_box.crop(gray),
                }
            )
    return results


if __name__ == "__main__":
    import sys

    from preprocess import preprocess_page

    if len(sys.argv) != 3:
        print("Usage: python segment.py <input_image> <output_dir>")
        sys.exit(1)

    import os

    os.makedirs(sys.argv[2], exist_ok=True)
    processed = preprocess_page(sys.argv[1])
    crops = segment_page(processed)
    print(f"Found {len(crops)} line crops across "
          f"{len(set(c['column_index'] for c in crops))} columns")

    for c in crops:
        fname = f"col{c['column_index']:02d}_line{c['line_index']:03d}.png"
        cv2.imwrite(os.path.join(sys.argv[2], fname), c["image"])
