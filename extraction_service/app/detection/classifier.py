"""Classify an image as containing a single card or multiple cards.

Uses dual area-threshold contour analysis: if card-shaped contours exist
in the multi-card range (1-35% of image area) the image is classified as
MULTI; otherwise, if a dominant contour covers >35% it is SINGLE.
"""

import enum
import logging
from typing import List, Tuple

import cv2
import numpy as np

from .card_normalizer import approximate_quad

logger = logging.getLogger(__name__)

MULTI_MIN_AREA = 0.01
MULTI_MAX_AREA = 0.35
SINGLE_MIN_AREA = 0.20
SINGLE_MAX_AREA = 0.98

MIN_SOLIDITY = 0.65
ASPECT_RATIO_LO = 1.05
ASPECT_RATIO_HI = 2.5

CANNY_THRESHOLDS = ((30, 100), (50, 150))


class ImageType(enum.Enum):
    SINGLE = "single"
    MULTI = "multi"


class ImageClassifier:
    """Decides whether an image contains a single card or multiple cards."""

    def classify(self, image: np.ndarray) -> ImageType:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_area = image.shape[0] * image.shape[1]

        single_hits = self._count_card_contours(
            gray, image_area, SINGLE_MIN_AREA, SINGLE_MAX_AREA,
        )
        multi_hits = self._count_card_contours(
            gray, image_area, MULTI_MIN_AREA, MULTI_MAX_AREA,
        )

        # A dominant large contour means single card -- small contours
        # inside it are internal card features (text box, art border),
        # not separate cards.
        if single_hits >= 1:
            logger.debug(
                "classifier: SINGLE (dominant contour in single range, "
                "multi_hits=%d are likely internal features)", multi_hits,
            )
            return ImageType.SINGLE

        if multi_hits >= 1:
            logger.debug("classifier: MULTI (%d contours in multi range)", multi_hits)
            return ImageType.MULTI

        logger.debug("classifier: SINGLE (fallback — no contours in either range)")
        return ImageType.SINGLE

    def _count_card_contours(
        self,
        gray: np.ndarray,
        image_area: int,
        min_frac: float,
        max_frac: float,
    ) -> int:
        """Count distinct card-shaped contours within the given area range."""
        min_area = image_area * min_frac
        max_area = image_area * max_frac

        quads: List[Tuple[np.ndarray, int]] = []

        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        for lo, hi in CANNY_THRESHOLDS:
            self._collect(bilateral, lo, hi, min_area, max_area, quads)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        for lo, hi in CANNY_THRESHOLDS:
            self._collect(blurred, lo, hi, min_area, max_area, quads)

        return self._deduplicate_count(quads)

    def _collect(
        self,
        preprocessed: np.ndarray,
        canny_lo: int,
        canny_hi: int,
        min_area: float,
        max_area: float,
        out: List[Tuple[np.ndarray, int]],
    ) -> None:
        edges = cv2.Canny(preprocessed, canny_lo, canny_hi)
        dilate_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            carea = cv2.contourArea(cnt)
            if carea < min_area or carea > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            if rect_area == 0:
                continue

            solidity = carea / rect_area
            if solidity < MIN_SOLIDITY:
                continue

            aspect = max(w, h) / max(1, min(w, h))
            if not (ASPECT_RATIO_LO <= aspect <= ASPECT_RATIO_HI):
                continue

            quad = approximate_quad(cnt)
            if quad is not None:
                out.append((quad, int(carea)))

    @staticmethod
    def _deduplicate_count(quads: List[Tuple[np.ndarray, int]]) -> int:
        """NMS-style deduplication, returns the number of distinct detections."""
        if not quads:
            return 0
        quads.sort(key=lambda q: q[1], reverse=True)
        keep: List[Tuple[int, int, int, int]] = []
        for quad, _ in quads:
            pts = quad.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            overlap = False
            for kx, ky, kw, kh in keep:
                x1 = max(x, kx)
                y1 = max(y, ky)
                x2 = min(x + w, kx + kw)
                y2 = min(y + h, ky + kh)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                smaller = min(w * h, kw * kh)
                if smaller > 0 and inter / smaller > 0.50:
                    overlap = True
                    break
            if not overlap:
                keep.append((x, y, w, h))
        return len(keep)
