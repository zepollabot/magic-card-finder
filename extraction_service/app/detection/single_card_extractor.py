"""Single-card extraction with multi-strategy contouring.

Handles images where a single card fills most of the frame (50-99% area).
Uses three contouring strategies in order, inspired by
https://github.com/tmikonen/magic_card_detector:

1. Canny edge detection with permissive area thresholds
2. Adaptive threshold on grayscale
3. Per-channel (BGR) threshold with combined contours

Falls back to treating the whole image as the card when no contour
strategy succeeds.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .protocols import CardCrop
from .card_normalizer import (
    approximate_quad,
    normalize_card,
    order_points,
    perspective_warp,
)

MIN_AREA_FRACTION = 0.10
MAX_AREA_FRACTION = 0.98
MIN_SOLIDITY = 0.55
ASPECT_RATIO_LO = 1.10
ASPECT_RATIO_HI = 2.0
NMS_OVERLAP_THRESH = 0.50

MTG_CARD_RATIO = 936 / 672  # ~1.393

CANNY_THRESHOLDS = ((20, 60), (30, 100), (40, 120), (50, 150))
GAUSSIAN_KERNELS = (3, 5, 7)

logger = logging.getLogger(__name__)


class SingleCardExtractor:
    """Extracts a single card from an image where it dominates the frame."""

    def extract(self, image: np.ndarray) -> List[CardCrop]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * MIN_AREA_FRACTION
        max_area = image_area * MAX_AREA_FRACTION

        all_candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []

        strategies = [
            ("canny", lambda: self._canny_contours(gray, min_area, max_area)),
            ("adaptive", lambda: self._adaptive_threshold_contours(gray, image, min_area, max_area)),
            ("per_channel", lambda: self._per_channel_contours(image, min_area, max_area)),
        ]

        for name, strategy in strategies:
            candidates = strategy()
            if candidates:
                logger.debug("single: %s found %d candidate(s)", name, len(candidates))
                all_candidates.extend(candidates)

        # Deduplicate across strategies, then pick the best card-shaped
        # contour — scored by aspect ratio match and area, with full-image
        # contours penalized (they are background, not card).
        deduplicated = self._nms(all_candidates)
        if deduplicated:
            best = self._pick_best(deduplicated, image.shape[:2])
            crop = self._warp_candidate(image, best)
            if crop is not None:
                _, _, area = best
                logger.debug(
                    "single: using best candidate, area=%d (%.1f%% of image)",
                    area, 100.0 * area / image_area,
                )
                return [crop]

        logger.debug("single: no contour found, using whole-image fallback")
        return [self._whole_image_fallback(image)]

    # ── Strategy 1: Canny edge detection ─────────────────────

    def _canny_contours(
        self, gray: np.ndarray, min_area: float, max_area: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        all_candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []

        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        for lo, hi in CANNY_THRESHOLDS:
            self._collect_canny(bilateral, lo, hi, min_area, max_area, all_candidates)

        for k in GAUSSIAN_KERNELS:
            blurred = cv2.GaussianBlur(gray, (k, k), 0)
            for lo, hi in CANNY_THRESHOLDS:
                self._collect_canny(blurred, lo, hi, min_area, max_area, all_candidates)

        return self._nms(all_candidates)

    def _collect_canny(
        self,
        preprocessed: np.ndarray,
        canny_lo: int,
        canny_hi: int,
        min_area: float,
        max_area: float,
        out: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> None:
        edges = cv2.Canny(preprocessed, canny_lo, canny_hi)
        dilate_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self._filter_contours(contours, min_area, max_area, out)

    # ── Strategy 2: Adaptive threshold ───────────────────────

    def _adaptive_threshold_contours(
        self, gray: np.ndarray, image: np.ndarray,
        min_area: float, max_area: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        fltr_size = 1 + 2 * (min(image.shape[0], image.shape[1]) // 20)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            fltr_size, 10,
        )
        contours, _ = cv2.findContours(
            np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        all_candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self._filter_contours(contours, min_area, max_area, all_candidates)
        return self._nms(all_candidates)

    # ── Strategy 3: Per-channel BGR threshold ────────────────

    def _per_channel_contours(
        self, image: np.ndarray,
        min_area: float, max_area: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        all_candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []

        for channel in cv2.split(image):
            enhanced = clahe.apply(channel)
            _, binary = cv2.threshold(enhanced, 110, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                np.uint8(binary), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
            )
            self._filter_contours(contours, min_area, max_area, all_candidates)

        return self._nms(all_candidates)

    # ── Shared contour filtering ─────────────────────────────

    def _filter_contours(
        self,
        contours: List[np.ndarray],
        min_area: float,
        max_area: float,
        out: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> None:
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

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                hull_solidity = carea / hull_area
                if hull_solidity < 0.80:
                    continue

            quad = approximate_quad(cnt)
            if quad is not None:
                out.append((cnt, quad, int(carea)))

    # ── NMS ──────────────────────────────────────────────────

    @staticmethod
    def _nms(
        candidates: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        if not candidates:
            return []
        candidates.sort(key=lambda c: c[2], reverse=True)
        keep: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for cand in candidates:
            cx, cy, cw, ch = cv2.boundingRect(cand[0])
            overlap = False
            for kept in keep:
                kx, ky, kw, kh = cv2.boundingRect(kept[0])
                x1 = max(cx, kx)
                y1 = max(cy, ky)
                x2 = min(cx + cw, kx + kw)
                y2 = min(cy + ch, ky + kh)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                smaller = min(cw * ch, kw * kh)
                if smaller > 0 and inter / smaller > NMS_OVERLAP_THRESH:
                    overlap = True
                    break
            if not overlap:
                keep.append(cand)
        return keep

    # ── Candidate selection and warping ──────────────────────

    @staticmethod
    def _pick_best(
        candidates: List[Tuple[np.ndarray, np.ndarray, int]],
        image_shape: Tuple[int, int] = (0, 0),
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Pick the candidate that best resembles an actual card.

        Scores each candidate by combining:
        - Aspect ratio closeness to MTG standard (1.393)
        - Area (larger is better, but full-image contours are penalized)

        A contour whose bounding box covers >95% of the image in both
        dimensions is almost certainly the background, not a card.
        """
        img_h, img_w = image_shape

        def _score(c: Tuple[np.ndarray, np.ndarray, int]) -> float:
            cnt, _, area = c
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                return -1.0

            aspect = max(w, h) / min(w, h)
            aspect_error = abs(aspect - MTG_CARD_RATIO) / MTG_CARD_RATIO

            # Penalize contours that span the entire image — they're
            # the background or the image border, not a card.
            if img_h > 0 and img_w > 0:
                w_coverage = w / img_w
                h_coverage = h / img_h
                if w_coverage > 0.95 and h_coverage > 0.95:
                    return -1.0

            # aspect_score: 1.0 when perfect match, 0 when 30%+ off
            aspect_score = max(0.0, 1.0 - aspect_error / 0.30)
            # area_score: normalized to [0, 1]
            area_score = area / max(1, img_h * img_w) if img_h > 0 else 0.5
            return aspect_score * 0.6 + area_score * 0.4

        return max(candidates, key=_score)

    @staticmethod
    def _warp_candidate(
        image: np.ndarray,
        candidate: Tuple[np.ndarray, np.ndarray, int],
    ) -> Optional[CardCrop]:
        _, quad, _ = candidate
        warp = perspective_warp(image, quad)
        if warp is None:
            return None
        normalized = normalize_card(warp)
        return CardCrop(image=normalized)

    # ── Whole-image fallback ─────────────────────────────────

    @staticmethod
    def _whole_image_fallback(image: np.ndarray) -> CardCrop:
        """Treat the entire image as the card, trimming obvious borders.

        Uses edge detection on the border strips to find where the card
        boundary starts, then crops to that region before normalizing.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        margin = 0.05
        top = _find_border_inset(gray, "top", margin)
        bottom = _find_border_inset(gray, "bottom", margin)
        left = _find_border_inset(gray, "left", margin)
        right = _find_border_inset(gray, "right", margin)

        y_start = top
        y_end = h - bottom
        x_start = left
        x_end = w - right

        if y_end <= y_start or x_end <= x_start:
            cropped = image
        else:
            cropped = image[y_start:y_end, x_start:x_end]

        normalized = normalize_card(cropped)
        return CardCrop(image=normalized)


def _find_border_inset(gray: np.ndarray, side: str, margin: float) -> int:
    """Find how many pixels of background border exist on a given side."""
    h, w = gray.shape[:2]
    scan_depth = int(max(h, w) * margin)
    if scan_depth < 3:
        return 0

    if side == "top":
        strip = gray[:scan_depth, :]
    elif side == "bottom":
        strip = gray[h - scan_depth :, :]
    elif side == "left":
        strip = gray[:, :scan_depth]
    elif side == "right":
        strip = gray[:, w - scan_depth :]
    else:
        return 0

    edges = cv2.Canny(strip, 50, 150)

    if side in ("top", "bottom"):
        row_sums = edges.sum(axis=1)
        threshold = row_sums.max() * 0.3
        for i, val in enumerate(row_sums):
            if val > threshold:
                return i if side == "top" else max(0, scan_depth - i - 1)
    else:
        col_sums = edges.sum(axis=0)
        threshold = col_sums.max() * 0.3
        for i, val in enumerate(col_sums):
            if val > threshold:
                return i if side == "left" else max(0, scan_depth - i - 1)

    return 0
