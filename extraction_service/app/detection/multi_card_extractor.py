"""Multi-card extraction using Canny edge detection + contour filtering.

This is the original detection algorithm from OpenCVCardDetector, extracted
into its own class for clarity.  It works well when multiple cards are
visible and each card occupies a small fraction of the total image.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .protocols import CardCrop
from .card_normalizer import (
    approximate_quad,
    normalize_card,
    order_points,
    perspective_warp,
)

SINGLE_CARD_RATIO = 936 / 672  # ~1.393
MULTI_CARD_TOLERANCE = 0.25
MAX_MERGED_CARDS = 4

MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.35
MIN_SOLIDITY = 0.65
ASPECT_RATIO_LO = 1.05
ASPECT_RATIO_HI = 7.0
NMS_OVERLAP_THRESH = 0.50

CANNY_THRESHOLDS = ((20, 60), (30, 100), (40, 120), (50, 150))
GAUSSIAN_KERNELS = (3, 5, 7)

logger = logging.getLogger(__name__)


class MultiCardExtractor:
    """Detects multiple card-like rectangles via multi-pass Canny edge detection."""

    def extract(self, image: np.ndarray) -> List[CardCrop]:
        candidates = self._find_card_contours(image)
        deduplicated = self._nms(candidates)
        return self._warp_and_normalize(image, deduplicated)

    # ── Detection ────────────────────────────────────────────

    def _find_card_contours(
        self, image: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Return ``[(contour, quad_approx, contour_area), ...]``."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * MIN_AREA_FRACTION
        max_area = image_area * MAX_AREA_FRACTION

        all_candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []

        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        for lo, hi in CANNY_THRESHOLDS:
            self._collect_from_edges(
                bilateral, lo, hi, min_area, max_area, all_candidates,
            )

        for k in GAUSSIAN_KERNELS:
            blurred = cv2.GaussianBlur(gray, (k, k), 0)
            for lo, hi in CANNY_THRESHOLDS:
                self._collect_from_edges(
                    blurred, lo, hi, min_area, max_area, all_candidates,
                )

        return all_candidates

    def _collect_from_edges(
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
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )

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

    # ── Warp + normalize ─────────────────────────────────────

    def _warp_and_normalize(
        self,
        image: np.ndarray,
        candidates: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> List[CardCrop]:
        crops: List[Tuple[float, float, CardCrop]] = []

        for _, quad, _ in candidates:
            warp = perspective_warp(image, quad)
            if warp is None:
                continue

            pts = quad.reshape(4, 2).astype("float32")
            rect = order_points(pts)
            center_x = float(rect[:, 0].mean())
            center_y = float(rect[:, 1].mean())

            for card_img in self._split_merged_cards(warp):
                normalized = normalize_card(card_img)
                crops.append((center_y, center_x, CardCrop(image=normalized)))

        crops.sort(key=lambda t: (t[0], t[1]))
        return [c for _, _, c in crops]

    # ── Multi-card splitting ─────────────────────────────────

    @staticmethod
    def _split_merged_cards(warp: np.ndarray) -> List[np.ndarray]:
        """Split a warp that may contain multiple touching cards."""
        h, w = warp.shape[:2]
        if h < 1 or w < 1:
            return [warp]

        def _err(ratio: float) -> float:
            return abs(ratio - SINGLE_CARD_RATIO) / SINGLE_CARD_RATIO

        if h >= w:
            hw_ratio = h / w
            best_n, best_e = 0, float("inf")
            for n in range(1, MAX_MERGED_CARDS + 1):
                e = _err(hw_ratio / n)
                if e <= MULTI_CARD_TOLERANCE and e < best_e:
                    best_n, best_e = n, e

            if best_n == 0:
                return []
            if best_n == 1:
                return [warp]

            logger.debug(
                "splitting %dx%d portrait warp into %d stacked card(s)",
                w, h, best_n,
            )
            strip = h // best_n
            return [warp[i * strip : (i + 1) * strip, :]
                    for i in range(best_n)]

        wh_ratio = w / h
        inv_card = 1.0 / SINGLE_CARD_RATIO
        best_n, best_e = 0, float("inf")
        for n in range(2, MAX_MERGED_CARDS + 1):
            per_card = wh_ratio / n
            e = abs(per_card - inv_card) / inv_card
            if e <= MULTI_CARD_TOLERANCE and e < best_e:
                best_n, best_e = n, e

        if best_n == 0:
            return []

        logger.debug(
            "splitting %dx%d landscape warp into %d side-by-side card(s)",
            w, h, best_n,
        )
        strip = w // best_n
        return [warp[:, i * strip : (i + 1) * strip]
                for i in range(best_n)]
