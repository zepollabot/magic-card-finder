"""OpenCV-based card detection with normalized 672x936 output.

Uses multiple Canny edge-detection passes (bilateral + gaussian blur,
several threshold pairs) combined with Non-Maximum Suppression to
reliably detect cards in real-world photos.
"""

import logging
import os
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .protocols import CardCrop, CardDetector

CARD_WIDTH = 672
CARD_HEIGHT = 936

# Width threshold below which FSRCNN x2 super-resolution is applied
# before normalizing to CARD_WIDTH x CARD_HEIGHT.
_SR_WIDTH_THRESHOLD = 400

_sr_model = None


def _get_sr_model():
    """Lazy-load the FSRCNN x2 super-resolution model (if available)."""
    global _sr_model
    if _sr_model is not None:
        return _sr_model
    model_path = os.getenv(
        "FSRCNN_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "FSRCNN_x2.pb"),
    )
    if not os.path.isfile(model_path):
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", 2)
        _sr_model = sr
        return sr
    except Exception:
        return None

SINGLE_CARD_RATIO = CARD_HEIGHT / CARD_WIDTH  # ~1.393
MULTI_CARD_TOLERANCE = 0.25
MAX_MERGED_CARDS = 4

MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.35
MIN_SOLIDITY = 0.65
ASPECT_RATIO_LO = 1.05
ASPECT_RATIO_HI = 7.0
APPROX_EPSILONS = (0.02, 0.03, 0.05, 0.08)
NMS_OVERLAP_THRESH = 0.50

CANNY_THRESHOLDS = ((20, 60), (30, 100), (40, 120), (50, 150))
GAUSSIAN_KERNELS = (3, 5, 7)

logger = logging.getLogger(__name__)


class OpenCVCardDetector:
    """
    Detects card-like rectangles via multi-pass Canny edge detection,
    perspective-warps them, then resizes to MTG standard 672x936.
    """

    def detect(self, image_bytes: bytes) -> List[CardCrop]:
        if not image_bytes:
            return []
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return []

        candidates = self._find_card_contours(image)
        deduplicated = self._nms(candidates)
        crops = self._warp_and_normalize(image, deduplicated)
        self._save_debug_crops(crops)
        return crops

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

            quad = self._approximate_quad(cnt)
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
            pts = quad.reshape(4, 2).astype("float32")
            rect = self._order_points(pts)

            width_a = np.linalg.norm(rect[2] - rect[3])
            width_b = np.linalg.norm(rect[1] - rect[0])
            max_width = int(max(width_a, width_b))

            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[0] - rect[3])
            max_height = int(max(height_a, height_b))

            if max_width < 1 or max_height < 1:
                continue

            dst = np.array(
                [[0, 0], [max_width - 1, 0],
                 [max_width - 1, max_height - 1], [0, max_height - 1]],
                dtype="float32",
            )

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(image, M, (max_width, max_height))

            center_x = float(rect[:, 0].mean())
            center_y = float(rect[:, 1].mean())

            for card_img in self._split_merged_cards(warp):
                card_img = self._ensure_portrait(card_img)
                card_img = self._fine_deskew(card_img)

                h_raw, w_raw = card_img.shape[:2]
                if w_raw < _SR_WIDTH_THRESHOLD:
                    sr = _get_sr_model()
                    if sr is not None:
                        try:
                            card_img = sr.upsample(card_img)
                        except Exception:
                            pass

                normalized = cv2.resize(
                    card_img, (CARD_WIDTH, CARD_HEIGHT),
                    interpolation=cv2.INTER_LANCZOS4,
                )

                blur = cv2.GaussianBlur(normalized, (0, 0), 3)
                normalized = cv2.addWeighted(normalized, 1.5, blur, -0.5, 0)

                crops.append((center_y, center_x, CardCrop(image=normalized)))

        crops.sort(key=lambda t: (t[0], t[1]))
        return [c for _, _, c in crops]

    # ── Card orientation ──────────────────────────────────────

    @staticmethod
    def _ensure_portrait(card: np.ndarray) -> np.ndarray:
        """Rotate 90 CW if the warp came out landscape.

        All MTG cards are portrait.  A landscape result means
        ``_order_points`` swapped width/height for a tilted card.
        """
        h, w = card.shape[:2]
        if w > h:
            return cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
        return card

    @staticmethod
    def _fine_deskew(card: np.ndarray) -> np.ndarray:
        """Correct small rotational tilt (< 10 deg) left by the warp.

        Uses HoughLinesP on Canny edges of the top 25 % of the card
        (where the title-bar frame lines are) to find the dominant
        near-horizontal angle, then applies an affine rotation to
        straighten the image.
        """
        h, w = card.shape[:2]
        if h < 40 or w < 40:
            return card

        strip_h = max(10, int(h * 0.25))
        strip = card[:strip_h, :]

        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if len(strip.shape) == 3 else strip
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30,
            minLineLength=w // 4,
            maxLineGap=10,
        )
        if lines is None or len(lines) == 0:
            return card

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dx) < 1:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) <= 15:
                angles.append(angle)

        if not angles:
            return card

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return card

        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            card, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    # ── Multi-card splitting ─────────────────────────────────

    @staticmethod
    def _split_merged_cards(warp: np.ndarray) -> List[np.ndarray]:
        """Split a warp that may contain multiple touching cards.

        Uses actual height and width (no rotation) to decide:

        * **Portrait** (h > w) — ``h/w`` is checked against
          ``N * 1.393`` to find N vertically-stacked cards.
        * **Landscape** (w ≥ h) — ``w/h`` is checked against
          ``N * (1/1.393)`` to find N side-by-side portrait cards.

        Returns portrait card images; rejects warps that don't match
        any valid single- or multi-card pattern.
        """
        h, w = warp.shape[:2]
        if h < 1 or w < 1:
            return [warp]

        def _err(ratio: float) -> float:
            return abs(ratio - SINGLE_CARD_RATIO) / SINGLE_CARD_RATIO

        if h >= w:
            # Portrait: check h/w against N * SINGLE_CARD_RATIO
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

        # Landscape: check w/h against N * (1 / SINGLE_CARD_RATIO)
        wh_ratio = w / h
        inv_card = 1.0 / SINGLE_CARD_RATIO  # ~0.718
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

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _approximate_quad(cnt: np.ndarray) -> Optional[np.ndarray]:
        peri = cv2.arcLength(cnt, True)
        for eps in APPROX_EPSILONS:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return np.intp(box).reshape(4, 1, 2)

    @staticmethod
    def _save_debug_crops(crops: List[CardCrop]) -> None:
        debug_dir = os.getenv("OCR_DEBUG_DIR", "").strip()
        if not debug_dir:
            return
        try:
            os.makedirs(debug_dir, exist_ok=True)
            batch = uuid.uuid4().hex[:8]
            for idx, crop in enumerate(crops):
                path = os.path.join(debug_dir, f"card_crop_{batch}_{idx}.png")
                cv2.imwrite(path, crop.image)
            logger.debug("saved %d card crop(s) to %s", len(crops), debug_dir)
        except Exception:
            pass

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
