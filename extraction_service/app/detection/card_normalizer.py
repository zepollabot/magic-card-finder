"""Shared post-processing for detected card crops.

Provides portrait orientation, fine deskew, optional super-resolution,
and final resize/sharpen to the standard MTG card size (672x936).
"""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

from .protocols import CardCrop

CARD_WIDTH = 672
CARD_HEIGHT = 936

_SR_WIDTH_THRESHOLD = 400

_sr_model = None

logger = logging.getLogger(__name__)


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


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def ensure_portrait(card: np.ndarray) -> np.ndarray:
    """Rotate 90 CW if the image came out landscape. All MTG cards are portrait."""
    h, w = card.shape[:2]
    if w > h:
        return cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    return card


def fine_deskew(card: np.ndarray) -> np.ndarray:
    """Correct small rotational tilt (< 15 deg) using HoughLinesP on the top strip."""
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


def apply_super_resolution(card: np.ndarray) -> np.ndarray:
    """Apply FSRCNN x2 upscale if the card is narrower than the threshold."""
    h_raw, w_raw = card.shape[:2]
    if w_raw >= _SR_WIDTH_THRESHOLD:
        return card
    sr = _get_sr_model()
    if sr is None:
        return card
    try:
        return sr.upsample(card)
    except Exception:
        return card


def normalize_size(card: np.ndarray) -> np.ndarray:
    """Resize to 672x936 with LANCZOS4 interpolation and unsharp-mask sharpen."""
    normalized = cv2.resize(
        card, (CARD_WIDTH, CARD_HEIGHT),
        interpolation=cv2.INTER_LANCZOS4,
    )
    blur = cv2.GaussianBlur(normalized, (0, 0), 3)
    normalized = cv2.addWeighted(normalized, 1.5, blur, -0.5, 0)
    return normalized


def normalize_card(card: np.ndarray) -> np.ndarray:
    """Full normalization pipeline: portrait -> deskew -> super-res -> resize."""
    card = ensure_portrait(card)
    card = fine_deskew(card)
    card = apply_super_resolution(card)
    card = normalize_size(card)
    return card


def perspective_warp(
    image: np.ndarray, quad: np.ndarray,
) -> Optional[np.ndarray]:
    """Perspective-warp a quadrilateral region out of *image*.

    *quad* must be an array of shape (4, 1, 2) or (4, 2).
    Returns the warped image, or ``None`` if dimensions are degenerate.
    """
    pts = quad.reshape(4, 2).astype("float32")
    rect = order_points(pts)

    width_a = np.linalg.norm(rect[2] - rect[3])
    width_b = np.linalg.norm(rect[1] - rect[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(rect[1] - rect[2])
    height_b = np.linalg.norm(rect[0] - rect[3])
    max_height = int(max(height_a, height_b))

    if max_width < 1 or max_height < 1:
        return None

    dst = np.array(
        [[0, 0], [max_width - 1, 0],
         [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def approximate_quad(cnt: np.ndarray) -> Optional[np.ndarray]:
    """Try to approximate *cnt* to a 4-point polygon, falling back to minAreaRect."""
    epsilons = (0.02, 0.03, 0.05, 0.08)
    peri = cv2.arcLength(cnt, True)
    for eps in epsilons:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return approx

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.intp(box).reshape(4, 1, 2)


def save_debug_crops(crops: List[CardCrop]) -> None:
    """Write card crops to OCR_DEBUG_DIR when set."""
    import uuid
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
