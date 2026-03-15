"""Unit tests for OpenCV card detector."""
import io

import cv2
import numpy as np
import pytest

from app.detection import CardCrop, OpenCVCardDetector

CARD_WIDTH = 672
CARD_HEIGHT = 936


def _encode_image(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def test_detect_returns_empty_for_invalid_bytes():
    detector = OpenCVCardDetector()
    result = detector.detect(b"not an image")
    assert result == []


def test_detect_returns_empty_for_empty_bytes():
    detector = OpenCVCardDetector()
    result = detector.detect(b"")
    assert result == []


def test_detect_returns_empty_for_image_with_no_card_like_contours():
    # Solid gray image with no quadrilateral
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert result == []


def test_detect_one_card_like_rectangle_returns_one_crop():
    # White rectangle on black (card-like aspect ~1.45)
    # MTG aspect 2.5:3.5 -> ~0.714 width/height -> w/h ~1.4
    h, w = 200, 290  # aspect ~1.45
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    y0, x0 = 50, 50
    img[y0 : y0 + h, x0 : x0 + w] = 255
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 1
    crop = result[0]
    assert isinstance(crop, CardCrop)
    assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


def test_detect_normalized_crop_shape():
    """Each crop must be 672 wide, 936 tall (MTG standard)."""
    h, w = 180, 260
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[60 : 60 + h, 80 : 80 + w] = 255
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) >= 1
    for crop in result:
        assert crop.image.shape[0] == CARD_HEIGHT
        assert crop.image.shape[1] == CARD_WIDTH
        assert crop.image.shape[2] == 3


def test_detect_two_disjoint_rectangles_returns_two_crops():
    img = np.zeros((500, 600, 3), dtype=np.uint8)
    # Two card-like rectangles
    for (y0, x0) in [(30, 30), (30, 320)]:
        img[y0 : y0 + 200, x0 : x0 + 290] = 255
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 2
    for crop in result:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
