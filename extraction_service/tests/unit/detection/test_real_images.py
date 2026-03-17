"""Integration-style tests using real example images.

These tests verify that the full detection pipeline (classify -> extract)
works correctly on actual card photos.  They are skipped when the example
images are not present (e.g. in CI without the data directory).
"""

import os
from pathlib import Path

import pytest

from app.detection import OpenCVCardDetector
from app.detection.card_normalizer import CARD_HEIGHT, CARD_WIDTH
from app.detection.classifier import ImageClassifier, ImageType

EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "data" / "example"

SINGLE_IMAGES = ["single_1.jpeg", "single_2.jpeg", "single_3.jpeg", "single_4.jpeg"]
MULTI_IMAGES = ["multi_1.jpg", "multi_2.jpg"]

_skip_no_examples = pytest.mark.skipif(
    not EXAMPLE_DIR.is_dir(),
    reason="Example images not available",
)


def _load_bytes(filename: str) -> bytes:
    path = EXAMPLE_DIR / filename
    return path.read_bytes()


def _load_cv2(filename: str):
    import cv2
    import numpy as np

    data = _load_bytes(filename)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# ── Classifier tests ─────────────────────────────────────────


@_skip_no_examples
@pytest.mark.parametrize("filename", SINGLE_IMAGES)
def test_classifier_identifies_single_card_images(filename):
    image = _load_cv2(filename)
    classifier = ImageClassifier()
    result = classifier.classify(image)
    assert result == ImageType.SINGLE, f"{filename} should be classified as SINGLE"


@_skip_no_examples
@pytest.mark.parametrize("filename", MULTI_IMAGES)
def test_classifier_identifies_multi_card_images(filename):
    image = _load_cv2(filename)
    classifier = ImageClassifier()
    result = classifier.classify(image)
    assert result == ImageType.MULTI, f"{filename} should be classified as MULTI"


# ── Single-card detection ────────────────────────────────────


@_skip_no_examples
@pytest.mark.parametrize("filename", SINGLE_IMAGES)
def test_single_card_image_returns_exactly_one_crop(filename):
    detector = OpenCVCardDetector()
    image_bytes = _load_bytes(filename)
    crops = detector.detect(image_bytes)
    assert len(crops) == 1, (
        f"{filename}: expected 1 crop, got {len(crops)}"
    )
    assert crops[0].image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


@_skip_no_examples
@pytest.mark.parametrize("filename", SINGLE_IMAGES)
def test_single_card_crop_is_portrait(filename):
    detector = OpenCVCardDetector()
    crops = detector.detect(_load_bytes(filename))
    assert len(crops) >= 1
    for crop in crops:
        h, w = crop.image.shape[:2]
        assert h > w, f"{filename}: crop should be portrait ({h}x{w})"


# ── Multi-card detection ─────────────────────────────────────


@_skip_no_examples
def test_multi_1_returns_multiple_crops():
    """multi_1.jpg contains 8 cards in a 4x2 grid."""
    detector = OpenCVCardDetector()
    crops = detector.detect(_load_bytes("multi_1.jpg"))
    assert len(crops) >= 4, (
        f"multi_1.jpg: expected at least 4 crops, got {len(crops)}"
    )
    for crop in crops:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


@_skip_no_examples
def test_multi_2_returns_multiple_crops():
    """multi_2.jpg contains 9 cards in a 3x3 grid."""
    detector = OpenCVCardDetector()
    crops = detector.detect(_load_bytes("multi_2.jpg"))
    assert len(crops) >= 4, (
        f"multi_2.jpg: expected at least 4 crops, got {len(crops)}"
    )
    for crop in crops:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


# ── All crops are valid ──────────────────────────────────────


@_skip_no_examples
@pytest.mark.parametrize("filename", SINGLE_IMAGES + MULTI_IMAGES)
def test_all_crops_have_correct_shape(filename):
    detector = OpenCVCardDetector()
    crops = detector.detect(_load_bytes(filename))
    assert len(crops) >= 1, f"{filename}: no crops detected"
    for crop in crops:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
