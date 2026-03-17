"""Unit tests for OpenCV card detector (orchestrator level).

Tests here exercise the full detect() pipeline through OpenCVCardDetector.
Lower-level tests for individual components live in their own test files:
  - test_card_normalizer.py
  - test_classifier.py
  - test_single_card_extractor.py
"""

import cv2
import numpy as np
import pytest

from app.detection import CardCrop, OpenCVCardDetector
from app.detection.card_normalizer import CARD_HEIGHT, CARD_WIDTH
from app.detection.multi_card_extractor import MultiCardExtractor


def _encode_image(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _make_card_rect(
    canvas: np.ndarray,
    y0: int, x0: int,
    h: int = 200, w: int = 145,
    color: int = 255,
) -> None:
    canvas[y0 : y0 + h, x0 : x0 + w] = color


# ── Basic edge cases ──


def test_detect_returns_empty_for_invalid_bytes():
    detector = OpenCVCardDetector()
    assert detector.detect(b"not an image") == []


def test_detect_returns_empty_for_empty_bytes():
    detector = OpenCVCardDetector()
    assert detector.detect(b"") == []


def test_detect_returns_crop_for_featureless_image():
    """A featureless image is classified as SINGLE and the whole-image
    fallback produces exactly one crop."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 1
    assert result[0].image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


# ── Single card detection ──


def test_detect_one_card_like_rectangle_returns_one_crop():
    h, w = 200, 145
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 50, 50, h, w)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 1
    crop = result[0]
    assert isinstance(crop, CardCrop)
    assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


def test_detect_normalized_crop_shape():
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 60, 80, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) >= 1
    for crop in result:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


# ── Multiple card detection ──


def test_detect_two_disjoint_rectangles_returns_two_crops():
    img = np.zeros((500, 600, 3), dtype=np.uint8)
    _make_card_rect(img, 30, 30, 200, 145)
    _make_card_rect(img, 30, 300, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 2
    for crop in result:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


def test_detect_four_cards_in_grid():
    img = np.zeros((600, 500, 3), dtype=np.uint8)
    positions = [(30, 30), (30, 260), (280, 30), (280, 260)]
    for y0, x0 in positions:
        _make_card_rect(img, y0, x0, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 4


# ── Single card: large card in image ──


def test_detect_large_card_returns_one_crop():
    """A large card (>35% area) should be classified SINGLE and extracted."""
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    _make_card_rect(img, 30, 30, h=540, w=340)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 1
    assert result[0].image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


# ── Portrait orientation ──


def test_crops_are_portrait_orientation():
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 50, 50, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) >= 1
    for crop in result:
        assert crop.image.shape[0] > crop.image.shape[1]


# ── Positional sorting ──


def test_crops_sorted_top_to_bottom_left_to_right():
    img = np.zeros((600, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 280, 260, 200, 145)
    _make_card_rect(img, 30, 30, 200, 145)
    _make_card_rect(img, 30, 260, 200, 145)
    _make_card_rect(img, 280, 30, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 4


# ── Multi-card splitting (tested via MultiCardExtractor) ──


def test_split_merged_cards_single_card():
    warp = np.zeros((936, 672, 3), dtype=np.uint8)
    result = MultiCardExtractor._split_merged_cards(warp)
    assert len(result) == 1


def test_split_merged_cards_two_stacked():
    warp = np.zeros((936 * 2, 672, 3), dtype=np.uint8)
    result = MultiCardExtractor._split_merged_cards(warp)
    assert len(result) == 2
    for card in result:
        assert card.shape[0] == 936
        assert card.shape[1] == 672


def test_split_merged_cards_two_side_by_side():
    warp = np.zeros((936, 672 * 2, 3), dtype=np.uint8)
    result = MultiCardExtractor._split_merged_cards(warp)
    assert len(result) == 2
    for card in result:
        assert card.shape[0] > card.shape[1]


def test_split_does_not_split_single_portrait_card():
    warp = np.zeros((617, 429, 3), dtype=np.uint8)
    result = MultiCardExtractor._split_merged_cards(warp)
    assert len(result) == 1


def test_split_rejects_landscape_that_is_not_multi_card():
    warp = np.zeros((100, 500, 3), dtype=np.uint8)
    result = MultiCardExtractor._split_merged_cards(warp)
    assert len(result) == 0


# ── Super-resolution model ──


def test_super_res_model_returns_none_when_missing():
    import app.detection.card_normalizer as mod
    saved = mod._sr_model
    mod._sr_model = None
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("FSRCNN_MODEL_PATH", "/nonexistent/model.pb")
            result = mod._get_sr_model()
        assert result is None
    finally:
        mod._sr_model = saved


def test_warp_and_normalize_uses_lanczos():
    card = np.random.randint(0, 255, (300, 216, 3), dtype=np.uint8)
    card[5:35, :] = 200
    encoded = _encode_image(card)
    detector = OpenCVCardDetector()
    results = detector.detect(encoded)
    if results:
        for crop in results:
            assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
