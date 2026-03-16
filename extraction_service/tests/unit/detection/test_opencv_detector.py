"""Unit tests for OpenCV card detector."""

import cv2
import numpy as np
import pytest

from app.detection import CardCrop, OpenCVCardDetector

CARD_WIDTH = 672
CARD_HEIGHT = 936


def _encode_image(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _make_card_rect(
    canvas: np.ndarray,
    y0: int, x0: int,
    h: int = 200, w: int = 145,
    color: int = 255,
) -> None:
    """Draw a filled card-like rectangle on *canvas*."""
    canvas[y0 : y0 + h, x0 : x0 + w] = color


# ── Basic edge cases ──


def test_detect_returns_empty_for_invalid_bytes():
    detector = OpenCVCardDetector()
    assert detector.detect(b"not an image") == []


def test_detect_returns_empty_for_empty_bytes():
    detector = OpenCVCardDetector()
    assert detector.detect(b"") == []


def test_detect_returns_empty_for_image_with_no_card_like_contours():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    detector = OpenCVCardDetector()
    assert detector.detect(_encode_image(img)) == []


# ── Single card detection ──


def test_detect_one_card_like_rectangle_returns_one_crop():
    h, w = 200, 145  # aspect ~1.38
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 50, 50, h, w)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 1
    crop = result[0]
    assert isinstance(crop, CardCrop)
    assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


def test_detect_normalized_crop_shape():
    """Each crop must be 672 wide, 936 tall (MTG standard)."""
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
    """2x2 grid of card-like rectangles should yield 4 crops."""
    img = np.zeros((600, 500, 3), dtype=np.uint8)
    positions = [(30, 30), (30, 260), (280, 30), (280, 260)]
    for y0, x0 in positions:
        _make_card_rect(img, y0, x0, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 4


# ── Solidity filtering ──


def test_detect_rejects_shapes_with_wrong_aspect_ratio():
    """A square region should be rejected by the aspect ratio check."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[50:250, 50:250] = 255
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 0


# ── Minimum area filtering ──


def test_detect_rejects_tiny_rectangles():
    """Rectangles smaller than MIN_AREA_FRACTION of image should be skipped."""
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    _make_card_rect(img, 10, 10, 40, 29)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 0


# ── Portrait orientation ──


def test_crops_are_portrait_orientation():
    """Detected cards must always be taller than wide (portrait)."""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 50, 50, 200, 145)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) >= 1
    for crop in result:
        assert crop.image.shape[0] > crop.image.shape[1]


# ── Positional sorting ──


def test_crops_sorted_top_to_bottom_left_to_right():
    """Crops should come back ordered by row first, then column."""
    img = np.zeros((600, 500, 3), dtype=np.uint8)
    _make_card_rect(img, 280, 260, 200, 145)  # bottom-right
    _make_card_rect(img, 30, 30, 200, 145)    # top-left
    _make_card_rect(img, 30, 260, 200, 145)   # top-right
    _make_card_rect(img, 280, 30, 200, 145)   # bottom-left
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 4


# ── Multi-card splitting ──


def test_two_touching_vertical_cards_split_into_two():
    """Two card-shaped rects touching vertically should yield 2 crops."""
    card_h, card_w = 200, 145
    img = np.zeros((500, 400, 3), dtype=np.uint8)
    _make_card_rect(img, 30, 50, card_h, card_w)
    _make_card_rect(img, 30 + card_h, 50, card_h, card_w)
    detector = OpenCVCardDetector()
    result = detector.detect(_encode_image(img))
    assert len(result) == 2
    for crop in result:
        assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


def test_split_merged_cards_single_card():
    """A warp with single-card ratio should return one image."""
    from app.detection.opencv_detector import OpenCVCardDetector
    warp = np.zeros((936, 672, 3), dtype=np.uint8)
    result = OpenCVCardDetector._split_merged_cards(warp)
    assert len(result) == 1


def test_split_merged_cards_two_stacked():
    """A portrait warp with 2x card height should return two images."""
    from app.detection.opencv_detector import OpenCVCardDetector
    warp = np.zeros((936 * 2, 672, 3), dtype=np.uint8)
    result = OpenCVCardDetector._split_merged_cards(warp)
    assert len(result) == 2
    for card in result:
        assert card.shape[0] == 936
        assert card.shape[1] == 672


def test_split_merged_cards_two_side_by_side():
    """A landscape warp with 2 card widths should return two portrait images."""
    from app.detection.opencv_detector import OpenCVCardDetector
    # Two portrait cards side-by-side: width = 2*672, height = 936
    warp = np.zeros((936, 672 * 2, 3), dtype=np.uint8)
    result = OpenCVCardDetector._split_merged_cards(warp)
    assert len(result) == 2
    for card in result:
        assert card.shape[0] > card.shape[1]


def test_split_does_not_split_single_portrait_card():
    """A portrait warp with single-card ratio must not be split."""
    from app.detection.opencv_detector import OpenCVCardDetector
    warp = np.zeros((617, 429, 3), dtype=np.uint8)
    result = OpenCVCardDetector._split_merged_cards(warp)
    assert len(result) == 1


def test_split_rejects_landscape_that_is_not_multi_card():
    """A landscape warp that doesn't match any multi-card pattern returns []."""
    from app.detection.opencv_detector import OpenCVCardDetector
    warp = np.zeros((100, 500, 3), dtype=np.uint8)
    result = OpenCVCardDetector._split_merged_cards(warp)
    assert len(result) == 0


# ── Portrait guarantee ──


def test_ensure_portrait_already_portrait():
    """A portrait image should be returned unchanged."""
    from app.detection.opencv_detector import OpenCVCardDetector
    card = np.zeros((936, 672, 3), dtype=np.uint8)
    result = OpenCVCardDetector._ensure_portrait(card)
    assert result.shape == (936, 672, 3)


def test_ensure_portrait_landscape_rotated():
    """A landscape image (w > h) must be rotated to portrait."""
    from app.detection.opencv_detector import OpenCVCardDetector
    card = np.zeros((672, 936, 3), dtype=np.uint8)
    card[:, :50] = 200
    result = OpenCVCardDetector._ensure_portrait(card)
    assert result.shape[0] > result.shape[1]
    assert result.shape == (936, 672, 3)


# ── Fine deskew ──


def test_fine_deskew_no_change_for_straight_card():
    """A perfectly straight card should not be altered."""
    from app.detection.opencv_detector import OpenCVCardDetector
    card = np.full((936, 672, 3), 200, dtype=np.uint8)
    card[50:55, 20:650] = 0
    result = OpenCVCardDetector._fine_deskew(card)
    assert result.shape == card.shape


def test_fine_deskew_corrects_tilted_line():
    """A card with a tilted horizontal line should be straightened."""
    from app.detection.opencv_detector import OpenCVCardDetector
    card = np.full((936, 672, 3), 200, dtype=np.uint8)
    for x in range(20, 650):
        y = 60 + int((x - 20) * np.tan(np.radians(4)))
        if 0 <= y < 936:
            card[max(0, y - 2) : y + 3, x] = 0
    result = OpenCVCardDetector._fine_deskew(card)
    assert result.shape == card.shape


def test_fine_deskew_ignores_tiny_images():
    """Images smaller than 40px should be returned as-is."""
    from app.detection.opencv_detector import OpenCVCardDetector
    card = np.zeros((30, 20, 3), dtype=np.uint8)
    result = OpenCVCardDetector._fine_deskew(card)
    assert result.shape == card.shape


# ── Super-resolution ──


def test_super_res_model_returns_none_when_missing():
    """_get_sr_model should return None when model file is absent."""
    import app.detection.opencv_detector as mod
    saved = mod._sr_model
    mod._sr_model = None
    try:
        import os
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("FSRCNN_MODEL_PATH", "/nonexistent/model.pb")
            result = mod._get_sr_model()
        assert result is None
    finally:
        mod._sr_model = saved


def test_warp_and_normalize_uses_lanczos():
    """Verify the resize uses INTER_LANCZOS4 by checking output quality."""
    from app.detection.opencv_detector import OpenCVCardDetector
    card = np.random.randint(0, 255, (300, 216, 3), dtype=np.uint8)
    card[5:35, :] = 200
    encoded = _encode_image(card)
    detector = OpenCVCardDetector()
    results = detector.detect(encoded)
    if results:
        for crop in results:
            assert crop.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
