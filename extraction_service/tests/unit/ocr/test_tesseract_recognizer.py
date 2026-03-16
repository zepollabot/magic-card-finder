"""Unit tests for Tesseract card recognizer and title region extractor."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.ocr import CardTitleRegionExtractor, TesseractCardRecognizer
from app.ocr.tesseract_recognizer import DEFAULT_LANGS
from app.ocr.title_region import (
    BORDER_PX,
    TITLE_LEFT_INSET,
    TITLE_RIGHT_INSET,
    TITLE_ZONE_BOTTOM,
    TITLE_ZONE_TOP,
    UPSCALE,
)

CARD_HEIGHT = 936
CARD_WIDTH = 672


def _make_card_image(height: int = CARD_HEIGHT, width: int = CARD_WIDTH) -> np.ndarray:
    """Minimal BGR card-sized image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _encode_image(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _make_recognizer() -> TesseractCardRecognizer:
    return TesseractCardRecognizer(title_extractor=CardTitleRegionExtractor())


# --- TesseractCardRecognizer tests ---


def test_default_lang_is_multilanguage():
    recognizer = _make_recognizer()
    assert recognizer.lang == DEFAULT_LANGS
    assert "ita" in recognizer.lang
    assert "eng" in recognizer.lang


def test_custom_lang_override():
    recognizer = TesseractCardRecognizer(
        title_extractor=CardTitleRegionExtractor(), lang="jpn"
    )
    assert recognizer.lang == "jpn"


def test_recognize_returns_empty_for_invalid_bytes():
    recognizer = _make_recognizer()
    result = recognizer.recognize(b"not an image")
    assert result == ""


def test_recognize_returns_empty_for_none_decode():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.cv2.imdecode", return_value=None):
        result = recognizer.recognize(b"fake")
    assert result == ""


def test_recognize_calls_tesseract():
    recognizer = _make_recognizer()
    img = _make_card_image()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Lightning Bolt"
        result = recognizer.recognize(img)
    assert result == "Lightning Bolt"
    mock_ts.assert_called_once()


def test_recognize_accepts_bytes():
    recognizer = _make_recognizer()
    img_bytes = _encode_image(_make_card_image())
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Island"
        result = recognizer.recognize(img_bytes)
    assert result == "Island"


def test_clean_text_collapses_whitespace():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "  Llanowar   Elves  \n"
        result = recognizer.recognize(_make_card_image())
    assert result == "Llanowar Elves"


def test_clean_text_strips_trailing_mana_noise():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Nibbio Infernale Furioso 4 è\n"
        result = recognizer.recognize(_make_card_image())
    assert result == "Nibbio Infernale Furioso"


def test_clean_text_strips_leading_noise():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "fi Nibbio Infernale di Bogardan\n"
        result = recognizer.recognize(_make_card_image())
    assert result == "Nibbio Infernale di Bogardan"


def test_clean_text_picks_best_line():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "wee Se Si Se CO\nNibbio Infernale Furioso 4.\n"
        result = recognizer.recognize(_make_card_image())
    assert result == "Nibbio Infernale Furioso"


def test_recognize_returns_empty_when_tesseract_returns_empty():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = ""
        result = recognizer.recognize(_make_card_image())
    assert result == ""


def test_tesseract_config_includes_oem_and_psm():
    recognizer = _make_recognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Test"
        recognizer.recognize(_make_card_image())
    _, kwargs = mock_ts.call_args
    assert "--psm 6" in kwargs["config"]
    assert "--oem 3" in kwargs["config"]


# --- CardTitleRegionExtractor tests ---


def test_extract_returns_tuple():
    """extract() must return (preprocessed, roi)."""
    extractor = CardTitleRegionExtractor()
    result = extractor.extract(_make_card_image())
    assert isinstance(result, tuple)
    assert len(result) == 2
    preprocessed, roi = result
    assert isinstance(preprocessed, np.ndarray)
    assert isinstance(roi, np.ndarray)


def test_preprocess_adds_white_border():
    extractor = CardTitleRegionExtractor()
    preprocessed, _ = extractor.extract(_make_card_image())
    assert preprocessed[0, 0] == 255
    assert preprocessed[-1, -1] == 255


def test_crop_title_zone_excludes_right_side():
    """The right portion (mana symbols) should be cropped out."""
    extractor = CardTitleRegionExtractor()
    roi = extractor._crop_title_zone(_make_card_image())
    expected_w = (
        CARD_WIDTH - int(CARD_WIDTH * TITLE_LEFT_INSET) - int(CARD_WIDTH * TITLE_RIGHT_INSET)
    )
    assert roi.shape[1] == expected_w


def test_preprocess_output_is_2d():
    """Preprocessed output should be a single-channel (grayscale) image."""
    extractor = CardTitleRegionExtractor()
    preprocessed, _ = extractor.extract(_make_card_image())
    assert len(preprocessed.shape) == 2


def test_preprocess_upscales():
    extractor = CardTitleRegionExtractor()
    roi = extractor._crop_title_zone(_make_card_image())
    preprocessed, _ = extractor.extract(_make_card_image())
    expected_h = roi.shape[0] * UPSCALE + 2 * BORDER_PX
    assert preprocessed.shape[0] == expected_h


def test_roi_matches_title_zone_crop():
    """The roi returned by extract() should match _crop_title_zone()."""
    extractor = CardTitleRegionExtractor()
    _, roi = extractor.extract(_make_card_image())
    direct_crop = extractor._crop_title_zone(_make_card_image())
    np.testing.assert_array_equal(roi, direct_crop)


def test_preprocess_light_background():
    """Light title bar with dark text should produce 2D white-bg output."""
    extractor = CardTitleRegionExtractor()
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), 30, dtype=np.uint8)
    y0 = int(CARD_HEIGHT * TITLE_ZONE_TOP)
    y1 = int(CARD_HEIGHT * TITLE_ZONE_BOTTOM)
    card[y0:y1, 40:500] = 200
    card[y0 + 5 : y0 + 15, 60:250] = 40
    preprocessed, _ = extractor.extract(card)
    assert isinstance(preprocessed, np.ndarray)
    assert len(preprocessed.shape) == 2
    assert np.mean(preprocessed) > 128


def test_preprocess_dark_background():
    """Dark title bar with light text should produce 2D white-bg output."""
    extractor = CardTitleRegionExtractor()
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), 40, dtype=np.uint8)
    y0 = int(CARD_HEIGHT * TITLE_ZONE_TOP)
    y1 = int(CARD_HEIGHT * TITLE_ZONE_BOTTOM)
    card[y0 + 5 : y0 + 15, 60:250] = 200
    preprocessed, _ = extractor.extract(card)
    assert isinstance(preprocessed, np.ndarray)
    assert len(preprocessed.shape) == 2
    assert np.mean(preprocessed) > 128


def test_preprocess_denoises():
    """Preprocessed output should still be valid after denoising step."""
    extractor = CardTitleRegionExtractor()
    card = _make_card_image()
    y0 = int(CARD_HEIGHT * TITLE_ZONE_TOP)
    y1 = int(CARD_HEIGHT * TITLE_ZONE_BOTTOM)
    card[y0:y1, 40:400] = 180
    np.random.seed(42)
    noise = np.random.randint(0, 30, card[y0:y1, 40:400].shape, dtype=np.uint8)
    card[y0:y1, 40:400] = np.clip(card[y0:y1, 40:400].astype(int) + noise, 0, 255).astype(np.uint8)
    preprocessed, _ = extractor.extract(card)
    assert isinstance(preprocessed, np.ndarray)
    assert len(preprocessed.shape) == 2


# --- Integration test (requires Tesseract installed) ---


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract not installed")
def test_recognize_integration_real_tesseract():
    """With real Tesseract: white title bar may be read as empty or garbage."""
    recognizer = _make_recognizer()
    img = _make_card_image()
    y0 = int(CARD_HEIGHT * TITLE_ZONE_TOP)
    y1 = int(CARD_HEIGHT * TITLE_ZONE_BOTTOM)
    img[y0:y1, 40:400] = 255
    result = recognizer.recognize(img)
    assert isinstance(result, str)
