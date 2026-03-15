"""Unit tests for Tesseract card recognizer."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.ocr import TesseractCardRecognizer
from app.ocr.tesseract_recognizer import (
    BORDER_PX,
    DEFAULT_LANGS,
    HORIZONTAL_INSET,
    TITLE_ZONE_BOTTOM,
    TITLE_ZONE_TOP,
    UPSCALE,
)


def _make_card_image(height: int = 936, width: int = 672) -> np.ndarray:
    """Minimal BGR card-sized image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _encode_image(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def test_default_lang_is_multilanguage():
    recognizer = TesseractCardRecognizer()
    assert recognizer.lang == DEFAULT_LANGS
    assert "ita" in recognizer.lang
    assert "eng" in recognizer.lang


def test_custom_lang_override():
    recognizer = TesseractCardRecognizer(lang="jpn")
    assert recognizer.lang == "jpn"


def test_recognize_returns_empty_for_invalid_bytes():
    recognizer = TesseractCardRecognizer()
    result = recognizer.recognize(b"not an image")
    assert result == ""


def test_recognize_returns_empty_for_none_decode():
    recognizer = TesseractCardRecognizer()
    with patch("app.ocr.tesseract_recognizer.cv2.imdecode", return_value=None):
        result = recognizer.recognize(b"fake")
    assert result == ""


def test_recognize_crops_title_zone_and_calls_tesseract():
    recognizer = TesseractCardRecognizer()
    img = _make_card_image()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Lightning Bolt"
        result = recognizer.recognize(img)
    assert result == "Lightning Bolt"
    mock_ts.assert_called_once()
    call_args = mock_ts.call_args
    preprocessed = call_args[0][0]
    title_h = int(936 * (TITLE_ZONE_BOTTOM - TITLE_ZONE_TOP))
    max_expected_h = title_h * UPSCALE + 2 * BORDER_PX + 10
    assert preprocessed.shape[0] <= max_expected_h


def test_recognize_accepts_bytes():
    recognizer = TesseractCardRecognizer()
    img = _make_card_image()
    img_bytes = _encode_image(img)
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Island"
        result = recognizer.recognize(img_bytes)
    assert result == "Island"


def test_clean_text_collapses_whitespace():
    recognizer = TesseractCardRecognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "  Llanowar   Elves  \n"
        result = recognizer.recognize(_make_card_image())
    assert result == "Llanowar Elves"


def test_recognize_returns_empty_string_when_tesseract_returns_empty():
    recognizer = TesseractCardRecognizer()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = ""
        result = recognizer.recognize(_make_card_image())
    assert result == ""


def test_preprocess_adds_white_border():
    recognizer = TesseractCardRecognizer()
    img = _make_card_image()
    roi = recognizer._crop_title_zone(img)
    preprocessed = recognizer._preprocess(roi)
    assert preprocessed[0, 0] == 255
    assert preprocessed[-1, -1] == 255


def test_crop_title_zone_applies_horizontal_inset():
    recognizer = TesseractCardRecognizer()
    img = _make_card_image()
    roi = recognizer._crop_title_zone(img)
    expected_w = 672 - 2 * int(672 * HORIZONTAL_INSET)
    assert roi.shape[1] == expected_w


def test_tesseract_config_includes_oem_and_psm():
    recognizer = TesseractCardRecognizer()
    img = _make_card_image()
    with patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string") as mock_ts:
        mock_ts.return_value = "Test"
        recognizer.recognize(img)
    _, kwargs = mock_ts.call_args
    assert "--psm 7" in kwargs["config"]
    assert "--oem 3" in kwargs["config"]


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
    recognizer = TesseractCardRecognizer()
    img = _make_card_image()
    y0 = int(936 * TITLE_ZONE_TOP)
    y1 = int(936 * TITLE_ZONE_BOTTOM)
    img[y0:y1, 40:400] = 255
    result = recognizer.recognize(img)
    assert isinstance(result, str)
