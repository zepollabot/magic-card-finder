"""Unit tests for TesseractCardRecognizer."""
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.ocr.tesseract_recognizer import (
    TesseractCardRecognizer,
    _pick_best_line,
    _trailing_junk_re,
)


class FakePreprocessor:
    """Returns the input as both preprocessed and roi."""

    def __init__(self):
        self.called_with = None

    def preprocess(self, image):
        self.called_with = image
        return image, image


def _make_image(h=40, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestTesseractCardRecognizer:
    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_recognize_calls_preprocessor(self, mock_tess):
        mock_tess.return_value = "Lightning Bolt"
        preprocessor = FakePreprocessor()
        recognizer = TesseractCardRecognizer(preprocessor=preprocessor)
        image = _make_image()

        recognizer.recognize(image)

        np.testing.assert_array_equal(preprocessor.called_with, image)

    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_recognize_returns_cleaned_text(self, mock_tess):
        mock_tess.return_value = "  Lightning  Bolt  \n"
        recognizer = TesseractCardRecognizer(preprocessor=FakePreprocessor())

        result = recognizer.recognize(_make_image())

        assert result == "Lightning Bolt"

    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_clean_text_strips_mana_symbols(self, mock_tess):
        mock_tess.return_value = "Lightning Bolt 2R"
        recognizer = TesseractCardRecognizer(preprocessor=FakePreprocessor())

        result = recognizer.recognize(_make_image())

        assert result == "Lightning Bolt"

    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_clean_text_strips_leading_noise(self, mock_tess):
        mock_tess.return_value = "## Lightning Bolt"
        recognizer = TesseractCardRecognizer(preprocessor=FakePreprocessor())

        result = recognizer.recognize(_make_image())

        assert result == "Lightning Bolt"

    def test_empty_image_returns_empty(self):
        recognizer = TesseractCardRecognizer(preprocessor=FakePreprocessor())

        assert recognizer.recognize(None) == ""
        assert recognizer.recognize(np.array([])) == ""

    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_debug_saves_files_when_enabled(self, mock_tess, tmp_path):
        mock_tess.return_value = "Test Card"
        recognizer = TesseractCardRecognizer(
            preprocessor=FakePreprocessor(),
            debug_dir=str(tmp_path),
        )

        recognizer.recognize(_make_image())

        files = list(tmp_path.iterdir())
        roi_files = [f for f in files if f.name.startswith("title_roi_")]
        prep_files = [f for f in files if f.name.startswith("title_preprocessed_")]
        assert len(roi_files) == 1
        assert len(prep_files) == 1

    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_debug_skipped_when_disabled(self, mock_tess, tmp_path):
        mock_tess.return_value = "Test Card"
        recognizer = TesseractCardRecognizer(
            preprocessor=FakePreprocessor(),
            debug_dir="",
        )

        recognizer.recognize(_make_image())

        assert len(list(tmp_path.iterdir())) == 0

    @patch("app.ocr.tesseract_recognizer.pytesseract.image_to_string")
    def test_tesseract_exception_returns_empty(self, mock_tess):
        mock_tess.side_effect = RuntimeError("tesseract crashed")
        recognizer = TesseractCardRecognizer(preprocessor=FakePreprocessor())

        assert recognizer.recognize(_make_image()) == ""


class TestPickBestLine:
    def test_selects_longest_alpha_line(self):
        text = "||||\nLightning Bolt\n2R"
        assert _pick_best_line(text) == "Lightning Bolt"

    def test_single_line(self):
        assert _pick_best_line("Lightning Bolt") == "Lightning Bolt"

    def test_empty_string(self):
        assert _pick_best_line("") == ""

    def test_noise_lines(self):
        text = "12\n---\nDragonlord Ojutai\n!@#"
        assert _pick_best_line(text) == "Dragonlord Ojutai"
