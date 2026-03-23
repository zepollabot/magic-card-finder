"""Unit tests for NameCropPreprocessor."""
import numpy as np
import cv2

from app.ocr.preprocessor import NameCropPreprocessor, BORDER_PX, UPSCALE, UPSCALE_HEIGHT_THRESHOLD


def _make_light_image(h=40, w=200):
    """Simulate a light title bar region (white background, dark text)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.putText(img, "Card Name", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return img


def _make_dark_image(h=40, w=200):
    """Simulate a dark title bar region (dark background, light text)."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(img, "Card Name", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    return img


class TestNameCropPreprocessor:
    def setup_method(self):
        self.preprocessor = NameCropPreprocessor()

    def test_output_is_binary(self):
        preprocessed, _ = self.preprocessor.preprocess(_make_light_image())
        unique_values = set(np.unique(preprocessed))
        assert unique_values <= {0, 255}

    def test_dark_image_handling(self):
        preprocessed, _ = self.preprocessor.preprocess(_make_dark_image())
        assert preprocessed.size > 0
        assert np.mean(preprocessed) >= 128

    def test_light_image_handling(self):
        preprocessed, _ = self.preprocessor.preprocess(_make_light_image())
        assert preprocessed.size > 0
        assert np.mean(preprocessed) >= 128

    def test_small_image_upscaled(self):
        small = _make_light_image(h=30, w=100)
        preprocessed, _ = self.preprocessor.preprocess(small)
        expected_h = 30 * UPSCALE + 2 * BORDER_PX
        expected_w = 100 * UPSCALE + 2 * BORDER_PX
        assert preprocessed.shape[0] == expected_h
        assert preprocessed.shape[1] == expected_w

    def test_large_image_not_upscaled(self):
        large = _make_light_image(h=80, w=400)
        preprocessed, _ = self.preprocessor.preprocess(large)
        expected_h = 80 + 2 * BORDER_PX
        expected_w = 400 + 2 * BORDER_PX
        assert preprocessed.shape[0] == expected_h
        assert preprocessed.shape[1] == expected_w

    def test_border_padding_added(self):
        img = _make_light_image(h=80, w=200)
        preprocessed, _ = self.preprocessor.preprocess(img)
        assert preprocessed.shape[0] == 80 + 2 * BORDER_PX
        assert preprocessed.shape[1] == 200 + 2 * BORDER_PX

    def test_majority_white_background(self):
        preprocessed, _ = self.preprocessor.preprocess(_make_light_image())
        assert np.mean(preprocessed) >= 128

    def test_returns_original_roi_copy(self):
        img = _make_light_image()
        _, roi = self.preprocessor.preprocess(img)
        np.testing.assert_array_equal(roi, img)
        assert roi is not img

    def test_tiny_image_passthrough(self):
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        preprocessed, roi = self.preprocessor.preprocess(tiny)
        assert preprocessed.size > 0
