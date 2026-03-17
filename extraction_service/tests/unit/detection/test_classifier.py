"""Unit tests for the ImageClassifier."""

import cv2
import numpy as np

from app.detection.classifier import ImageClassifier, ImageType


def _make_card_rect(
    canvas: np.ndarray,
    y0: int, x0: int,
    h: int = 200, w: int = 145,
    color: int = 255,
) -> None:
    canvas[y0 : y0 + h, x0 : x0 + w] = color


class TestClassifier:
    def setup_method(self):
        self.classifier = ImageClassifier()

    def test_single_large_card_classified_as_single(self):
        """A single card occupying ~60% of the image should be SINGLE."""
        img = np.zeros((600, 400, 3), dtype=np.uint8)
        _make_card_rect(img, 30, 30, h=540, w=340)
        result = self.classifier.classify(img)
        assert result == ImageType.SINGLE

    def test_two_disjoint_cards_classified_as_multi(self):
        """Two separate cards should be MULTI."""
        img = np.zeros((500, 600, 3), dtype=np.uint8)
        _make_card_rect(img, 30, 30, 200, 145)
        _make_card_rect(img, 30, 300, 200, 145)
        result = self.classifier.classify(img)
        assert result == ImageType.MULTI

    def test_four_cards_classified_as_multi(self):
        """2x2 grid should be MULTI."""
        img = np.zeros((600, 500, 3), dtype=np.uint8)
        positions = [(30, 30), (30, 260), (280, 30), (280, 260)]
        for y0, x0 in positions:
            _make_card_rect(img, y0, x0, 200, 145)
        result = self.classifier.classify(img)
        assert result == ImageType.MULTI

    def test_empty_image_classified_as_single_fallback(self):
        """An image with no card shapes falls back to SINGLE."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = self.classifier.classify(img)
        assert result == ImageType.SINGLE

    def test_one_small_card_classified_as_multi(self):
        """A single small card (within multi range) should be MULTI."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        _make_card_rect(img, 50, 50, 200, 145)
        result = self.classifier.classify(img)
        assert result == ImageType.MULTI
