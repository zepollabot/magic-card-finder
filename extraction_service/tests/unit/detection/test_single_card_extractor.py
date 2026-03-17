"""Unit tests for SingleCardExtractor."""

import cv2
import numpy as np

from app.detection.card_normalizer import CARD_HEIGHT, CARD_WIDTH
from app.detection.protocols import CardCrop
from app.detection.single_card_extractor import SingleCardExtractor


def _encode(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _make_card_rect(
    canvas: np.ndarray,
    y0: int, x0: int,
    h: int, w: int,
    color: int = 255,
) -> None:
    canvas[y0 : y0 + h, x0 : x0 + w] = color


class TestSingleCardExtractor:
    def setup_method(self):
        self.extractor = SingleCardExtractor()

    def test_extracts_large_card_from_image(self):
        """A card covering ~60% of the image should be detected."""
        img = np.zeros((600, 400, 3), dtype=np.uint8)
        _make_card_rect(img, 30, 30, h=540, w=340)
        crops = self.extractor.extract(img)
        assert len(crops) == 1
        assert isinstance(crops[0], CardCrop)
        assert crops[0].image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)

    def test_extracts_card_filling_most_of_frame(self):
        """A card covering ~90% should still be detected."""
        img = np.zeros((500, 360, 3), dtype=np.uint8)
        _make_card_rect(img, 10, 10, h=480, w=340)
        crops = self.extractor.extract(img)
        assert len(crops) == 1
        assert crops[0].image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)

    def test_fallback_returns_one_crop_for_featureless_image(self):
        """When no contour is found, the whole-image fallback should kick in."""
        img = np.full((936, 672, 3), 100, dtype=np.uint8)
        crops = self.extractor.extract(img)
        assert len(crops) == 1
        assert crops[0].image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)

    def test_crop_is_portrait(self):
        """All output crops must be portrait."""
        img = np.zeros((600, 400, 3), dtype=np.uint8)
        _make_card_rect(img, 30, 30, h=540, w=340)
        crops = self.extractor.extract(img)
        assert len(crops) >= 1
        for crop in crops:
            assert crop.image.shape[0] > crop.image.shape[1]

    def test_extract_returns_list(self):
        """Return type is always a list."""
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        _make_card_rect(img, 20, 20, h=360, w=260)
        result = self.extractor.extract(img)
        assert isinstance(result, list)

    def test_prefers_card_over_background_contour(self):
        """When a card sits on a large plain background, the card contour
        should be picked, not the full-image background contour."""
        img = np.full((936, 672, 3), 210, dtype=np.uint8)
        # Card-shaped rectangle centered in image (~38% area)
        card_y, card_x, card_w, card_h = 135, 159, 424, 567
        img[card_y:card_y + card_h, card_x:card_x + card_w] = 40
        # Lighter card interior
        m = 8
        img[card_y + m:card_y + card_h - m, card_x + m:card_x + card_w - m] = 60

        crops = self.extractor.extract(img)
        assert len(crops) == 1
        # The top of the crop should be dark (card content), not bright
        # (background). If the background were picked, brightness > 180.
        top_strip = crops[0].image[:50, :, :]
        assert np.mean(top_strip) < 150
