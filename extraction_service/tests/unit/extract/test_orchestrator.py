"""Unit tests for ExtractCardNamesService orchestrator."""
import threading
from typing import List

import numpy as np
import pytest

from app.detection import CardCrop, CardDetector
from app.extract import ExtractCardNamesService
from app.ocr import TextRecognizer


class FakeDetector:
    """Returns configurable crops per image."""

    def __init__(self, crops_per_image: List[int]) -> None:
        self.crops_per_image = crops_per_image
        self._lock = threading.Lock()
        self._call_count = 0

    def detect(self, image_bytes: bytes) -> List[CardCrop]:
        with self._lock:
            idx = self._call_count
            self._call_count += 1
        n = self.crops_per_image[idx] if idx < len(self.crops_per_image) else 0
        return [
            CardCrop(image=np.zeros((936, 672, 3), dtype=np.uint8)) for _ in range(n)
        ]


class FakeRecognizer:
    """Returns names in order; each call returns next name from list."""

    def __init__(self, names: List[str]) -> None:
        self.names = names
        self._lock = threading.Lock()
        self._idx = 0

    def recognize(self, card_image) -> str:
        with self._lock:
            if self._idx < len(self.names):
                name = self.names[self._idx]
                self._idx += 1
                return name
        return ""


def test_extract_empty_images_returns_empty():
    detector = FakeDetector([])
    recognizer = FakeRecognizer([])
    service = ExtractCardNamesService(detector, recognizer, max_workers=2)
    result = service.extract([])
    assert result == []


def test_extract_one_image_two_cards_returns_one_entry_with_two_names():
    detector = FakeDetector([2])  # one image, two cards
    recognizer = FakeRecognizer(["Lightning Bolt", "Island"])
    service = ExtractCardNamesService(detector, recognizer, max_workers=1)
    result = service.extract([b"fake_image_1"])
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] == ["Lightning Bolt", "Island"]


def test_extract_two_images_correct_order_and_names():
    detector = FakeDetector([2, 1])  # first image 2 cards, second 1 card
    recognizer = FakeRecognizer(["A", "B", "C"])
    service = ExtractCardNamesService(detector, recognizer, max_workers=1)
    result = service.extract([b"img1", b"img2"])
    assert len(result) == 2
    assert result[0] == (0, ["A", "B"])
    assert result[1] == (1, ["C"])


def test_extract_image_with_no_cards_returns_empty_names_list():
    detector = FakeDetector([0])  # one image, no cards
    recognizer = FakeRecognizer([])
    service = ExtractCardNamesService(detector, recognizer, max_workers=2)
    result = service.extract([b"fake"])
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] == []


def test_extract_parallelization_preserves_order():
    """Results must be ordered by image_index and by crop order within image."""
    detector = FakeDetector([3, 2, 1])
    recognizer = FakeRecognizer(["1a", "1b", "1c", "2a", "2b", "3a"])
    service = ExtractCardNamesService(detector, recognizer, max_workers=1)
    result = service.extract([b"a", b"b", b"c"])
    assert result[0] == (0, ["1a", "1b", "1c"])
    assert result[1] == (1, ["2a", "2b"])
    assert result[2] == (2, ["3a"])
