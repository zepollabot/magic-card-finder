"""Protocols for OCR (dependency inversion)."""
from typing import Protocol

import numpy as np


class Preprocessor(Protocol):
    """Preprocesses a name-crop image for OCR."""

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(preprocessed_for_ocr, original_roi)``."""
        ...


class TextRecognizer(Protocol):
    """Recognizes card name text from a name-crop image."""

    def recognize(self, image: np.ndarray) -> str:
        """Return recognized text, or empty string on failure."""
        ...
