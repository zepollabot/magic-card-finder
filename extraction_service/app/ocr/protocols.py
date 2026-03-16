"""Protocols for text recognition (dependency inversion)."""
from typing import Protocol, Union

import numpy as np


class TitleRegionExtractor(Protocol):
    """Extracts the title text region from a normalized card image."""

    def extract(self, card_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop and preprocess the title area.

        Returns ``(preprocessed, roi)`` where *preprocessed* is the
        clean binary image and *roi* is the raw title crop before
        preprocessing.
        """
        ...


class TextRecognizer(Protocol):
    """Recognizes card name from a single card image."""

    def recognize(self, card_image: Union[bytes, np.ndarray]) -> str:
        """
        Return the card name (title) from a normalized card image.
        Returns empty string when nothing recognized.
        """
        ...
