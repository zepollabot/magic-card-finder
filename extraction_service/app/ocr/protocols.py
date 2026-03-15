"""Protocols for text recognition (dependency inversion)."""
from typing import Protocol, Union

import numpy as np


class TextRecognizer(Protocol):
    """Recognizes card name from a single card image."""

    def recognize(self, card_image: Union[bytes, np.ndarray]) -> str:
        """
        Return the card name (title) from a normalized card image.
        Returns empty string when nothing recognized.
        """
        ...
