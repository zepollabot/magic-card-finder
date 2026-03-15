"""Protocols for card detection (dependency inversion)."""
from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


@dataclass
class CardCrop:
    """A single perspective-corrected, normalized card image."""

    image: np.ndarray  # BGR, shape (CARD_HEIGHT, CARD_WIDTH, 3)


class CardDetector(Protocol):
    """Detects card-like quadrilaterals and returns normalized crops."""

    def detect(self, image_bytes: bytes) -> List[CardCrop]:
        """
        Detect cards in raw image bytes and return normalized crops.
        Returns empty list if decode fails or no cards found.
        """
        ...
