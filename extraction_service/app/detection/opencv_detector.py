"""OpenCV-based card detection with classify-then-extract architecture.

Classifies the input image as single-card or multi-card, then delegates
to the appropriate extractor.  All crops are normalized to 672x936.
"""

import logging
from typing import List

import cv2
import numpy as np

from .protocols import CardCrop, CardDetector
from .card_normalizer import save_debug_crops
from .classifier import ImageClassifier, ImageType
from .single_card_extractor import SingleCardExtractor
from .multi_card_extractor import MultiCardExtractor

logger = logging.getLogger(__name__)


class OpenCVCardDetector:
    """
    Detects card-like rectangles via a two-stage pipeline:

    1. **Classify** the image as single-card or multi-card.
    2. **Extract** using the appropriate strategy.

    The public interface (``detect(image_bytes) -> List[CardCrop]``)
    is unchanged from the original implementation.
    """

    def __init__(self) -> None:
        self._classifier = ImageClassifier()
        self._single_extractor = SingleCardExtractor()
        self._multi_extractor = MultiCardExtractor()

    def detect(self, image_bytes: bytes) -> List[CardCrop]:
        image = self._decode(image_bytes)
        if image is None:
            return []

        image_type = self._classifier.classify(image)
        logger.debug("detect: image classified as %s", image_type.value)

        if image_type == ImageType.SINGLE:
            crops = self._single_extractor.extract(image)
        else:
            crops = self._multi_extractor.extract(image)

        save_debug_crops(crops)
        return crops

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray | None:
        if not image_bytes:
            return None
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
