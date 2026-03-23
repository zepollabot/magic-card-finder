"""Protocols for card detection (dependency inversion)."""
from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


@dataclass
class Detection:
    """Single bounding-box detection from YOLO."""

    cls_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    crop: np.ndarray  # BGR cropped region


class CardDetector(Protocol):
    """Detects card regions in an image and returns detection results."""

    def detect(self, image: np.ndarray, conf: float = 0.4) -> List[Detection]: ...


class DebugSaver(Protocol):
    """Saves debug artifacts (original image, annotated detections, crops)."""

    def save(
        self,
        batch_id: str,
        image_index: int,
        original: np.ndarray,
        detections: List[Detection],
    ) -> None: ...
