"""YOLO-based card detector implementation."""
import logging
from typing import List

import numpy as np

from .protocols import CardDetector, Detection

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "card",
    1: "name",
    2: "power-toughness",
    3: "text",
    4: "type",
}


class YOLOCardDetector:
    """Runs YOLO inference and returns per-box Detection objects.

    The YOLO model is injected via the constructor (DIP) so the class
    is testable without loading real weights.
    """

    def __init__(self, model) -> None:
        self._model = model

    def detect(self, image: np.ndarray, conf: float = 0.4) -> List[Detection]:
        if image is None or image.size == 0:
            return []

        results = self._model(image, conf=conf)
        if not results:
            return []

        detections: List[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return []

        h, w = image.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2].copy()
            detections.append(
                Detection(
                    cls_name=cls_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    crop=crop,
                )
            )

        logger.debug(
            "yolo: detected %d object(s) (conf>=%.2f)",
            len(detections),
            conf,
        )
        return detections
