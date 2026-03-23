"""Debug file savers for detector output."""
import logging
import os
from typing import List

import cv2
import numpy as np

from .protocols import DebugSaver, Detection

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "card": (0, 255, 0),
    "name": (255, 0, 0),
    "power-toughness": (0, 0, 255),
    "text": (255, 255, 0),
    "type": (0, 255, 255),
}

DEFAULT_COLOR = (128, 128, 128)


class FileDetectorDebugger:
    """Writes debug images to disk when DETECTOR_DEBUG_DIR is configured."""

    def __init__(self, debug_dir: str) -> None:
        self._debug_dir = debug_dir

    def save(
        self,
        batch_id: str,
        image_index: int,
        original: np.ndarray,
        detections: List[Detection],
    ) -> None:
        try:
            os.makedirs(self._debug_dir, exist_ok=True)

            prefix = f"{batch_id}_{image_index}"

            cv2.imwrite(
                os.path.join(self._debug_dir, f"{prefix}_original.jpg"),
                original,
            )

            annotated = original.copy()
            name_crop_idx = 0
            for det in detections:
                color = CLASS_COLORS.get(det.cls_name, DEFAULT_COLOR)
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det.cls_name} {det.confidence:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

                if det.cls_name == "name":
                    cv2.imwrite(
                        os.path.join(
                            self._debug_dir,
                            f"{prefix}_name_{name_crop_idx}.jpg",
                        ),
                        det.crop,
                    )
                    name_crop_idx += 1

            cv2.imwrite(
                os.path.join(self._debug_dir, f"{prefix}_detections.jpg"),
                annotated,
            )
            logger.debug(
                "debug: saved %d artifacts to %s",
                2 + name_crop_idx,
                self._debug_dir,
            )
        except Exception:
            logger.exception("debug: failed to save artifacts")


class NullDebugger:
    """No-op debugger (Null Object pattern) used when debug is disabled."""

    def save(
        self,
        batch_id: str,
        image_index: int,
        original: np.ndarray,
        detections: List[Detection],
    ) -> None:
        pass
