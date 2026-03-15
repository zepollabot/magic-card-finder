from typing import List, Tuple

import cv2
import numpy as np


class CardDetectionResult:
    def __init__(self, image: np.ndarray, bbox: Tuple[int, int, int, int]):
        self.image = image
        self.bbox = bbox


class CardDetectionService:
    """
    OpenCV-based card detection and cropping.
    Detects card-like rectangles and returns perspective-corrected crops.
    """

    def detect_cards(self, image_bytes: bytes) -> List[CardDetectionResult]:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_results: List[CardDetectionResult] = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area < 10_000:
                continue

            aspect_ratio = max(w, h) / max(1, min(w, h))
            if not (1.3 <= aspect_ratio <= 1.6):
                continue

            pts = approx.reshape(4, 2).astype("float32")
            rect = self._order_points(pts)

            width_a = np.linalg.norm(rect[2] - rect[3])
            width_b = np.linalg.norm(rect[1] - rect[0])
            max_width = int(max(width_a, width_b))

            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[0] - rect[3])
            max_height = int(max(height_a, height_b))

            dst = np.array(
                [
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1],
                ],
                dtype="float32",
            )

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(image, M, (max_width, max_height))

            card_results.append(CardDetectionResult(warp, (x, y, w, h)))

        return card_results

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


