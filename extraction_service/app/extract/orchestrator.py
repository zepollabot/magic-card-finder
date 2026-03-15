"""Orchestrates detection and OCR with parallelization."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from app.detection import CardCrop, CardDetector
from app.ocr import TextRecognizer

logger = logging.getLogger(__name__)


def _default_workers() -> int:
    try:
        n = os.cpu_count() or 4
        return min(32, n + 4)
    except Exception:
        return 4


class ExtractCardNamesService:
    """
    Use case: list of raw images -> list of (image_index, card_names).
    Runs detection in parallel across images, then OCR in parallel across all crops.
    """

    def __init__(
        self,
        detector: CardDetector,
        recognizer: TextRecognizer,
        max_workers: int | None = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self._max_workers = max_workers or _default_workers()

    def extract(self, images: List[bytes]) -> List[Tuple[int, List[str]]]:
        """
        For each image, detect cards then recognize names. Results are ordered
        by image_index; card_names within each image follow detection order.
        """
        if not images:
            return []

        logger.debug("extract: starting detection for %d image(s)", len(images))
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            # Parallel detection: one future per image
            detection_futures = {
                pool.submit(self.detector.detect, img): idx
                for idx, img in enumerate(images)
            }
            # (image_index, crop_index, crop)
            indexed_crops: List[Tuple[int, int, CardCrop]] = []
            for fut in as_completed(detection_futures):
                image_index = detection_futures[fut]
                try:
                    crops = fut.result()
                except Exception:
                    crops = []
                for crop_index, crop in enumerate(crops):
                    indexed_crops.append((image_index, crop_index, crop))

            indexed_crops.sort(key=lambda x: (x[0], x[1]))
            total_crops = len(indexed_crops)
            logger.debug("extract: detection done — %d crop(s) from %d image(s)", total_crops, len(images))

            # Parallel OCR: one future per crop
            ocr_futures = {
                pool.submit(self._recognize_crop, crop): (img_idx, crop_idx)
                for img_idx, crop_idx, crop in indexed_crops
            }
            # (image_index, crop_index, name)
            name_results: List[Tuple[int, int, str]] = []
            for fut in as_completed(ocr_futures):
                img_idx, crop_idx = ocr_futures[fut]
                try:
                    name = fut.result()
                except Exception:
                    name = ""
                name_results.append((img_idx, crop_idx, name))

        # Group by image_index, sort by crop_index, build list of names
        by_image: dict[int, List[Tuple[int, str]]] = {}
        for img_idx, crop_idx, name in name_results:
            if img_idx not in by_image:
                by_image[img_idx] = []
            by_image[img_idx].append((crop_idx, name))
        for img_idx in by_image:
            by_image[img_idx].sort(key=lambda x: x[0])
        # One entry per input image, in order; empty images get empty list
        result: List[Tuple[int, List[str]]] = []
        for idx in range(len(images)):
            names = [n for _, n in sorted(by_image.get(idx, []), key=lambda x: x[0])]
            result.append((idx, names))
        total_names = sum(len(n) for _, n in result)
        logger.debug(
            "extract: OCR done — %d name(s) total (per image: %s)",
            total_names,
            [len(n) for _, n in result],
        )
        return result

    def _recognize_crop(self, crop: CardCrop) -> str:
        return self.recognizer.recognize(crop.image)
