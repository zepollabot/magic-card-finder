"""Tesseract-based card title OCR with preprocessing."""
from typing import Union

import cv2
import numpy as np
import pytesseract

DEFAULT_LANGS = "eng+ita+fra+spa+deu"

TITLE_ZONE_TOP = 0.04
TITLE_ZONE_BOTTOM = 0.15
HORIZONTAL_INSET = 0.05

CARD_HEIGHT = 936
CARD_WIDTH = 672

UPSCALE = 3
BORDER_PX = 20


class TesseractCardRecognizer:
    """
    Preprocesses card image, crops the title zone, and runs Tesseract
    with multi-language support (PSM 7, single text line).
    """

    def __init__(self, lang: str = DEFAULT_LANGS) -> None:
        self.lang = lang
        self._tess_config = "--psm 7 --oem 3"

    def recognize(self, card_image: Union[bytes, np.ndarray]) -> str:
        if isinstance(card_image, bytes):
            np_arr = np.frombuffer(card_image, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            img = card_image
        if img is None:
            return ""
        roi = self._crop_title_zone(img)
        preprocessed = self._preprocess(roi)
        raw = pytesseract.image_to_string(
            preprocessed, lang=self.lang, config=self._tess_config
        )
        return self._clean_text(raw)

    def _crop_title_zone(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        y_start = max(0, int(h * TITLE_ZONE_TOP))
        y_end = max(y_start + 1, int(h * TITLE_ZONE_BOTTOM))
        x_start = max(0, int(w * HORIZONTAL_INSET))
        x_end = max(x_start + 1, w - int(w * HORIZONTAL_INSET))
        return img[y_start:y_end, x_start:x_end]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        scaled = cv2.resize(
            roi, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        padded = cv2.copyMakeBorder(
            binary,
            BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        return padded

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        one_line = " ".join(text.split())
        return one_line.strip()

