"""Tesseract-based card name recognizer."""
import logging
import os
import re
import uuid
from typing import Union

import cv2
import numpy as np
import pytesseract

from .protocols import Preprocessor

DEFAULT_LANGS = "eng+ita+fra+spa+deu"

logger = logging.getLogger(__name__)

_TRAILING_JUNK_RE = None


def _trailing_junk_re():
    """Lazy-compiled regex that strips trailing mana-cost / OCR noise."""
    global _TRAILING_JUNK_RE
    if _TRAILING_JUNK_RE is None:
        _TRAILING_JUNK_RE = re.compile(
            r"(?:\s+(?:[^\s]{1,2}|[^A-Za-zÀ-ÿ\s]+))+$"
        )
    return _TRAILING_JUNK_RE


def _pick_best_line(text: str) -> str:
    """Pick the line with the most consecutive alpha characters."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return text.strip()

    def _score(line: str) -> int:
        alpha_runs = re.findall(r"[A-Za-zÀ-ÿ]{2,}", line)
        return sum(len(r) for r in alpha_runs)

    return max(lines, key=_score)


class TesseractCardRecognizer:
    """Runs Tesseract on a pre-processed name-crop image.

    The ``Preprocessor`` is injected (DIP) so preprocessing logic
    can be swapped without modifying this class.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        lang: str = DEFAULT_LANGS,
        debug_dir: str = "",
    ) -> None:
        self._preprocessor = preprocessor
        self.lang = lang
        self._debug_dir = debug_dir
        self._tess_config = "--psm 7 --oem 3"

    def recognize(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""

        preprocessed, roi = self._preprocessor.preprocess(image)
        self._save_debug(roi, preprocessed)

        text = self._run_tesseract(preprocessed)
        logger.debug("tesseract: result=%r", text)
        return text

    def _run_tesseract(self, preprocessed: np.ndarray) -> str:
        try:
            raw = pytesseract.image_to_string(
                preprocessed,
                lang=self.lang,
                config=self._tess_config,
            )
        except Exception:
            logger.exception("tesseract: OCR failed")
            return ""

        return self._clean_text(raw)

    def _save_debug(self, original: np.ndarray, preprocessed: np.ndarray) -> None:
        if not self._debug_dir:
            return
        try:
            os.makedirs(self._debug_dir, exist_ok=True)
            suffix = uuid.uuid4().hex[:8]
            cv2.imwrite(
                os.path.join(self._debug_dir, f"title_roi_{suffix}.png"),
                original,
            )
            cv2.imwrite(
                os.path.join(self._debug_dir, f"title_preprocessed_{suffix}.png"),
                preprocessed,
            )
        except Exception:
            pass

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""

        line = _pick_best_line(text)
        cleaned = " ".join(line.split()).strip()
        cleaned = _trailing_junk_re().sub("", cleaned)
        cleaned = re.sub(
            r"^(?:[^\s]{1,2}\s+|[^A-Za-zÀ-ÿ\s]+\s*)+",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^[^A-Za-zÀ-ÿ]+", "", cleaned)
        return cleaned.strip()
