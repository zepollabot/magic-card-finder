"""Tesseract-based card title OCR."""

import logging
import os
import uuid
from typing import Union

import cv2
import numpy as np
import pytesseract

from .protocols import TitleRegionExtractor

DEFAULT_LANGS = "eng+ita+fra+spa+deu"

logger = logging.getLogger(__name__)

_TRAILING_JUNK_RE = None


def _trailing_junk_re():
    """Lazy-compiled regex that strips trailing mana-cost / OCR noise.

    Repeatedly removes trailing tokens that are:
      - purely non-alphabetic (numbers, symbols, punctuation), or
      - 1-2 characters long (likely OCR artefacts from mana symbols).
    """
    global _TRAILING_JUNK_RE
    if _TRAILING_JUNK_RE is None:
        import re

        _TRAILING_JUNK_RE = re.compile(
            r"(?:\s+(?:[^\s]{1,2}|[^A-Za-zÀ-ÿ\s]+))+$"
        )
    return _TRAILING_JUNK_RE


def _pick_best_line(text: str) -> str:
    """When Tesseract reads noise lines from dark border rows, the actual
    card title is typically the line with the most consecutive letters.
    Pick the line that looks most like a real card name.
    """
    import re

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return text.strip()

    def _score(line: str) -> int:
        alpha_runs = re.findall(r"[A-Za-zÀ-ÿ]{2,}", line)
        return sum(len(r) for r in alpha_runs)

    best = max(lines, key=_score)
    return best


class TesseractCardRecognizer:
    """
    Runs Tesseract on a pre-extracted title region.

    Preprocessing is delegated to a ``TitleRegionExtractor`` (SRP / DIP),
    so this class only owns OCR invocation and result cleaning.
    """

    def __init__(
        self,
        title_extractor: TitleRegionExtractor,
        lang: str = DEFAULT_LANGS,
    ) -> None:
        self.lang = lang
        self._title_extractor = title_extractor
        self._debug_tesseract = bool(os.getenv("OCR_DEBUG_TESSERACT", "").strip())

        self._tess_config = "--psm 6 --oem 3"

    def recognize(self, card_image: Union[bytes, np.ndarray]) -> str:
        if isinstance(card_image, bytes):
            np_arr = np.frombuffer(card_image, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            img = card_image
        if img is None:
            return ""

        preprocessed, roi = self._title_extractor.extract(img)
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

        cleaned = self._clean_text(raw)
        if self._debug_tesseract:
            logger.debug(
                "tesseract cfg=%r raw_len=%d cleaned=%r",
                self._tess_config,
                len(raw or ""),
                cleaned[:80],
            )
        return cleaned

    def _save_debug(self, original: np.ndarray, preprocessed: np.ndarray) -> None:
        debug_dir = os.getenv("OCR_DEBUG_DIR", "").strip()
        if not debug_dir:
            return
        try:
            os.makedirs(debug_dir, exist_ok=True)
            suffix = uuid.uuid4().hex[:8]
            cv2.imwrite(os.path.join(debug_dir, f"title_roi_{suffix}.png"), original)
            cv2.imwrite(
                os.path.join(debug_dir, f"title_preprocessed_{suffix}.png"),
                preprocessed,
            )
        except Exception:
            pass

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        import re

        line = _pick_best_line(text)
        cleaned = " ".join(line.split()).strip()
        cleaned = _trailing_junk_re().sub("", cleaned)
        # Strip leading 1-2 char tokens or purely non-alpha tokens.
        cleaned = re.sub(
            r"^(?:[^\s]{1,2}\s+|[^A-Za-zÀ-ÿ\s]+\s*)+", "", cleaned,
        )
        # Strip leading non-alpha characters glued to the first word.
        cleaned = re.sub(r"^[^A-Za-zÀ-ÿ]+", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned
