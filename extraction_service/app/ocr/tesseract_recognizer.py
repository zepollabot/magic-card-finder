"""Tesseract-based card title OCR with preprocessing."""
from typing import Union

import logging
import os
import uuid

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


logger = logging.getLogger(__name__)


class TesseractCardRecognizer:
    """
    Preprocesses card image, crops the title zone, and runs Tesseract
    with multi-language support (PSM 7, single text line).
    """

    def __init__(self, lang: str = DEFAULT_LANGS) -> None:
        self.lang = lang
        # When true, log raw / cleaned outputs for each Tesseract config.
        self._debug_tesseract = bool(os.getenv("OCR_DEBUG_TESSERACT", "").strip())
        base_whitelist = os.getenv(
            "TESSERACT_TITLE_WHITELIST",
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            " ,.'-",
        )
        self._tess_configs = [
            # Original, un-whitelisted config (kept first because it worked well
            # on many titles, including clean ones like the debug samples).
            "--psm 7 --oem 3",
            # Stricter configs that can help on noisier scans.
            f"--psm 7 --oem 3 -c tessedit_char_whitelist={base_whitelist}",
            f"--psm 6 --oem 3 -c tessedit_char_whitelist={base_whitelist}",
        ]

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

        # Optional debug: dump ROI and preprocessed image to disk when
        # OCR_DEBUG_DIR is set (e.g. "/tmp/ocr-debug").
        debug_dir = os.getenv("OCR_DEBUG_DIR", "").strip()
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                suffix = uuid.uuid4().hex[:8]
                roi_path = os.path.join(debug_dir, f"title_roi_{suffix}.png")
                pre_path = os.path.join(debug_dir, f"title_preprocessed_{suffix}.png")
                cv2.imwrite(roi_path, roi)
                cv2.imwrite(pre_path, preprocessed)
            except Exception:
                # Debugging helper – failures here must not break OCR.
                pass

        cleaned = self._run_tesseract(preprocessed)
        logger.debug(
            "tesseract recognizer: cleaned=%r (empty=%s)",
            cleaned,
            not bool(cleaned),
        )
        return cleaned

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

        # Heuristic: if the image is already almost pure black/white (like our
        # preprocessed debug strips), avoid heavy filtering which can break
        # thin glyphs. We detect this by measuring how many pixels are very
        # dark or very light.
        total = gray.size
        if total == 0:
            return gray
        dark = (gray <= 30).sum()
        light = (gray >= 225).sum()
        bw_ratio = (dark + light) / float(total)

        if bw_ratio > 0.9:
            # Minimal preprocessing: keep existing binary structure, just close
            # small gaps and add a border.
            binary = gray
        else:
            denoised = cv2.medianBlur(gray, 3)
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
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

    def _run_tesseract(self, preprocessed: np.ndarray) -> str:
        """
        Run Tesseract with a couple of configs and pick the best result.
        Heuristic: prefer the longest non-empty cleaned string.
        """
        best: str = ""
        for cfg in self._tess_configs:
            try:
                raw = pytesseract.image_to_string(
                    preprocessed,
                    lang=self.lang,
                    config=cfg,
                )
            except Exception:
                continue
            cleaned = self._clean_text(raw)
            if self._debug_tesseract:
                # Log a short preview to avoid huge lines.
                preview = cleaned[:80]
                logger.debug(
                    "tesseract cfg=%r raw_len=%d cleaned_preview=%r",
                    cfg,
                    len(raw or ""),
                    preview,
                )
            if cleaned and len(cleaned) > len(best):
                best = cleaned
        return best

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        one_line = " ".join(text.split())
        return one_line.strip()

