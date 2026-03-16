"""Extracts and preprocesses the title text region from a card image."""

import cv2
import numpy as np

# Generous vertical zone that captures the full title bar on all card layouts.
# The dark-frame noise at the top is removed during preprocessing, so we
# can afford to start very early.  The bottom extends into the illustration
# area to ensure the text is never cut off.
TITLE_ZONE_TOP = 0.01
TITLE_ZONE_BOTTOM = 0.12
TITLE_LEFT_INSET = 0.06
TITLE_RIGHT_INSET = 0.22

UPSCALE = 3
BORDER_PX = 20


class CardTitleRegionExtractor:
    """
    Crops the title bar of a normalized MTG card image and
    produces a clean binary image suitable for OCR.

    The right ~22 % of the title bar is discarded to remove
    mana-cost symbols which confuse character recognition.
    """

    def extract(self, card_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(preprocessed, roi)`` where *roi* is the raw title crop."""
        roi = self._crop_title_zone(card_image)
        return self._preprocess(roi), roi

    def _crop_title_zone(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        y_start = max(0, int(h * TITLE_ZONE_TOP))
        y_end = max(y_start + 1, int(h * TITLE_ZONE_BOTTOM))
        x_start = max(0, int(w * TITLE_LEFT_INSET))
        x_end = max(x_start + 1, w - int(w * TITLE_RIGHT_INSET))
        return img[y_start:y_end, x_start:x_end]

    @staticmethod
    def _find_bright_band(gray: np.ndarray) -> tuple[int, int]:
        """Find the bright title-bar band in the grayscale ROI.

        The title bar is a horizontal bright stripe between darker regions
        (frame on top, illustration on bottom).  We find the longest
        contiguous run of rows whose mean brightness is above a threshold.
        """
        h, _ = gray.shape[:2]
        if h < 6:
            return 0, h

        row_means = gray.mean(axis=1).astype(np.float64)
        lo, hi = float(row_means.min()), float(row_means.max())
        spread = hi - lo
        if spread < 15:
            return 0, h

        thr = lo + 0.45 * spread

        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for i in range(h):
            if row_means[i] >= thr:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_len = 0
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        if best_len < h * 0.15:
            return 0, h

        band_start = best_start
        band_end = best_start + best_len
        margin = min(3, best_len // 3)
        band_start = min(band_start + margin, band_end - 1)
        band_end = max(band_end - margin, band_start + 1)
        return band_start, band_end

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi

        gray = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if len(roi.shape) == 3
            else roi
        )

        is_dark = float(np.median(gray)) < 120

        if not is_dark:
            y0, y1 = self._find_bright_band(gray)
            gray = gray[y0:y1, :]

        h2, w2 = gray.shape[:2]
        if h2 < 2 or w2 < 2:
            return gray

        scaled = cv2.resize(
            gray, (w2 * UPSCALE, h2 * UPSCALE), interpolation=cv2.INTER_CUBIC,
        )
        if scaled.size == 0:
            return scaled

        scaled = cv2.fastNlMeansDenoising(scaled, h=10)

        blurred = cv2.GaussianBlur(scaled, (3, 3), 0)

        if is_dark:
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                blockSize=21, C=8,
            )
        else:
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                blockSize=21, C=8,
            )

        if np.mean(binary) < 128:
            binary = cv2.bitwise_not(binary)

        return cv2.copyMakeBorder(
            binary,
            BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
            cv2.BORDER_CONSTANT,
            value=255,
        )
