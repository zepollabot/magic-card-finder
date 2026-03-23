"""Image preprocessor for name-crop regions."""
import cv2
import numpy as np

UPSCALE = 3
UPSCALE_HEIGHT_THRESHOLD = 50
BORDER_PX = 20


class NameCropPreprocessor:
    """Preprocesses a YOLO-cropped name region into a clean binary image
    suitable for Tesseract OCR.

    Since the input is already cropped to the name region by the detector,
    no title-zone cropping or bright-band detection is needed.
    """

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(preprocessed, original_roi)``."""
        roi = image.copy()
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi, roi

        gray = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if len(roi.shape) == 3
            else roi.copy()
        )

        is_dark = float(np.median(gray)) < 120

        h2, w2 = gray.shape[:2]
        if h2 < UPSCALE_HEIGHT_THRESHOLD:
            gray = cv2.resize(
                gray,
                (w2 * UPSCALE, h2 * UPSCALE),
                interpolation=cv2.INTER_CUBIC,
            )

        if gray.size == 0:
            return gray, roi

        gray = cv2.fastNlMeansDenoising(gray, h=10)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        if is_dark:
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=21,
                C=8,
            )
        else:
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=21,
                C=8,
            )

        if np.mean(binary) < 128:
            binary = cv2.bitwise_not(binary)

        padded = cv2.copyMakeBorder(
            binary,
            BORDER_PX,
            BORDER_PX,
            BORDER_PX,
            BORDER_PX,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        return padded, roi
