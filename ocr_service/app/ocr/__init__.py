"""OCR module: name-crop image -> card name string."""
from .protocols import Preprocessor, TextRecognizer
from .preprocessor import NameCropPreprocessor
from .tesseract_recognizer import TesseractCardRecognizer

__all__ = [
    "Preprocessor",
    "TextRecognizer",
    "NameCropPreprocessor",
    "TesseractCardRecognizer",
]
