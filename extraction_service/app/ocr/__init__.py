"""OCR module: single card image -> card name string."""
from .protocols import TextRecognizer
from .tesseract_recognizer import TesseractCardRecognizer

__all__ = ["TextRecognizer", "TesseractCardRecognizer"]
