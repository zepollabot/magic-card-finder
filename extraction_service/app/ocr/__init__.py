"""OCR module: single card image -> card name string."""
from .protocols import TextRecognizer, TitleRegionExtractor
from .tesseract_recognizer import TesseractCardRecognizer
from .title_region import CardTitleRegionExtractor

__all__ = [
    "TextRecognizer",
    "TitleRegionExtractor",
    "TesseractCardRecognizer",
    "CardTitleRegionExtractor",
]
