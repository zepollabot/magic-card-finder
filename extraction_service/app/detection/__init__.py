"""Detection module: raw image -> normalized card crops."""
from .protocols import CardCrop, CardDetector
from .opencv_detector import OpenCVCardDetector
from .classifier import ImageClassifier, ImageType
from .single_card_extractor import SingleCardExtractor
from .multi_card_extractor import MultiCardExtractor

__all__ = [
    "CardCrop",
    "CardDetector",
    "OpenCVCardDetector",
    "ImageClassifier",
    "ImageType",
    "SingleCardExtractor",
    "MultiCardExtractor",
]
