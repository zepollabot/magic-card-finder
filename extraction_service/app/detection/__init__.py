"""Detection module: raw image -> normalized card crops."""
from .protocols import CardCrop, CardDetector
from .opencv_detector import OpenCVCardDetector

__all__ = ["CardCrop", "CardDetector", "OpenCVCardDetector"]
