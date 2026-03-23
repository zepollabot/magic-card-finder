"""Detection module: YOLO-based card name region detection."""
from .protocols import Detection, CardDetector, DebugSaver
from .yolo_detector import YOLOCardDetector
from .debug import FileDetectorDebugger, NullDebugger

__all__ = [
    "Detection",
    "CardDetector",
    "DebugSaver",
    "YOLOCardDetector",
    "FileDetectorDebugger",
    "NullDebugger",
]
