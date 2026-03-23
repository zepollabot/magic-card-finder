"""Unit tests for detector debug savers."""
import os
from unittest.mock import patch

import numpy as np

from app.detection.debug import FileDetectorDebugger, NullDebugger
from app.detection.protocols import Detection


def _make_detection(cls_name="name", x1=10, y1=20, x2=50, y2=60, conf=0.9):
    crop = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    return Detection(
        cls_name=cls_name,
        confidence=conf,
        bbox=(x1, y1, x2, y2),
        crop=crop,
    )


def _make_image(h=200, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestFileDetectorDebugger:
    def test_saves_original_image(self, tmp_path):
        debugger = FileDetectorDebugger(str(tmp_path))
        detections = [_make_detection()]

        debugger.save("abc", 0, _make_image(), detections)

        assert (tmp_path / "abc_0_original.jpg").exists()

    def test_saves_annotated_image(self, tmp_path):
        debugger = FileDetectorDebugger(str(tmp_path))
        detections = [_make_detection()]

        debugger.save("abc", 0, _make_image(), detections)

        assert (tmp_path / "abc_0_detections.jpg").exists()

    def test_saves_name_crops(self, tmp_path):
        debugger = FileDetectorDebugger(str(tmp_path))
        detections = [
            _make_detection(cls_name="name"),
            _make_detection(cls_name="name", x1=60, x2=100),
        ]

        debugger.save("abc", 0, _make_image(), detections)

        assert (tmp_path / "abc_0_name_0.jpg").exists()
        assert (tmp_path / "abc_0_name_1.jpg").exists()

    def test_does_not_save_non_name_crops(self, tmp_path):
        debugger = FileDetectorDebugger(str(tmp_path))
        detections = [_make_detection(cls_name="card")]

        debugger.save("abc", 0, _make_image(), detections)

        assert not (tmp_path / "abc_0_name_0.jpg").exists()
        assert (tmp_path / "abc_0_original.jpg").exists()
        assert (tmp_path / "abc_0_detections.jpg").exists()

    def test_creates_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "subdir" / "nested"
        debugger = FileDetectorDebugger(str(new_dir))

        debugger.save("abc", 0, _make_image(), [_make_detection()])

        assert new_dir.exists()

    def test_handles_empty_detections(self, tmp_path):
        debugger = FileDetectorDebugger(str(tmp_path))

        debugger.save("abc", 0, _make_image(), [])

        assert (tmp_path / "abc_0_original.jpg").exists()
        assert (tmp_path / "abc_0_detections.jpg").exists()


class TestNullDebugger:
    def test_does_nothing(self, tmp_path):
        debugger = NullDebugger()
        debugger.save("abc", 0, _make_image(), [_make_detection()])

        assert len(list(tmp_path.iterdir())) == 0
