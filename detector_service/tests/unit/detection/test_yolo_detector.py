"""Unit tests for YOLOCardDetector."""
import numpy as np
import pytest

from app.detection.yolo_detector import YOLOCardDetector, CLASS_NAMES


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResults:
    def __init__(self, boxes: list[FakeBox] | None = None):
        self.boxes = boxes


class FakeYOLOModel:
    def __init__(self, results: list[FakeResults] | None = None):
        self._results = results or []

    def __call__(self, image, conf=0.4):
        return self._results


def _make_image(h=200, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestYOLOCardDetector:
    def test_detect_maps_class_ids_to_names(self):
        boxes = [FakeBox(cls_id=i, conf=0.9, xyxy=[10, 10, 50, 50]) for i in range(5)]
        model = FakeYOLOModel([FakeResults(boxes)])
        detector = YOLOCardDetector(model)

        detections = detector.detect(_make_image())

        for det, expected_name in zip(detections, CLASS_NAMES.values()):
            assert det.cls_name == expected_name

    def test_detect_filters_by_class(self):
        boxes = [
            FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 50, 50]),
            FakeBox(cls_id=1, conf=0.9, xyxy=[60, 10, 100, 50]),
            FakeBox(cls_id=3, conf=0.9, xyxy=[110, 10, 150, 50]),
        ]
        model = FakeYOLOModel([FakeResults(boxes)])
        detector = YOLOCardDetector(model)

        detections = detector.detect(_make_image())
        name_dets = [d for d in detections if d.cls_name == "name"]

        assert len(name_dets) == 1
        assert name_dets[0].bbox == (60, 10, 100, 50)

    def test_detect_applies_confidence_threshold(self):
        boxes = [
            FakeBox(cls_id=1, conf=0.2, xyxy=[10, 10, 50, 50]),
            FakeBox(cls_id=1, conf=0.8, xyxy=[60, 10, 100, 50]),
        ]
        model = FakeYOLOModel([FakeResults(boxes)])
        detector = YOLOCardDetector(model)

        detections = detector.detect(_make_image(), conf=0.5)

        assert len(detections) == 2
        assert detections[0].confidence == pytest.approx(0.2)
        assert detections[1].confidence == pytest.approx(0.8)

    def test_detect_crops_correct_region(self):
        image = np.arange(200 * 300 * 3, dtype=np.uint8).reshape(200, 300, 3)
        boxes = [FakeBox(cls_id=1, conf=0.9, xyxy=[10, 20, 50, 60])]
        model = FakeYOLOModel([FakeResults(boxes)])
        detector = YOLOCardDetector(model)

        detections = detector.detect(image)
        expected_crop = image[20:60, 10:50]

        np.testing.assert_array_equal(detections[0].crop, expected_crop)

    def test_detect_empty_results(self):
        model = FakeYOLOModel([FakeResults(None)])
        detector = YOLOCardDetector(model)

        assert detector.detect(_make_image()) == []

    def test_detect_none_image(self):
        model = FakeYOLOModel([])
        detector = YOLOCardDetector(model)

        assert detector.detect(None) == []

    def test_detect_empty_image(self):
        model = FakeYOLOModel([])
        detector = YOLOCardDetector(model)

        assert detector.detect(np.array([])) == []

    def test_detect_clamps_bbox_to_image_bounds(self):
        image = _make_image(100, 100)
        boxes = [FakeBox(cls_id=1, conf=0.9, xyxy=[-5, -5, 110, 110])]
        model = FakeYOLOModel([FakeResults(boxes)])
        detector = YOLOCardDetector(model)

        detections = detector.detect(image)

        assert detections[0].bbox == (0, 0, 100, 100)

    def test_detect_skips_degenerate_bbox(self):
        image = _make_image(100, 100)
        boxes = [FakeBox(cls_id=1, conf=0.9, xyxy=[50, 50, 50, 50])]
        model = FakeYOLOModel([FakeResults(boxes)])
        detector = YOLOCardDetector(model)

        assert detector.detect(image) == []
