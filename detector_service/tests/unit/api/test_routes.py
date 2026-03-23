"""Unit tests for detector API routes."""
import base64
from typing import List
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.detection.protocols import Detection
from app.main import app


def _make_detection(cls_name="name", conf=0.9, bbox=(10, 20, 50, 60)):
    crop = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), dtype=np.uint8)
    return Detection(cls_name=cls_name, confidence=conf, bbox=bbox, crop=crop)


class FakeDetector:
    def __init__(self, detections: List[Detection] | None = None):
        self._detections = detections or []

    def detect(self, image, conf=0.4):
        return self._detections


def _encode_image(h=100, w=100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


class TestDetectRoute:
    def setup_method(self):
        app.state.detector = FakeDetector([_make_detection()])
        app.state.debugger = MagicMock()
        app.state.confidence = 0.4
        app.state.model_name = "test_model"
        self.client = TestClient(app)

    def test_detect_success(self):
        resp = self.client.post(
            "/v1/detect", json={"images": [_encode_image()]}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert len(data["results"][0]["detections"]) == 1
        assert data["results"][0]["detections"][0]["cls"] == "name"
        assert "crop_b64" in data["results"][0]["detections"][0]

    def test_detect_invalid_base64(self):
        resp = self.client.post(
            "/v1/detect", json={"images": ["not-valid-base64!!!"]}
        )
        assert resp.status_code == 400

    def test_detect_empty_images(self):
        resp = self.client.post("/v1/detect", json={"images": []})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_detect_service_unavailable(self):
        app.state.detector = None
        resp = self.client.post(
            "/v1/detect", json={"images": [_encode_image()]}
        )
        assert resp.status_code == 503

    def test_detect_filters_to_name_class(self):
        app.state.detector = FakeDetector([
            _make_detection(cls_name="card"),
            _make_detection(cls_name="name"),
            _make_detection(cls_name="text"),
        ])
        resp = self.client.post(
            "/v1/detect", json={"images": [_encode_image()]}
        )
        data = resp.json()
        detections = data["results"][0]["detections"]
        assert len(detections) == 1
        assert detections[0]["cls"] == "name"

    def test_detect_response_meta(self):
        resp = self.client.post(
            "/v1/detect", json={"images": [_encode_image()]}
        )
        meta = resp.json()["meta"]
        assert meta["processor"] == "yolo26"
        assert meta["model"] == "test_model"

    def test_detect_multiple_images(self):
        app.state.detector = FakeDetector([_make_detection()])
        resp = self.client.post(
            "/v1/detect",
            json={"images": [_encode_image(), _encode_image()]},
        )
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["image_index"] == 0
        assert data["results"][1]["image_index"] == 1
