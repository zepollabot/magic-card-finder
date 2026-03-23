"""Unit tests for OCR API routes."""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


class FakeRecognizer:
    def __init__(self, text: str = "Lightning Bolt"):
        self._text = text

    def recognize(self, image):
        return self._text


def _encode_image(h=40, w=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


class TestRecognizeRoute:
    def setup_method(self):
        app.state.recognizer = FakeRecognizer()
        self.client = TestClient(app)

    def test_recognize_success(self):
        resp = self.client.post(
            "/v1/recognize", json={"images": [_encode_image()]}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "Lightning Bolt"

    def test_recognize_invalid_base64(self):
        resp = self.client.post(
            "/v1/recognize", json={"images": ["not-valid!!!"]}
        )
        assert resp.status_code == 400

    def test_recognize_empty_images(self):
        resp = self.client.post("/v1/recognize", json={"images": []})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_recognize_preserves_order(self):
        app.state.recognizer = FakeRecognizer("Card A")
        resp = self.client.post(
            "/v1/recognize",
            json={"images": [_encode_image(), _encode_image(), _encode_image()]},
        )
        data = resp.json()
        for i, result in enumerate(data["results"]):
            assert result["image_index"] == i

    def test_recognize_service_unavailable(self):
        app.state.recognizer = None
        resp = self.client.post(
            "/v1/recognize", json={"images": [_encode_image()]}
        )
        assert resp.status_code == 503

    def test_recognize_response_meta(self):
        resp = self.client.post(
            "/v1/recognize", json={"images": [_encode_image()]}
        )
        meta = resp.json()["meta"]
        assert meta["processor"] == "tesseract"
        assert meta["version"] == "1.0"
