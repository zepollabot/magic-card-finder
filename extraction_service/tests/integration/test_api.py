"""Integration tests for POST /v1/extract/cards."""
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _encode_image(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_extract_cards_requires_images(client):
    """Empty or missing images -> 422."""
    resp = client.post("/v1/extract/cards", json={})
    assert resp.status_code == 422


def test_extract_cards_invalid_base64_returns_400(client):
    resp = client.post(
        "/v1/extract/cards",
        json={"images": ["not-valid-base64!!!"]},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "detail" in data


def test_extract_cards_valid_request_returns_200_and_schema(client):
    """One small image (no card) -> 200, results with one entry, empty card_names."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = _encode_image(img)
    resp = client.post("/v1/extract/cards", json={"images": [b64]})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["image_index"] == 0
    assert "card_names" in data["results"][0]
    assert isinstance(data["results"][0]["card_names"], list)
    assert "meta" in data
    assert data["meta"]["processor"] == "opencv+tesseract"


def test_extract_cards_multiple_images_returns_one_result_per_image(client):
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    b64 = _encode_image(img)
    resp = client.post("/v1/extract/cards", json={"images": [b64, b64, b64]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert [r["image_index"] for r in data["results"]] == [0, 1, 2]


def test_extract_cards_card_like_rectangle_may_return_name(client):
    """Image with one card-like rectangle: one result entry, possibly one name (OCR)."""
    h, w = 200, 290
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[50 : 50 + h, 50 : 50 + w] = 255
    b64 = _encode_image(img)
    resp = client.post("/v1/extract/cards", json={"images": [b64]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["image_index"] == 0
    # May have 0 or 1 card name depending on Tesseract
    assert len(data["results"][0]["card_names"]) <= 1
