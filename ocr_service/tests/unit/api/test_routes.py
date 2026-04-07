"""Unit tests for OCR API routes."""
import base64

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


class FakeRecognizer:
    def __init__(self, text: str = "Lightning Bolt"):
        self._text = text

    async def recognize(self, image: bytes) -> str:
        return self._text


def _encode_image():
    """Return a trivially valid base64 string representing image bytes."""
    return base64.b64encode(b"fake-jpeg-bytes").decode("ascii")


class TestRecognizeRoute:
    @pytest.fixture(autouse=True)
    def _setup(self):
        app.state.recognizer = FakeRecognizer()

    @pytest.mark.asyncio
    async def test_recognize_success(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/recognize", json={"images": [_encode_image()]}
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "Lightning Bolt"

    @pytest.mark.asyncio
    async def test_recognize_invalid_base64(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/recognize", json={"images": ["not-valid!!!"]}
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_recognize_empty_images(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/recognize", json={"images": []})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    @pytest.mark.asyncio
    async def test_recognize_preserves_order(self):
        app.state.recognizer = FakeRecognizer("Card A")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/recognize",
                json={
                    "images": [_encode_image(), _encode_image(), _encode_image()]
                },
            )
        data = resp.json()
        for i, result in enumerate(data["results"]):
            assert result["image_index"] == i

    @pytest.mark.asyncio
    async def test_recognize_service_unavailable(self):
        app.state.recognizer = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/recognize", json={"images": [_encode_image()]}
            )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_recognize_response_meta(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/recognize", json={"images": [_encode_image()]}
            )
        meta = resp.json()["meta"]
        assert meta["processor"] == "ollama"
        assert meta["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_recognize_multiple_images(self):
        app.state.recognizer = FakeRecognizer("Card B")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/recognize",
                json={"images": [_encode_image(), _encode_image()]},
            )
        data = resp.json()
        assert len(data["results"]) == 2
        assert all(r["text"] == "Card B" for r in data["results"])
