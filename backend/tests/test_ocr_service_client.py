"""Unit tests for OcrServiceClient."""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.services.ocr_service_client import OcrServiceClient


def _make_client(response_data=None, status_code=200, side_effect=None):
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    if side_effect:
        mock_http.post.side_effect = side_effect
    else:
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = response_data or {}
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_resp
            )
        mock_http.post.return_value = mock_resp
    return OcrServiceClient(
        client=mock_http,
        base_url="http://test:8003",
        timeout=10.0,
    )


def _make_response(texts):
    results = [{"image_index": i, "text": t} for i, t in enumerate(texts)]
    return {"results": results, "meta": {"processor": "tesseract", "version": "1.0"}}


class TestOcrServiceClient:
    @pytest.mark.asyncio
    async def test_recognize_success(self):
        response = _make_response(["Lightning Bolt", "Counterspell"])
        client = _make_client(response)

        texts = await client.recognize([b"img0", b"img1"])

        assert texts == ["Lightning Bolt", "Counterspell"]

    @pytest.mark.asyncio
    async def test_recognize_http_error_returns_empty(self):
        client = _make_client(status_code=500)

        texts = await client.recognize([b"img0", b"img1"])

        assert texts == ["", ""]

    @pytest.mark.asyncio
    async def test_recognize_timeout(self):
        client = _make_client(side_effect=httpx.TimeoutException("timeout"))

        texts = await client.recognize([b"img0"])

        assert texts == [""]

    @pytest.mark.asyncio
    async def test_recognize_empty_images(self):
        client = _make_client()

        texts = await client.recognize([])

        assert texts == []

    @pytest.mark.asyncio
    async def test_recognize_strips_whitespace(self):
        response = _make_response(["  Lightning Bolt  ", "  Counterspell\n"])
        client = _make_client(response)

        texts = await client.recognize([b"img0", b"img1"])

        assert texts == ["Lightning Bolt", "Counterspell"]

    @pytest.mark.asyncio
    async def test_recognize_pads_missing_results(self):
        response = {"results": [{"image_index": 0, "text": "Card A"}]}
        client = _make_client(response)

        texts = await client.recognize([b"img0", b"img1"])

        assert len(texts) == 2
        assert texts[0] == "Card A"
        assert texts[1] == ""
