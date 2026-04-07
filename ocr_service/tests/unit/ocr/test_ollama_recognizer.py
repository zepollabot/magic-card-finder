"""Unit tests for OllamaTextRecognizer."""
import base64
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.ocr.ollama_recognizer import OllamaTextRecognizer


def _make_recognizer(
    response_data=None,
    status_code=200,
    side_effect=None,
):
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    if side_effect:
        mock_client.post.side_effect = side_effect
    else:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status_code
        mock_resp.json.return_value = response_data or {}
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_resp
            )
        mock_client.post.return_value = mock_resp

    recognizer = OllamaTextRecognizer(
        client=mock_client,
        base_url="http://test-ollama:11434",
        model="llava",
        timeout=30.0,
    )
    return recognizer, mock_client


class TestOllamaTextRecognizer:
    @pytest.mark.asyncio
    async def test_recognize_success(self):
        recognizer, _ = _make_recognizer({"response": "Lightning Bolt"})

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == "Lightning Bolt"

    @pytest.mark.asyncio
    async def test_recognize_strips_whitespace(self):
        recognizer, _ = _make_recognizer({"response": "  Lightning Bolt  \n"})

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == "Lightning Bolt"

    @pytest.mark.asyncio
    async def test_recognize_empty_response(self):
        recognizer, _ = _make_recognizer({"response": ""})

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == ""

    @pytest.mark.asyncio
    async def test_recognize_missing_response_field(self):
        recognizer, _ = _make_recognizer({"some_other_field": "value"})

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == ""

    @pytest.mark.asyncio
    async def test_recognize_http_error_returns_empty(self):
        recognizer, _ = _make_recognizer(status_code=500)

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == ""

    @pytest.mark.asyncio
    async def test_recognize_timeout_returns_empty(self):
        recognizer, _ = _make_recognizer(
            side_effect=httpx.TimeoutException("timeout")
        )

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == ""

    @pytest.mark.asyncio
    async def test_recognize_connection_error_returns_empty(self):
        recognizer, _ = _make_recognizer(
            side_effect=httpx.ConnectError("connection refused")
        )

        text = await recognizer.recognize(b"fake-image-bytes")

        assert text == ""

    @pytest.mark.asyncio
    async def test_recognize_sends_correct_payload(self):
        recognizer, mock_client = _make_recognizer(
            {"response": "Counterspell"}
        )
        image_data = b"test-image-content"

        await recognizer.recognize(image_data)

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.args[0] == "http://test-ollama:11434/api/generate"

        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "llava"
        assert payload["stream"] is False
        assert payload["images"] == [
            base64.b64encode(image_data).decode("ascii")
        ]
        assert "extract the text" in payload["prompt"]

    @pytest.mark.asyncio
    async def test_recognize_uses_configured_timeout(self):
        recognizer, mock_client = _make_recognizer(
            {"response": "Counterspell"}
        )

        await recognizer.recognize(b"img")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_model_property(self):
        recognizer, _ = _make_recognizer()

        assert recognizer.model == "llava"

    @pytest.mark.asyncio
    async def test_base_url_trailing_slash_stripped(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Bolt"}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        recognizer = OllamaTextRecognizer(
            client=mock_client,
            base_url="http://test-ollama:11434/",
            model="llava",
        )

        await recognizer.recognize(b"img")

        url = mock_client.post.call_args.args[0]
        assert url == "http://test-ollama:11434/api/generate"
