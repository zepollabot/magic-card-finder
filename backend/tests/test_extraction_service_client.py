"""Tests for ExtractionServiceClient."""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.services.card_name_extractor import ExtractionServiceClient


@pytest.mark.asyncio
async def test_extract_names_from_images_returns_list_per_image_from_mock_response():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(
        return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {"image_index": 0, "card_names": ["Lightning Bolt", "Island"]},
                    {"image_index": 1, "card_names": ["Tarmogoyf"]},
                ],
                "meta": {"processor": "opencv+tesseract"},
            },
            raise_for_status=MagicMock(),
        )
    )
    extractor = ExtractionServiceClient(client, base_url="http://extraction:8001")
    result = await extractor.extract_names_from_images([b"img1", b"img2"])
    assert result == [["Lightning Bolt", "Island"], ["Tarmogoyf"]]
    client.post.assert_called_once()
    call = client.post.call_args
    assert call[0][0] == "http://extraction:8001/v1/extract/cards"
    assert "images" in call[1]["json"]
    assert len(call[1]["json"]["images"]) == 2


@pytest.mark.asyncio
async def test_extract_names_from_images_empty_images_returns_empty():
    client = AsyncMock(spec=httpx.AsyncClient)
    extractor = ExtractionServiceClient(client, base_url="http://extraction:8001")
    result = await extractor.extract_names_from_images([])
    assert result == []
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_extract_names_from_images_empty_base_url_returns_empty_lists():
    client = AsyncMock(spec=httpx.AsyncClient)
    extractor = ExtractionServiceClient(client, base_url="")
    result = await extractor.extract_names_from_images([b"img1", b"img2"])
    assert result == [[], []]
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_extract_names_from_images_http_error_returns_empty_lists():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    extractor = ExtractionServiceClient(client, base_url="http://extraction:8001")
    result = await extractor.extract_names_from_images([b"img1"])
    assert result == [[]]


@pytest.mark.asyncio
async def test_extract_names_from_images_500_returns_empty_lists():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
    )
    extractor = ExtractionServiceClient(client, base_url="http://extraction:8001")
    result = await extractor.extract_names_from_images([b"img1"])
    assert result == [[]]
