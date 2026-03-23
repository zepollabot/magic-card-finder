"""Unit tests for DetectorServiceClient."""
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.services.detector_service_client import (
    DetectorServiceClient,
    DetectionResult,
    NameCrop,
)


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
    return DetectorServiceClient(
        client=mock_http,
        base_url="http://test:8002",
        timeout=10.0,
    )


def _make_response(detections_per_image=None):
    results = []
    for idx, dets in enumerate(detections_per_image or []):
        det_items = []
        for det in dets:
            crop_b64 = base64.b64encode(det.get("crop", b"fake")).decode()
            det_items.append({
                "cls": det.get("cls", "name"),
                "confidence": det.get("confidence", 0.9),
                "bbox": det.get("bbox", [10, 20, 50, 60]),
                "crop_b64": crop_b64,
            })
        results.append({"image_index": idx, "detections": det_items})
    return {"results": results, "meta": {"processor": "yolo26", "model": "test"}}


class TestDetectorServiceClient:
    @pytest.mark.asyncio
    async def test_detect_success(self):
        response = _make_response([[{"crop": b"image_data"}]])
        client = _make_client(response)

        results = await client.detect([b"raw_image"])

        assert len(results) == 1
        assert results[0].image_index == 0
        assert len(results[0].name_crops) == 1
        assert results[0].name_crops[0].image_bytes == b"image_data"

    @pytest.mark.asyncio
    async def test_detect_maps_image_indices(self):
        response = _make_response([
            [{"crop": b"a"}],
            [{"crop": b"b"}, {"crop": b"c"}],
        ])
        client = _make_client(response)

        results = await client.detect([b"img0", b"img1"])

        assert results[0].image_index == 0
        assert results[1].image_index == 1
        assert len(results[1].name_crops) == 2

    @pytest.mark.asyncio
    async def test_detect_http_error_returns_empty(self):
        client = _make_client(status_code=500)

        results = await client.detect([b"img0", b"img1"])

        assert len(results) == 2
        assert all(len(r.name_crops) == 0 for r in results)

    @pytest.mark.asyncio
    async def test_detect_timeout(self):
        client = _make_client(side_effect=httpx.TimeoutException("timeout"))

        results = await client.detect([b"img0"])

        assert len(results) == 1
        assert len(results[0].name_crops) == 0

    @pytest.mark.asyncio
    async def test_detect_malformed_response(self):
        client = _make_client({"results": [{"image_index": 0}]})

        results = await client.detect([b"img0"])

        assert len(results) == 1
        assert len(results[0].name_crops) == 0

    @pytest.mark.asyncio
    async def test_detect_empty_images(self):
        client = _make_client()

        results = await client.detect([])

        assert results == []
