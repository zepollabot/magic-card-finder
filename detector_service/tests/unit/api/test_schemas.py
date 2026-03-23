"""Unit tests for detector API schemas."""
from pydantic import ValidationError
import pytest

from app.api.schemas import (
    DetectRequest,
    DetectionItem,
    DetectResultItem,
    DetectResponse,
    ResponseMeta,
)


class TestDetectRequest:
    def test_requires_images(self):
        with pytest.raises(ValidationError):
            DetectRequest()

    def test_accepts_list_of_strings(self):
        req = DetectRequest(images=["abc123", "def456"])
        assert len(req.images) == 2

    def test_accepts_empty_list(self):
        req = DetectRequest(images=[])
        assert req.images == []


class TestDetectionItem:
    def test_serialization(self):
        item = DetectionItem(
            cls="name",
            confidence=0.95,
            bbox=[100, 50, 400, 90],
            crop_b64="abc123",
        )
        data = item.model_dump()
        assert data["cls"] == "name"
        assert data["bbox"] == [100, 50, 400, 90]
        assert len(data["bbox"]) == 4
        assert data["confidence"] == 0.95


class TestDetectResponse:
    def test_meta_fields(self):
        resp = DetectResponse(
            results=[],
            meta=ResponseMeta(processor="yolo26", model="best.pt"),
        )
        assert resp.meta.processor == "yolo26"
        assert resp.meta.model == "best.pt"

    def test_full_response_serialization(self):
        resp = DetectResponse(
            results=[
                DetectResultItem(
                    image_index=0,
                    detections=[
                        DetectionItem(
                            cls="name",
                            confidence=0.9,
                            bbox=[10, 20, 30, 40],
                            crop_b64="data",
                        )
                    ],
                )
            ],
            meta=ResponseMeta(processor="yolo26", model="best.pt"),
        )
        data = resp.model_dump()
        assert data["results"][0]["image_index"] == 0
        assert len(data["results"][0]["detections"]) == 1
