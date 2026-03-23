"""Unit tests for OCR API schemas."""
from pydantic import ValidationError
import pytest

from app.api.schemas import (
    RecognizeRequest,
    RecognizeResultItem,
    RecognizeResponse,
    ResponseMeta,
)


class TestRecognizeRequest:
    def test_requires_images(self):
        with pytest.raises(ValidationError):
            RecognizeRequest()

    def test_accepts_list_of_strings(self):
        req = RecognizeRequest(images=["abc", "def"])
        assert len(req.images) == 2

    def test_accepts_empty_list(self):
        req = RecognizeRequest(images=[])
        assert req.images == []


class TestRecognizeResultItem:
    def test_serialization(self):
        item = RecognizeResultItem(image_index=0, text="Lightning Bolt")
        data = item.model_dump()
        assert data["image_index"] == 0
        assert data["text"] == "Lightning Bolt"

    def test_empty_text(self):
        item = RecognizeResultItem(image_index=1, text="")
        assert item.text == ""


class TestRecognizeResponse:
    def test_full_response(self):
        resp = RecognizeResponse(
            results=[
                RecognizeResultItem(image_index=0, text="Lightning Bolt"),
                RecognizeResultItem(image_index=1, text="Counterspell"),
            ],
            meta=ResponseMeta(processor="tesseract", version="1.0"),
        )
        data = resp.model_dump()
        assert len(data["results"]) == 2
        assert data["meta"]["processor"] == "tesseract"
        assert data["meta"]["version"] == "1.0"
