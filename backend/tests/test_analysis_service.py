"""Unit tests for AnalysisService parsing, helpers, and extraction pipeline."""
import pytest
from unittest.mock import AsyncMock

from app.services.analysis_service import AnalysisService
from app.services.detector_service_client import DetectionResult, NameCrop
from app.services.progress import NoOpProgressReporter


class FakeDetectorClient:
    def __init__(self, results=None):
        self._results = results or []

    async def detect(self, images):
        return self._results


class FakeOcrClient:
    def __init__(self, texts=None):
        self._texts = texts or []

    async def recognize(self, images):
        return self._texts


def _make_service(detector_results=None, ocr_texts=None):
    return AnalysisService(
        detector_client=FakeDetectorClient(detector_results),
        ocr_client=FakeOcrClient(ocr_texts),
    )


# ------------------------------------------------------------------
# _parse_name_lines tests (existing)
# ------------------------------------------------------------------


def test_parse_name_lines_one_per_line():
    """Each non-empty line is one card; no comma = no set hint."""
    out = AnalysisService._parse_name_lines(["Llanowar Elves", "Lightning Bolt", ""])
    assert out == [
        ("Llanowar Elves", None),
        ("Lightning Bolt", None),
    ]


def test_parse_name_lines_name_comma_set():
    """Line 'Name, Set' yields (name, set)."""
    out = AnalysisService._parse_name_lines(["Lightning Bolt, Dominaria", "Tarmogoyf, Modern Masters"])
    assert out == [
        ("Lightning Bolt", "Dominaria"),
        ("Tarmogoyf", "Modern Masters"),
    ]


def test_parse_name_lines_mixed():
    """Mix of plain names and 'Name, Set'."""
    out = AnalysisService._parse_name_lines([
        "Llanowar Elves",
        "Lightning Bolt, Foundations",
        "Birds of Paradise",
    ])
    assert out == [
        ("Llanowar Elves", None),
        ("Lightning Bolt", "Foundations"),
        ("Birds of Paradise", None),
    ]


def test_parse_name_lines_skips_empty():
    """Empty and whitespace-only lines are skipped."""
    out = AnalysisService._parse_name_lines(["  A  ", "", "  ", "B, SET"])
    assert out == [
        ("A", None),
        ("B", "SET"),
    ]


# ------------------------------------------------------------------
# _extract_names_from_images tests (new)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_names_detection_then_ocr():
    """Detector returns crops, OCR returns names, result is flat list."""
    detector_results = [
        DetectionResult(
            image_index=0,
            name_crops=[
                NameCrop(bbox=(10, 20, 50, 60), confidence=0.9, image_bytes=b"crop1"),
                NameCrop(bbox=(70, 20, 110, 60), confidence=0.8, image_bytes=b"crop2"),
            ],
        ),
    ]
    ocr_texts = ["Lightning Bolt", "Counterspell"]

    service = _make_service(detector_results, ocr_texts)
    step_index = {"card_detection": 1, "card_recognition": 2}
    reporter = NoOpProgressReporter()

    names = await service._extract_names_from_images(
        [b"raw_image"], step_index, reporter
    )

    assert names == ["Lightning Bolt", "Counterspell"]


@pytest.mark.asyncio
async def test_extract_names_empty_detection_returns_empty():
    """When detector finds nothing, OCR is not called, result is empty."""
    detector_results = [DetectionResult(image_index=0, name_crops=[])]

    service = _make_service(detector_results, [])
    step_index = {"card_detection": 1, "card_recognition": 2}
    reporter = NoOpProgressReporter()

    names = await service._extract_names_from_images(
        [b"raw_image"], step_index, reporter
    )

    assert names == []


@pytest.mark.asyncio
async def test_extract_names_filters_empty_ocr_results():
    """Empty/whitespace OCR results are filtered out."""
    detector_results = [
        DetectionResult(
            image_index=0,
            name_crops=[
                NameCrop(bbox=(10, 20, 50, 60), confidence=0.9, image_bytes=b"crop1"),
                NameCrop(bbox=(70, 20, 110, 60), confidence=0.8, image_bytes=b"crop2"),
            ],
        ),
    ]
    ocr_texts = ["Lightning Bolt", "", "  "]

    service = _make_service(detector_results, ocr_texts)
    step_index = {"card_detection": 1, "card_recognition": 2}
    reporter = NoOpProgressReporter()

    names = await service._extract_names_from_images(
        [b"raw_image"], step_index, reporter
    )

    assert names == ["Lightning Bolt"]
