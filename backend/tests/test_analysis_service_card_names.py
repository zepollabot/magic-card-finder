"""Tests for AnalysisService: card name and image extraction flows."""
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import AnalyzeRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.detector_service_client import DetectionResult, NameCrop
from app.services.card_name_resolver import CardNameResolver
from app.services.pricing_aggregator import PricingAggregator
from app.services.scryfall_client import ScryfallClient


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

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


class FakeResolver(CardNameResolver):
    def __init__(self, mapping: Dict[str, List[Dict[str, Any]]]) -> None:
        self.mapping = mapping
        self.calls: List[tuple[str, Optional[str]]] = []

    async def resolve(self, raw_name: str, set_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        self.calls.append((raw_name, set_hint))
        return self.mapping.get(raw_name, [])


class FakePricing(PricingAggregator):
    def __init__(self) -> None:
        super().__init__(sources=[])

    async def get_prices_for_card(self, scryfall_card: dict):
        return []


class FakeScryfallClient(ScryfallClient):
    """Returns each resolved card as its only printing."""

    def __init__(self, printings_map: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        self._printings_map = printings_map or {}

    async def search_printings(self, exact_name: str) -> List[Dict[str, Any]]:
        return self._printings_map.get(exact_name, [])


def _stub_db():
    """Patch SessionLocal to return an in-memory mock that accepts ORM calls."""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.one_or_none.return_value = None

    call_count = {"n": 0}

    def _track_add(obj):
        call_count["n"] += 1
        if hasattr(obj, "id") and obj.id is None:
            obj.id = call_count["n"]

    db.add.side_effect = _track_add
    return db


def _make_service(
    resolver_map=None,
    scryfall_map=None,
    detector_results=None,
    ocr_texts=None,
):
    resolver = FakeResolver(resolver_map or {})
    scryfall = FakeScryfallClient(scryfall_map or {})
    return AnalysisService(
        detector_client=FakeDetectorClient(detector_results),
        ocr_client=FakeOcrClient(ocr_texts),
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
    ), resolver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TARMOGOYF = {"name": "Tarmogoyf", "id": "id-1", "set": "set1", "set_name": "Set 1",
             "collector_number": "1", "image_uris": {"normal": "http://img/1", "small": "http://img/1s"}}

BOLT = {"name": "Lightning Bolt", "id": "bolt-1", "set": "2ed", "set_name": "Unlimited Edition",
        "collector_number": "161", "image_uris": {"normal": "http://img/bolt", "small": "http://img/bolt-s"}}

ISLAND = {"name": "Island", "id": "island-1", "set": "2ed", "set_name": "Unlimited Edition",
          "collector_number": "291", "image_uris": {"normal": "http://img/island", "small": "http://img/island-s"}}


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_uses_resolver_and_handles_success(mock_session_local):
    mock_session_local.return_value = _stub_db()

    service, resolver = _make_service(
        resolver_map={"Tarmogoyf": [TARMOGOYF]},
        scryfall_map={"Tarmogoyf": [TARMOGOYF]},
    )

    result: AnalysisResponse = await service.analyze_card_names(["Tarmogoyf"])

    assert isinstance(result, AnalysisResponse)
    assert len(result.cards) == 1
    assert result.cards[0].card_name == "Tarmogoyf"
    assert resolver.calls == [("Tarmogoyf", None)]


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_with_set_hint(mock_session_local):
    mock_session_local.return_value = _stub_db()

    service, resolver = _make_service(
        resolver_map={"Lightning Bolt": [BOLT]},
        scryfall_map={"Lightning Bolt": [BOLT]},
    )

    result = await service.analyze_card_names(["Lightning Bolt, Unlimited Edition"])

    assert len(result.cards) == 1
    assert result.cards[0].card_name == "Lightning Bolt"
    assert resolver.calls == [("Lightning Bolt", "Unlimited Edition")]


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_deduplicates_canonical(mock_session_local):
    mock_session_local.return_value = _stub_db()

    service, _ = _make_service(
        resolver_map={"Tarmogoyf": [TARMOGOYF]},
        scryfall_map={"Tarmogoyf": [TARMOGOYF]},
    )

    result = await service.analyze_card_names(["Tarmogoyf", "Tarmogoyf"])

    assert len(result.cards) == 1


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_unresolvable_returns_empty(mock_session_local):
    mock_session_local.return_value = _stub_db()

    service, _ = _make_service()

    result = await service.analyze_card_names(["NonexistentCard"])

    assert isinstance(result, AnalysisResponse)
    assert len(result.cards) == 0


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_multiple_printings(mock_session_local):
    """When search_printings returns multiple printings, all appear as priced entries."""
    mock_session_local.return_value = _stub_db()

    tarmogoyf_mm = {**TARMOGOYF, "id": "id-2", "set": "mma", "set_name": "Modern Masters"}
    service, _ = _make_service(
        resolver_map={"Tarmogoyf": [TARMOGOYF]},
        scryfall_map={"Tarmogoyf": [TARMOGOYF, tarmogoyf_mm]},
    )

    result = await service.analyze_card_names(["Tarmogoyf"])

    assert len(result.cards) == 1
    sets = {p.set_name for p in result.cards[0].prices}
    assert "Set 1" in sets
    assert "Modern Masters" in sets


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_images_with_detector_and_ocr(mock_session_local):
    """Images go through detector then OCR; recognized names are resolved."""
    mock_session_local.return_value = _stub_db()

    detector_results = [
        DetectionResult(
            image_index=0,
            name_crops=[
                NameCrop(bbox=(10, 20, 50, 60), confidence=0.9, image_bytes=b"crop1"),
                NameCrop(bbox=(70, 20, 110, 60), confidence=0.8, image_bytes=b"crop2"),
            ],
        ),
    ]
    ocr_texts = ["Lightning Bolt", "Island"]
    service, resolver = _make_service(
        resolver_map={"Lightning Bolt": [BOLT], "Island": [ISLAND]},
        scryfall_map={"Lightning Bolt": [BOLT], "Island": [ISLAND]},
        detector_results=detector_results,
        ocr_texts=ocr_texts,
    )
    request = AnalyzeRequest(urls=None)
    result = await service.analyze_images_and_urls(request, [b"fake_image_bytes"])

    assert isinstance(result, AnalysisResponse)
    assert len(result.cards) == 2
    names = {c.card_name for c in result.cards}
    assert names == {"Lightning Bolt", "Island"}
    assert resolver.calls == [("Lightning Bolt", None), ("Island", None)]


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_images_empty_names_skipped(mock_session_local):
    """Empty or whitespace-only names from OCR should be skipped."""
    mock_session_local.return_value = _stub_db()

    detector_results = [
        DetectionResult(
            image_index=0,
            name_crops=[
                NameCrop(bbox=(10, 20, 50, 60), confidence=0.9, image_bytes=b"crop1"),
                NameCrop(bbox=(70, 20, 110, 60), confidence=0.8, image_bytes=b"crop2"),
                NameCrop(bbox=(120, 20, 160, 60), confidence=0.7, image_bytes=b"crop3"),
            ],
        ),
    ]
    ocr_texts = ["Lightning Bolt", "", "  "]
    service, resolver = _make_service(
        resolver_map={"Lightning Bolt": [BOLT]},
        scryfall_map={"Lightning Bolt": [BOLT]},
        detector_results=detector_results,
        ocr_texts=ocr_texts,
    )
    request = AnalyzeRequest(urls=None)
    result = await service.analyze_images_and_urls(request, [b"img"])

    assert len(result.cards) == 1
    assert result.cards[0].card_name == "Lightning Bolt"
    assert resolver.calls == [("Lightning Bolt", None)]


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_images_and_card_names_produce_same_structure(mock_session_local):
    """Both flows should produce identical report structure for the same card."""
    mock_session_local.return_value = _stub_db()

    service_names, _ = _make_service(
        resolver_map={"Lightning Bolt": [BOLT]},
        scryfall_map={"Lightning Bolt": [BOLT]},
    )
    names_result = await service_names.analyze_card_names(["Lightning Bolt"])

    mock_session_local.return_value = _stub_db()
    detector_results = [
        DetectionResult(
            image_index=0,
            name_crops=[NameCrop(bbox=(10, 20, 50, 60), confidence=0.9, image_bytes=b"crop")],
        ),
    ]
    service_img, _ = _make_service(
        resolver_map={"Lightning Bolt": [BOLT]},
        scryfall_map={"Lightning Bolt": [BOLT]},
        detector_results=detector_results,
        ocr_texts=["Lightning Bolt"],
    )
    img_result = await service_img.analyze_images_and_urls(
        AnalyzeRequest(urls=None), [b"img"],
    )

    assert len(names_result.cards) == len(img_result.cards) == 1
    assert names_result.cards[0].card_name == img_result.cards[0].card_name
    assert len(names_result.cards[0].prices) == len(img_result.cards[0].prices)
