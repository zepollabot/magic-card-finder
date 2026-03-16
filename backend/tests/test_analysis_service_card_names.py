"""Tests for AnalysisService: card name and image extraction flows."""
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import AnalyzeRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.card_name_extractor import CardNameExtractor
from app.services.card_name_resolver import CardNameResolver
from app.services.pricing_aggregator import PricingAggregator
from app.services.scryfall_client import ScryfallClient


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

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


class FakeExtractor(CardNameExtractor):
    """Returns fixed names per image for testing."""

    def __init__(self, names_per_image: List[List[str]]) -> None:
        self.names_per_image = names_per_image

    async def extract_names_from_images(self, images: List[bytes]) -> List[List[str]]:
        return self.names_per_image[: len(images)]


def _stub_db():
    """Patch SessionLocal to return an in-memory mock that accepts ORM calls."""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.one_or_none.return_value = None

    mock_analysis = MagicMock()
    mock_analysis.id = 42
    mock_card = MagicMock()
    mock_card.id = 1
    mock_card.name = "stub"
    mock_card.set_name = None
    mock_card.collector_number = None
    mock_card.image_url = None
    mock_card.thumbnail_url = None
    mock_ac = MagicMock()
    mock_ac.id = 1

    call_count = {"n": 0}
    original_add = db.add

    def _track_add(obj):
        call_count["n"] += 1
        if hasattr(obj, "id") and obj.id is None:
            obj.id = call_count["n"]

    db.add.side_effect = _track_add
    return db


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

    resolver = FakeResolver({"Tarmogoyf": [TARMOGOYF]})
    scryfall = FakeScryfallClient({"Tarmogoyf": [TARMOGOYF]})
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
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

    resolver = FakeResolver({"Lightning Bolt": [BOLT]})
    scryfall = FakeScryfallClient({"Lightning Bolt": [BOLT]})
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
    )

    result = await service.analyze_card_names(["Lightning Bolt, Unlimited Edition"])

    assert len(result.cards) == 1
    assert result.cards[0].card_name == "Lightning Bolt"
    assert resolver.calls == [("Lightning Bolt", "Unlimited Edition")]


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_deduplicates_canonical(mock_session_local):
    mock_session_local.return_value = _stub_db()

    resolver = FakeResolver({"Tarmogoyf": [TARMOGOYF]})
    scryfall = FakeScryfallClient({"Tarmogoyf": [TARMOGOYF]})
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
    )

    result = await service.analyze_card_names(["Tarmogoyf", "Tarmogoyf"])

    assert len(result.cards) == 1


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_unresolvable_returns_empty(mock_session_local):
    mock_session_local.return_value = _stub_db()

    resolver = FakeResolver({})
    scryfall = FakeScryfallClient({})
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
    )

    result = await service.analyze_card_names(["NonexistentCard"])

    assert isinstance(result, AnalysisResponse)
    assert len(result.cards) == 0


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_card_names_multiple_printings(mock_session_local):
    """When search_printings returns multiple printings, all appear as priced entries."""
    mock_session_local.return_value = _stub_db()

    tarmogoyf_mm = {**TARMOGOYF, "id": "id-2", "set": "mma", "set_name": "Modern Masters"}
    resolver = FakeResolver({"Tarmogoyf": [TARMOGOYF]})
    scryfall = FakeScryfallClient({"Tarmogoyf": [TARMOGOYF, tarmogoyf_mm]})
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
    )

    result = await service.analyze_card_names(["Tarmogoyf"])

    assert len(result.cards) == 1
    sets = {p.set_name for p in result.cards[0].prices}
    assert "Set 1" in sets
    assert "Modern Masters" in sets


@pytest.mark.asyncio
@patch("app.services.analysis_service.SessionLocal")
async def test_analyze_images_with_extractor_uses_extractor_and_resolver(mock_session_local):
    """When card_name_extractor is set, images are sent to extractor;
    extracted names are resolved through the same shared pipeline."""
    mock_session_local.return_value = _stub_db()

    resolver = FakeResolver({
        "Lightning Bolt": [BOLT],
        "Island": [ISLAND],
    })
    scryfall = FakeScryfallClient({
        "Lightning Bolt": [BOLT],
        "Island": [ISLAND],
    })
    extractor = FakeExtractor([["Lightning Bolt", "Island"]])
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
        card_name_extractor=extractor,
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
async def test_analyze_images_with_extractor_empty_names_skipped(mock_session_local):
    """Empty or whitespace-only names from the extractor should be skipped."""
    mock_session_local.return_value = _stub_db()

    resolver = FakeResolver({"Lightning Bolt": [BOLT]})
    scryfall = FakeScryfallClient({"Lightning Bolt": [BOLT]})
    extractor = FakeExtractor([["Lightning Bolt", "", "  "]])
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        scryfall=scryfall,
        card_name_extractor=extractor,
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

    def _make_service(extractor=None):
        return AnalysisService(
            name_resolver=FakeResolver({"Lightning Bolt": [BOLT]}),
            pricing=FakePricing(),
            scryfall=FakeScryfallClient({"Lightning Bolt": [BOLT]}),
            card_name_extractor=extractor,
        )

    names_result = await _make_service().analyze_card_names(["Lightning Bolt"])

    mock_session_local.return_value = _stub_db()
    extractor = FakeExtractor([["Lightning Bolt"]])
    img_result = await _make_service(extractor).analyze_images_and_urls(
        AnalyzeRequest(urls=None), [b"img"],
    )

    assert len(names_result.cards) == len(img_result.cards) == 1
    assert names_result.cards[0].card_name == img_result.cards[0].card_name
    assert len(names_result.cards[0].prices) == len(img_result.cards[0].prices)
