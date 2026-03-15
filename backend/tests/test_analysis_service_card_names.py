import asyncio
from typing import Any, Dict, List, Optional

import pytest

from app.schemas import AnalyzeRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.card_name_extractor import CardNameExtractor
from app.services.card_name_resolver import CardNameResolver
from app.services.pricing_aggregator import PricingAggregator


class FakeResolver(CardNameResolver):
    def __init__(self, mapping: Dict[str, List[Dict[str, Any]]]) -> None:
        self.mapping = mapping
        self.calls: List[tuple[str, Optional[str]]] = []

    async def resolve(self, raw_name: str, set_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        self.calls.append((raw_name, set_hint))
        return self.mapping.get(raw_name, [])


class FakePricing(PricingAggregator):
    async def get_prices_for_card(self, scryfall_card: dict):
        # Return no external prices so that AnalysisService still populates
        # __noprice__ entries and persists cards.
        return []


@pytest.mark.asyncio
async def test_analyze_card_names_uses_resolver_and_handles_success(tmp_path, monkeypatch):
    # Resolver returns a single card; pricing returns no prices.
    resolver = FakeResolver(
        {
            "Tarmogoyf": [{"name": "Tarmogoyf", "id": "id-1", "set": "set1", "set_name": "Set 1"}],
        }
    )
    service = AnalysisService(name_resolver=resolver, pricing=FakePricing())

    result: AnalysisResponse = await service.analyze_card_names(["Tarmogoyf"])

    assert isinstance(result, AnalysisResponse)
    assert result.cards
    assert result.cards[0].card_name == "Tarmogoyf"
    # Resolver should have been called with the raw name and no set hint.
    assert resolver.calls == [("Tarmogoyf", None)]


class FakeExtractor(CardNameExtractor):
    """Returns fixed names per image for testing."""

    def __init__(self, names_per_image: List[List[str]]) -> None:
        self.names_per_image = names_per_image

    async def extract_names_from_images(self, images: List[bytes]) -> List[List[str]]:
        return self.names_per_image[: len(images)]


@pytest.mark.asyncio
async def test_analyze_images_with_extractor_uses_extractor_and_resolver():
    """When card_name_extractor is set, images are sent to extractor; names are resolved."""
    resolver = FakeResolver(
        {
            "Lightning Bolt": [
                {"name": "Lightning Bolt", "id": "bolt-1", "set": "2ed", "set_name": "Unlimited Edition"}
            ],
            "Island": [
                {"name": "Island", "id": "island-1", "set": "2ed", "set_name": "Unlimited Edition"}
            ],
        }
    )
    extractor = FakeExtractor([["Lightning Bolt", "Island"]])
    service = AnalysisService(
        name_resolver=resolver,
        pricing=FakePricing(),
        card_name_extractor=extractor,
    )
    request = AnalyzeRequest(urls=None)
    result = await service.analyze_images_and_urls(request, [b"fake_image_bytes"])

    assert isinstance(result, AnalysisResponse)
    assert len(result.cards) == 2
    names = {c.card_name for c in result.cards}
    assert names == {"Lightning Bolt", "Island"}
    assert resolver.calls == [("Lightning Bolt", None), ("Island", None)]

