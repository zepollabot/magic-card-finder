"""
Comprehensive tests for pricing: ScryfallPriceSource, OpenTCGClient,
CardmarketClient, and PricingAggregator.
"""
import os
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas import CardPriceInfo
from app.services.price_sources import (
    PriceSource,
    ScryfallPriceSource,
    get_default_price_sources,
)
from app.services.open_tcg_client import OpenTCGClient
from app.services.cardmarket_client import CardmarketClient
from app.services.cardtrader_client import CardTraderClient
from app.services.pricing_aggregator import PricingAggregator


# -----------------------------------------------------------------------------
# ScryfallPriceSource
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scryfall_price_source_is_always_enabled():
    source = ScryfallPriceSource()
    assert source.is_enabled() is True


@pytest.mark.asyncio
async def test_scryfall_price_source_extracts_usd_and_eur():
    source = ScryfallPriceSource()
    card = {
        "name": "Llanowar Elves",
        "prices": {"usd": "0.25", "eur": "0.22"},
    }
    result = await source.get_prices_for_card(card)
    assert len(result) == 2
    by_currency = {p.currency: p for p in result}
    assert by_currency["USD"].price_avg == 0.25
    assert by_currency["USD"].source == "scryfall"
    assert by_currency["EUR"].price_avg == 0.22
    assert by_currency["EUR"].source == "scryfall"


@pytest.mark.asyncio
async def test_scryfall_price_source_handles_missing_prices():
    source = ScryfallPriceSource()
    card = {"name": "Black Lotus", "prices": {}}
    result = await source.get_prices_for_card(card)
    assert result == []


@pytest.mark.asyncio
async def test_scryfall_price_source_handles_numeric_prices():
    source = ScryfallPriceSource()
    card = {"prices": {"usd": 1.5, "eur": 1.2}}
    result = await source.get_prices_for_card(card)
    assert len(result) == 2
    assert result[0].price_avg == 1.5
    assert result[1].price_avg == 1.2


@pytest.mark.asyncio
async def test_scryfall_price_source_skips_invalid_price_values():
    source = ScryfallPriceSource()
    card = {"prices": {"usd": "n/a", "eur": "0.10"}}
    result = await source.get_prices_for_card(card)
    assert len(result) == 1
    assert result[0].currency == "EUR"


# -----------------------------------------------------------------------------
# OpenTCGClient
# -----------------------------------------------------------------------------


def test_open_tcg_client_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPEN_TCG_API_ENABLED", raising=False)
    client = OpenTCGClient()
    assert client.is_enabled() is False


def test_open_tcg_client_enabled_via_env(monkeypatch):
    monkeypatch.setenv("OPEN_TCG_API_ENABLED", "1")
    client = OpenTCGClient()
    assert client.is_enabled() is True


def test_open_tcg_client_enabled_override_constructor():
    client = OpenTCGClient(enabled=True)
    assert client.is_enabled() is True
    client = OpenTCGClient(enabled=False)
    assert client.is_enabled() is False


@pytest.mark.asyncio
async def test_open_tcg_client_returns_empty_when_disabled():
    client = OpenTCGClient(enabled=False)
    result = await client.get_prices_for_card({"name": "Test", "id": "abc"})
    assert result == []


@pytest.mark.asyncio
async def test_open_tcg_client_returns_empty_when_no_card_name():
    client = OpenTCGClient(enabled=True)
    result = await client.get_prices_for_card({})
    assert result == []


@pytest.mark.asyncio
async def test_open_tcg_client_full_flow_mocked():
    """Test Open TCG flow with mocked HTTP: search -> products -> pricing."""
    client = OpenTCGClient(enabled=True, base_url="https://fake.api/v1")

    search_response = MagicMock()
    search_response.raise_for_status = MagicMock()
    search_response.json.return_value = {
        "sets": [{"id": 23556, "name": "Foundations", "abbreviation": "FDN"}],
    }

    products_response = MagicMock()
    products_response.raise_for_status = MagicMock()
    products_response.json.return_value = {
        "products": [
            {
                "id": 557921,
                "name": "Llanowar Elves",
                "scryfall_id": "6a0b230b-1234-5678-9abc-def012345678",
            },
        ],
    }

    pricing_response = MagicMock()
    pricing_response.raise_for_status = MagicMock()
    pricing_response.json.return_value = {
        "prices": {
            "557921": {
                "tcg": {"Normal": {"market": 0.22, "low": 0.18}},
                "manapool": {},
            },
        },
    }

    async def mock_get(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if "search" in url:
            return search_response
        if "/sets/23556/pricing" in url:
            return pricing_response
        if "/sets/23556" in url and "pricing" not in url:
            return products_response
        return MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value={}))

    mock_client_instance = MagicMock()
    mock_client_instance.get = AsyncMock(side_effect=mock_get)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("app.services.open_tcg_client.httpx.AsyncClient", return_value=mock_client_instance):
        result = await client.get_prices_for_card(
            {
                "name": "Llanowar Elves",
                "set_name": "Foundations",
                "id": "6a0b230b-1234-5678-9abc-def012345678",
            }
        )

    assert len(result) == 1
    assert result[0].source == "opentcg"
    assert result[0].currency == "USD"
    assert result[0].price_avg == 0.22


@pytest.mark.asyncio
async def test_open_tcg_client_uses_manapool_when_no_tcg():
    """When tcg is empty, parse manapool prices."""
    client = OpenTCGClient(enabled=True, base_url="https://fake.api/v1")

    search_response = MagicMock()
    search_response.raise_for_status = MagicMock()
    search_response.json.return_value = {"sets": [{"id": 1, "name": "Set"}]}

    products_response = MagicMock()
    products_response.raise_for_status = MagicMock()
    products_response.json.return_value = {
        "products": [{"id": 100, "name": "Card"}],
    }

    pricing_response = MagicMock()
    pricing_response.raise_for_status = MagicMock()
    pricing_response.json.return_value = {
        "prices": {
            "100": {"tcg": {}, "manapool": {"normal": 0.15}},
        },
    }

    async def mock_get(*args, **kwargs):
        url = args[0] if args else ""
        if "search" in url:
            return search_response
        if "pricing" in url:
            return pricing_response
        return products_response

    mock_client_instance = MagicMock()
    mock_client_instance.get = AsyncMock(side_effect=mock_get)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("app.services.open_tcg_client.httpx.AsyncClient", return_value=mock_client_instance):
        result = await client.get_prices_for_card({"name": "Card", "set_name": "Set"})

    assert len(result) == 1
    assert result[0].source == "opentcg"
    assert result[0].price_avg == 0.15


# -----------------------------------------------------------------------------
# CardmarketClient
# -----------------------------------------------------------------------------


def test_cardmarket_client_disabled_without_env(monkeypatch):
    monkeypatch.delenv("CARDMARKET_API_ENABLED", raising=False)
    monkeypatch.delenv("CARDMARKET_APP_TOKEN", raising=False)
    client = CardmarketClient()
    assert client.is_enabled() is False


def test_cardmarket_client_disabled_without_credentials(monkeypatch):
    monkeypatch.setenv("CARDMARKET_API_ENABLED", "1")
    monkeypatch.delenv("CARDMARKET_APP_TOKEN", raising=False)
    client = CardmarketClient()
    assert client.is_enabled() is False


def test_cardmarket_client_enabled_with_credentials(monkeypatch):
    monkeypatch.setenv("CARDMARKET_API_ENABLED", "1")
    monkeypatch.setenv("CARDMARKET_APP_TOKEN", "t")
    monkeypatch.setenv("CARDMARKET_APP_SECRET", "s")
    monkeypatch.setenv("CARDMARKET_ACCESS_TOKEN", "at")
    monkeypatch.setenv("CARDMARKET_ACCESS_TOKEN_SECRET", "as")
    client = CardmarketClient()
    assert client.is_enabled() is True


def test_cardmarket_client_constructor_override():
    client = CardmarketClient(
        enabled=True,
        app_token="t",
        app_secret="s",
        access_token="at",
        access_secret="as",
    )
    assert client.is_enabled() is True


@pytest.mark.asyncio
async def test_cardmarket_client_returns_empty_when_disabled():
    client = CardmarketClient(enabled=False)
    result = await client.get_prices_for_card({"name": "Test"})
    assert result == []


@pytest.mark.asyncio
async def test_cardmarket_client_returns_empty_when_no_card_name():
    client = CardmarketClient(
        enabled=True,
        app_token="t",
        app_secret="s",
        access_token="at",
        access_secret="as",
    )
    result = await client.get_prices_for_card({})
    assert result == []


@pytest.mark.asyncio
async def test_cardmarket_client_extract_product_id_from_list():
    client = CardmarketClient(
        enabled=True,
        app_token="t",
        app_secret="s",
        access_token="at",
        access_secret="as",
    )
    assert client._extract_product_id([{"idProduct": 12345}]) == 12345
    assert client._extract_product_id([{"id": 999}]) == 999


@pytest.mark.asyncio
async def test_cardmarket_client_parse_numeric_price():
    client = CardmarketClient(enabled=True, app_token="t", app_secret="s", access_token="at", access_secret="as")
    result = client._parse_product_prices({"price": 2.50})
    assert len(result) == 1
    assert result[0].source == "cardmarket"
    assert result[0].currency == "EUR"
    assert result[0].price_avg == 2.50


@pytest.mark.asyncio
async def test_cardmarket_client_parse_price_guide_dict():
    client = CardmarketClient(enabled=True, app_token="t", app_secret="s", access_token="at", access_secret="as")
    result = client._parse_product_prices({
        "priceGuide": {"avg": 1.5, "low": 1.2, "trend": 1.4},
    })
    assert len(result) == 1
    assert result[0].price_avg == 1.5
    assert result[0].price_low == 1.2
    assert result[0].trend_price == 1.4


@pytest.mark.asyncio
async def test_cardmarket_client_full_flow_mocked():
    """Test Cardmarket find + product with mocked OAuth session."""
    client = CardmarketClient(
        enabled=True,
        app_token="t",
        app_secret="s",
        access_token="at",
        access_secret="as",
    )

    find_response = MagicMock()
    find_response.raise_for_status = MagicMock()
    find_response.json.return_value = [{"idProduct": 265535}]

    product_response = MagicMock()
    product_response.raise_for_status = MagicMock()
    product_response.json.return_value = {
        "priceGuide": {"avg": 3.25, "low": 2.80, "trend": 3.10},
    }

    mock_session = MagicMock()
    mock_session.get.side_effect = [find_response, product_response]

    with patch("app.services.cardmarket_client.OAuth1Session", return_value=mock_session):
        result = await client.get_prices_for_card({"name": "Tarmogoyf"})

    assert len(result) == 1
    assert result[0].source == "cardmarket"
    assert result[0].currency == "EUR"
    assert result[0].price_avg == 3.25
    assert result[0].trend_price == 3.10


# -----------------------------------------------------------------------------
# PricingAggregator
# -----------------------------------------------------------------------------


class FakePriceSource:
    """Minimal PriceSource for testing aggregator."""

    def __init__(self, enabled: bool = True, prices: List[CardPriceInfo] | None = None):
        self._enabled = enabled
        self._prices = prices or []

    def is_enabled(self) -> bool:
        return self._enabled

    def get_source_name(self) -> str:
        if self._prices:
            return self._prices[0].source or "fake"
        return "fake"

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        return list(self._prices)


@pytest.mark.asyncio
async def test_pricing_aggregator_merges_multiple_sources():
    source_a = FakePriceSource(prices=[CardPriceInfo(source="a", currency="USD", price_avg=1.0)])
    source_b = FakePriceSource(prices=[CardPriceInfo(source="b", currency="EUR", price_avg=0.9)])
    aggregator = PricingAggregator(sources=[source_a, source_b])
    result = await aggregator.get_prices_for_card({"name": "Card"})
    assert len(result) == 2
    assert result[0].source == "a"
    assert result[1].source == "b"


@pytest.mark.asyncio
async def test_pricing_aggregator_skips_disabled_sources():
    source_on = FakePriceSource(enabled=True, prices=[CardPriceInfo(source="on", currency="USD", price_avg=1.0)])
    source_off = FakePriceSource(enabled=False, prices=[CardPriceInfo(source="off", currency="EUR", price_avg=2.0)])
    aggregator = PricingAggregator(sources=[source_on, source_off])
    result = await aggregator.get_prices_for_card({"name": "Card"})
    assert len(result) == 1
    assert result[0].source == "on"


@pytest.mark.asyncio
async def test_pricing_aggregator_continues_on_source_failure():
    class FailingSource(FakePriceSource):
        async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
            raise RuntimeError("API down")

    source_ok = FakePriceSource(prices=[CardPriceInfo(source="ok", currency="USD", price_avg=1.0)])
    source_fail = FailingSource(enabled=True)
    aggregator = PricingAggregator(sources=[source_ok, source_fail])
    result = await aggregator.get_prices_for_card({"name": "Card"})
    assert len(result) == 1
    assert result[0].source == "ok"


@pytest.mark.asyncio
async def test_pricing_aggregator_default_sources_include_scryfall():
    aggregator = PricingAggregator()  # uses get_default_price_sources()
    card = {"name": "Test", "prices": {"usd": "1.00"}}
    result = await aggregator.get_prices_for_card(card)
    # At least Scryfall should return a price; Open TCG/Cardmarket may be disabled
    assert any(p.source == "scryfall" for p in result)
    assert any(p.currency == "USD" and p.price_avg == 1.0 for p in result)


@pytest.mark.asyncio
async def test_pricing_aggregator_empty_sources_returns_empty():
    aggregator = PricingAggregator(sources=[])
    result = await aggregator.get_prices_for_card({"name": "Card"})
    assert result == []


# -----------------------------------------------------------------------------
# Protocol & factory
# -----------------------------------------------------------------------------


def test_open_tcg_client_satisfies_price_source_protocol():
    """OpenTCGClient is a valid PriceSource (structural subtyping)."""
    client = OpenTCGClient(enabled=False)
    assert isinstance(client, PriceSource)


def test_cardmarket_client_satisfies_price_source_protocol():
    """CardmarketClient is a valid PriceSource."""
    client = CardmarketClient(enabled=False)
    assert isinstance(client, PriceSource)


def test_scryfall_price_source_satisfies_protocol():
    """ScryfallPriceSource is a valid PriceSource."""
    source = ScryfallPriceSource()
    assert isinstance(source, PriceSource)


def test_get_default_price_sources_returns_four_sources():
    sources = get_default_price_sources()
    assert len(sources) == 4
    assert isinstance(sources[0], ScryfallPriceSource)
    assert isinstance(sources[1], OpenTCGClient)
    assert isinstance(sources[2], CardmarketClient)
    assert isinstance(sources[3], CardTraderClient)


# -----------------------------------------------------------------------------
# CardTraderClient
# -----------------------------------------------------------------------------


def test_cardtrader_client_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CARDTRADER_API_ENABLED", raising=False)
    monkeypatch.delenv("CARDTRADER_API_TOKEN", raising=False)
    client = CardTraderClient()
    assert client.is_enabled() is False


def test_cardtrader_client_enabled_with_token(monkeypatch):
    monkeypatch.setenv("CARDTRADER_API_ENABLED", "1")
    monkeypatch.setenv("CARDTRADER_API_TOKEN", "secret-token")
    client = CardTraderClient()
    assert client.is_enabled() is True


def test_cardtrader_client_constructor_override():
    client = CardTraderClient(enabled=True, token="tk")
    assert client.is_enabled() is True
    client = CardTraderClient(enabled=False, token="tk")
    assert client.is_enabled() is False


@pytest.mark.asyncio
async def test_cardtrader_client_returns_empty_when_disabled():
    client = CardTraderClient(enabled=False, token="tk")
    result = await client.get_prices_for_card(
        {"id": "abc-123", "set": "usg", "name": "Priest of Titania"}
    )
    assert result == []


@pytest.mark.asyncio
async def test_cardtrader_client_returns_empty_when_no_scryfall_id_or_set():
    client = CardTraderClient(enabled=True, token="tk")
    result = await client.get_prices_for_card({"name": "Card"})
    assert result == []
    result = await client.get_prices_for_card({"id": "abc", "name": "Card"})
    assert result == []
    result = await client.get_prices_for_card({"set": "usg", "name": "Card"})
    assert result == []


def test_cardtrader_client_satisfies_price_source_protocol():
    """CardTraderClient is a valid PriceSource."""
    client = CardTraderClient(enabled=False)
    assert isinstance(client, PriceSource)


@pytest.mark.asyncio
async def test_cardtrader_client_parse_prices_cents_to_units():
    """Marketplace prices are in cents; client converts to currency units."""
    client = CardTraderClient(enabled=True, token="tk")
    products = [
        {"price": {"cents": 150, "currency": "EUR"}},
        {"price": {"cents": 200, "currency": "EUR"}},
        {"price": {"cents": 100, "currency": "EUR"}},
    ]
    result = client._parse_prices(products)
    assert len(result) == 1
    assert result[0].source == "cardtrader"
    assert result[0].currency == "EUR"
    assert result[0].price_low == 1.0
    assert result[0].price_high == 2.0
    assert result[0].price_avg == pytest.approx(1.5)
