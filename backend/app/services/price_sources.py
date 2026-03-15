"""
Pricing abstraction and built-in sources.

- PriceSource: protocol for any pricing backend (Single Responsibility, Dependency Inversion).
- New sources implement the protocol and are registered in get_default_price_sources()
  (Open/Closed: extend by adding, not by modifying existing code).
"""
import logging
from typing import List, Optional, Protocol, runtime_checkable

from ..schemas import CardPriceInfo

logger = logging.getLogger(__name__)


@runtime_checkable
class PriceSource(Protocol):
    """
    Abstraction for a single price data source.
    Implement this protocol to add new pricing backends without changing the aggregator.
    """

    def is_enabled(self) -> bool:
        """Return True if this source should be queried (e.g. env var set, credentials present)."""
        ...

    def get_source_name(self) -> str:
        """Return the source identifier used in CardPriceInfo.source (e.g. 'scryfall', 'cardtrader')."""
        ...

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        """Fetch prices for the given canonical Scryfall card. Return empty list on failure or skip."""
        ...


class ScryfallPriceSource:
    """
    Extracts prices from the Scryfall card object (no HTTP).
    Always enabled; used as the primary source from the card we already have.
    """

    def is_enabled(self) -> bool:
        return True

    def get_source_name(self) -> str:
        return "scryfall"

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        prices: List[CardPriceInfo] = []
        raw = scryfall_card.get("prices") or {}
        usd = self._to_float(raw.get("usd"))
        eur = self._to_float(raw.get("eur"))
        card_name = (scryfall_card.get("name") or "").strip() or "(no name)"
        if usd is not None:
            prices.append(
                CardPriceInfo(
                    source="scryfall",
                    currency="USD",
                    price_low=usd,
                    price_avg=usd,
                    price_high=usd,
                    trend_price=None,
                )
            )
        if eur is not None:
            prices.append(
                CardPriceInfo(
                    source="scryfall",
                    currency="EUR",
                    price_low=eur,
                    price_avg=eur,
                    price_high=eur,
                    trend_price=None,
                )
            )
        logger.debug("scryfall: card=%s embedded prices -> %d (usd=%s eur=%s)", card_name, len(prices), usd, eur)
        return prices

    @staticmethod
    def _to_float(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def get_default_price_sources() -> List[PriceSource]:
    """
    Returns the default list of price sources in order of use.
    To add a new source: implement PriceSource and append it here.
    """
    from .open_tcg_client import OpenTCGClient
    from .cardmarket_client import CardmarketClient
    from .cardtrader_client import CardTraderClient

    return [
        ScryfallPriceSource(),
        OpenTCGClient(),
        CardmarketClient(),
        CardTraderClient(),
    ]


def get_enabled_price_source_names() -> List[str]:
    """Return source names for all enabled price sources (for report column headers)."""
    return [s.get_source_name() for s in get_default_price_sources() if s.is_enabled()]
