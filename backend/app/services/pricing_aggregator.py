"""
Aggregates prices from multiple sources via the PriceSource protocol.
Depends on abstractions (Dependency Inversion); new sources are added by
implementing PriceSource and registering in get_default_price_sources().
"""
import asyncio
import logging
from typing import List, Optional, Sequence

from ..schemas import CardPriceInfo

from .price_sources import PriceSource

logger = logging.getLogger(__name__)


class PricingAggregator:
    """
    Orchestrates multiple price sources and merges results.
    Single responsibility: aggregate only. Open/Closed: new sources
    are added by extending the sources list, not by editing this class.
    """

    def __init__(self, sources: Optional[Sequence[PriceSource]] = None) -> None:
        if sources is not None:
            self._sources = list(sources)
        else:
            from .price_sources import get_default_price_sources

            self._sources = get_default_price_sources()

    def get_enabled_source_names(self) -> List[str]:
        """Return source names for all enabled price sources (for report column headers)."""
        return [s.get_source_name() for s in self._sources if s.is_enabled()]

    async def _fetch_from_source(self, source: PriceSource, card_name: str, scryfall_card: dict) -> List[CardPriceInfo]:
        name = source.get_source_name()
        if not source.is_enabled():
            logger.info("pricing: source=%s disabled, skipping card=%s", name, card_name)
            return []
        try:
            logger.info("pricing: querying source=%s for card=%s", name, card_name)
            result = await source.get_prices_for_card(scryfall_card)
            logger.info(
                "pricing: source=%s returned %d price(s) for card=%s",
                name,
                len(result),
                card_name,
            )
            return result
        except Exception as e:
            logger.warning(
                "pricing: source=%s failed for card=%s: %s",
                name,
                card_name,
                e,
                exc_info=True,
            )
            return []

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        card_name = (scryfall_card.get("name") or "").strip() or "(no name)"
        if not self._sources:
            return []

        tasks = [self._fetch_from_source(source, card_name, scryfall_card) for source in self._sources]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        prices: List[CardPriceInfo] = []
        for result in results:
            prices.extend(result)
        return prices

