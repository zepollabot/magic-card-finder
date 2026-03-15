"""
Open TCG API (tcgtracking.com) client for additional pricing.
No API key required. Enable with OPEN_TCG_API_ENABLED=1.
"""
import logging
import os
from typing import List, Optional

import httpx

from ..schemas import CardPriceInfo

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    v = os.getenv("OPEN_TCG_API_ENABLED", "0").strip().lower()
    return v in ("1", "true", "yes")


def _base_url() -> str:
    return os.getenv(
        "OPEN_TCG_API_BASE_URL",
        "https://tcgtracking.com/tcgapi/v1",
    ).rstrip("/")


# Magic: The Gathering category id in Open TCG API
MTG_CAT = 1


class OpenTCGClient:
    """
    Fetches MTG card prices from Open TCG API (tcgtracking.com).
    Implements PriceSource protocol; enable with OPEN_TCG_API_ENABLED=1.
    """

    def __init__(self, enabled: Optional[bool] = None, base_url: Optional[str] = None) -> None:
        self._enabled = enabled if enabled is not None else _is_enabled()
        self._base_url = base_url or _base_url()

    def is_enabled(self) -> bool:
        return self._enabled

    def get_source_name(self) -> str:
        return "opentcg"

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        if not self._enabled:
            return []

        card_name = (scryfall_card.get("name") or "").strip()
        set_name = (scryfall_card.get("set_name") or "").strip()
        scryfall_id = (scryfall_card.get("id") or "").strip()

        if not card_name:
            logger.debug("opentcg: no card name, skipping")
            return []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                set_id = await self._find_set_id(client, set_name or card_name, preferred_set_name=set_name)
                if set_id is None:
                    logger.info("opentcg: no set_id for card=%s query=%s", card_name, set_name or card_name)
                    return []

                products = await self._get_set_products(client, set_id)
                if not products:
                    logger.info("opentcg: no products for set_id=%s card=%s", set_id, card_name)
                    return []

                product = self._find_product(products, scryfall_id=scryfall_id, card_name=card_name)
                if product is None:
                    logger.info("opentcg: product not found in set for card=%s", card_name)
                    return []

                product_id = product.get("id")
                if product_id is None:
                    return []

                pricing = await self._get_set_pricing(client, set_id)
                if not pricing:
                    logger.info("opentcg: no pricing for set_id=%s card=%s", set_id, card_name)
                    return []

                result = self._parse_prices(product_id, pricing)
                logger.info("opentcg: card=%s set_id=%s product_id=%s -> %d price(s)", card_name, set_id, product_id, len(result))
                return result
        except Exception as e:
            logger.warning("opentcg: failed for card=%s: %s", card_name, e, exc_info=True)
            return []

    async def _find_set_id(
        self,
        client: httpx.AsyncClient,
        query: str,
        preferred_set_name: Optional[str] = None,
    ) -> Optional[int]:
        if not query:
            return None
        try:
            url = f"{self._base_url}/{MTG_CAT}/search"
            r = await client.get(url, params={"q": query[:80]})
            logger.info("opentcg: GET %s?q=... -> %d", url, r.status_code)
            r.raise_for_status()
            data = r.json()
            sets = data.get("sets") or []
            if not sets:
                return None
            preferred = (preferred_set_name or "").strip().lower()
            for s in sets:
                if isinstance(s, dict) and "id" in s:
                    if preferred:
                        name = (s.get("name") or "").strip().lower()
                        abbr = (s.get("abbreviation") or "").strip().lower()
                        if name == preferred or abbr == preferred:
                            return int(s["id"])
                    else:
                        return int(s["id"])
            # Fallback: first set (e.g. when preferred name had no match)
            if sets and isinstance(sets[0], dict) and "id" in sets[0]:
                return int(sets[0]["id"])
            return None
        except Exception:
            return None

    async def _get_set_products(self, client: httpx.AsyncClient, set_id: int) -> List[dict]:
        try:
            url = f"{self._base_url}/{MTG_CAT}/sets/{set_id}"
            r = await client.get(url)
            logger.info("opentcg: GET %s -> %d", url, r.status_code)
            r.raise_for_status()
            data = r.json()
            products = data.get("products") or []
            logger.debug("opentcg: set_id=%s products count=%d", set_id, len(products))
            return products
        except Exception as e:
            logger.warning("opentcg: GET sets/%s failed: %s", set_id, e)
            return []

    def _find_product(
        self,
        products: List[dict],
        *,
        scryfall_id: str,
        card_name: str,
    ) -> Optional[dict]:
        card_name_lower = card_name.lower()
        for p in products:
            if scryfall_id and (p.get("scryfall_id") or "").strip().lower() == scryfall_id.lower():
                return p
            if (p.get("name") or "").strip().lower() == card_name_lower:
                return p
            if (p.get("clean_name") or "").strip().lower() == card_name_lower:
                return p
        return None

    async def _get_set_pricing(self, client: httpx.AsyncClient, set_id: int) -> dict:
        try:
            url = f"{self._base_url}/{MTG_CAT}/sets/{set_id}/pricing"
            r = await client.get(url)
            logger.info("opentcg: GET %s -> %d", url, r.status_code)
            r.raise_for_status()
            data = r.json()
            return data.get("prices") or {}
        except Exception as e:
            logger.warning("opentcg: GET sets/%s/pricing failed: %s", set_id, e)
            return {}

    def _parse_prices(self, product_id: int, pricing: dict) -> List[CardPriceInfo]:
        out: List[CardPriceInfo] = []
        by_id = pricing.get("prices") if isinstance(pricing.get("prices"), dict) else pricing
        if not by_id:
            return out

        # Key can be str or int
        block = by_id.get(str(product_id)) or by_id.get(product_id)
        if not block or not isinstance(block, dict):
            return out

        tcg = block.get("tcg") or {}
        if isinstance(tcg, dict):
            for _subtype, vals in tcg.items():
                if isinstance(vals, dict):
                    market = vals.get("market")
                    low = vals.get("low")
                    if market is not None or low is not None:
                        val = self._to_float(market) or self._to_float(low)
                        if val is not None:
                            out.append(
                                CardPriceInfo(
                                    source="opentcg",
                                    currency="USD",
                                    price_low=val,
                                    price_avg=val,
                                    price_high=val,
                                    trend_price=None,
                                )
                            )
                            break

        manapool = block.get("manapool") or {}
        if isinstance(manapool, dict) and not out:
            for _finish, val in manapool.items():
                f = self._to_float(val)
                if f is not None:
                    out.append(
                        CardPriceInfo(
                            source="opentcg",
                            currency="USD",
                            price_low=f,
                            price_avg=f,
                            price_high=f,
                            trend_price=None,
                        )
                    )
                    break

        return out

    @staticmethod
    def _to_float(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
