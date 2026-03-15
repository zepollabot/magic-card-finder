"""
CardTrader API client for marketplace pricing.
Requires Bearer token. Enable with CARDTRADER_API_ENABLED=1 and CARDTRADER_API_TOKEN.
See: https://www.cardtrader.com/it/docs/api/full/reference
"""
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import CardPriceInfo

logger = logging.getLogger(__name__)


CARDTRADER_BASE = "https://api.cardtrader.com/api/v2"


def _is_enabled() -> bool:
    v = os.getenv("CARDTRADER_API_ENABLED", "0").strip().lower()
    return v in ("1", "true", "yes")


def _get_token() -> str:
    return (os.getenv("CARDTRADER_API_TOKEN") or "").strip()


class CardTraderClient:
    """
    Fetches MTG card prices from CardTrader API (Bearer token).
    Resolves card via Scryfall ID → expansion → blueprint → marketplace products.
    Implements PriceSource protocol; enable with CARDTRADER_API_ENABLED=1 and CARDTRADER_API_TOKEN.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._enabled = enabled if enabled is not None else _is_enabled()
        self._token = (token or _get_token()).strip()
        self._base_url = (base_url or os.getenv("CARDTRADER_API_BASE_URL", CARDTRADER_BASE)).rstrip("/")
        self._expansion_by_code: Dict[str, Dict[str, Any]] = {}

    def is_enabled(self) -> bool:
        return self._enabled and bool(self._token)

    def get_source_name(self) -> str:
        return "cardtrader"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        if not self.is_enabled():
            logger.debug("cardtrader: disabled, skipping")
            return []

        scryfall_id = (scryfall_card.get("id") or "").strip()
        set_code = (scryfall_card.get("set") or "").strip()
        set_name = (scryfall_card.get("set_name") or "").strip()
        card_name = (scryfall_card.get("name") or "").strip() or "(no name)"
        if not scryfall_id or not set_code:
            logger.info(
                "cardtrader: skipping card=%s (missing id or set; id=%s set=%s)",
                card_name,
                "yes" if scryfall_id else "no",
                set_code or "(empty)",
            )
            return []

        logger.info(
            "cardtrader: resolving card=%s set_code=%s scryfall_id=%s",
            card_name,
            set_code,
            scryfall_id[:8] + "..." if len(scryfall_id) > 8 else scryfall_id,
        )
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                expansion_id = await self._resolve_expansion_id(client, set_code, set_name=set_name)
                if expansion_id is None:
                    logger.warning(
                        "cardtrader: no expansion found for set_code=%s set_name=%s",
                        set_code,
                        set_name or "(empty)",
                    )
                    return []

                blueprint_id = await self._find_blueprint_by_scryfall_id(
                    client, expansion_id, scryfall_id
                )
                if blueprint_id is None:
                    logger.warning(
                        "cardtrader: no blueprint found for expansion_id=%s scryfall_id=%s",
                        expansion_id,
                        scryfall_id[:8] + "...",
                    )
                    return []

                products = await self._get_marketplace_products(client, blueprint_id)
                result = self._parse_prices(products)
                logger.info(
                    "cardtrader: card=%s blueprint_id=%s products=%d -> %d price(s)",
                    card_name,
                    blueprint_id,
                    len(products),
                    len(result),
                )
                return result
        except Exception as e:
            logger.warning(
                "cardtrader: failed for card=%s set_code=%s: %s",
                card_name,
                set_code,
                e,
                exc_info=True,
            )
            return []

    async def _resolve_expansion_id(
        self,
        client: httpx.AsyncClient,
        set_code: str,
        set_name: Optional[str] = None,
    ) -> Optional[int]:
        """Get CardTrader expansion id for a set code (e.g. usg). Uses in-memory cache.

        Fallback: when code lookup fails, try matching by expansion name against the cached list.
        """
        set_code_lower = set_code.lower()
        if set_code_lower in self._expansion_by_code:
            exp = self._expansion_by_code[set_code_lower]
            if isinstance(exp, dict) and "id" in exp:
                logger.debug("cardtrader: expansion cache hit set_code=%s -> id=%s", set_code, exp["id"])
                return int(exp["id"])
            return None

        url = f"{self._base_url}/expansions"
        try:
            r = await client.get(url, headers=self._headers())
            logger.info("cardtrader: GET %s -> %d", url, r.status_code)
            r.raise_for_status()
            expansions = r.json()
            if not isinstance(expansions, list):
                logger.warning("cardtrader: expansions response not a list (type=%s)", type(expansions).__name__)
                return None
            for exp in expansions:
                if isinstance(exp, dict):
                    code = (exp.get("code") or "").strip().lower()
                    if code:
                        self._expansion_by_code[code] = exp

            # 1) Try by code
            exp = self._expansion_by_code.get(set_code_lower)
            if exp and "id" in exp:
                logger.info("cardtrader: set_code=%s -> expansion_id=%s", set_code, exp["id"])
                return int(exp["id"])

            # 2) Fallback: try by expansion name (case-insensitive exact match first, then contains)
            name = (set_name or "").strip().lower()
            if name:
                exact_match = None
                partial_match = None
                for e in expansions:
                    if not isinstance(e, dict):
                        continue
                    n = (e.get("name") or "").strip().lower()
                    if not n:
                        continue
                    if n == name:
                        exact_match = e
                        break
                    if name in n and partial_match is None:
                        partial_match = e
                chosen = exact_match or partial_match
                if chosen and "id" in chosen:
                    logger.info(
                        "cardtrader: fallback by name '%s' -> expansion_id=%s (code=%s)",
                        set_name,
                        chosen["id"],
                        chosen.get("code"),
                    )
                    return int(chosen["id"])

            logger.warning(
                "cardtrader: set_code=%s set_name=%s not in expansions (cached %d codes)",
                set_code,
                set_name or "(empty)",
                len(self._expansion_by_code),
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.warning("cardtrader: GET %s -> %d %s", url, e.response.status_code, e.response.text[:200])
            return None
        except Exception as e:
            logger.warning("cardtrader: GET %s failed: %s", url, e, exc_info=True)
            return None

    async def _find_blueprint_by_scryfall_id(
        self,
        client: httpx.AsyncClient,
        expansion_id: int,
        scryfall_id: str,
    ) -> Optional[int]:
        url = f"{self._base_url}/blueprints/export"
        params = {"expansion_id": expansion_id}
        try:
            r = await client.get(url, params=params, headers=self._headers())
            logger.info("cardtrader: GET %s?expansion_id=%s -> %d", url, expansion_id, r.status_code)
            r.raise_for_status()
            blueprints = r.json()
            if not isinstance(blueprints, list):
                logger.warning("cardtrader: blueprints response not a list (type=%s)", type(blueprints).__name__)
                return None
            sid_lower = scryfall_id.lower()
            for bp in blueprints:
                if isinstance(bp, dict):
                    bp_sid = (bp.get("scryfall_id") or "").strip().lower()
                    if bp_sid == sid_lower:
                        bid = bp.get("id")
                        if bid is not None:
                            logger.info("cardtrader: blueprint_id=%s for scryfall_id=%s", bid, sid_lower[:8] + "...")
                            return int(bid)
            logger.warning(
                "cardtrader: scryfall_id not found in %d blueprints for expansion_id=%s",
                len(blueprints),
                expansion_id,
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(
                "cardtrader: GET %s?expansion_id=%s -> %d %s",
                url,
                expansion_id,
                e.response.status_code,
                e.response.text[:200],
            )
            return None
        except Exception as e:
            logger.warning("cardtrader: GET blueprints/export failed: %s", e, exc_info=True)
            return None

    async def _get_marketplace_products(
        self,
        client: httpx.AsyncClient,
        blueprint_id: int,
    ) -> List[Dict[str, Any]]:
        """GET /marketplace/products?blueprint_id=X. Returns list of product objects."""
        url = f"{self._base_url}/marketplace/products"
        params = {"blueprint_id": blueprint_id}
        try:
            r = await client.get(url, params=params, headers=self._headers())
            logger.info("cardtrader: GET %s?blueprint_id=%s -> %d", url, blueprint_id, r.status_code)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                logger.warning("cardtrader: marketplace response not a dict (type=%s)", type(data).__name__)
                return []
            key = str(blueprint_id)
            products = data.get(key)
            if isinstance(products, list):
                logger.debug("cardtrader: marketplace products count=%d for blueprint_id=%s", len(products), blueprint_id)
                return products
            logger.warning(
                "cardtrader: no key %s in marketplace response (keys=%s)",
                key,
                list(data.keys())[:10],
            )
            return []
        except httpx.HTTPStatusError as e:
            logger.warning(
                "cardtrader: GET %s?blueprint_id=%s -> %d %s",
                url,
                blueprint_id,
                e.response.status_code,
                e.response.text[:200],
            )
            return []
        except Exception as e:
            logger.warning("cardtrader: GET marketplace/products failed: %s", e, exc_info=True)
            return []

    def _parse_prices(self, products: List[Dict[str, Any]]) -> List[CardPriceInfo]:
        """
        Convert marketplace products (price in cents) to CardPriceInfo list.

        - Filters by language when CARDTRADER_LANGUAGE is set (default: 'en').
        - Uses a trimmed mean to avoid single erroneous listings (e.g. 10k EUR) skewing the average:
          drops the top and bottom 10% of prices when there are enough data points.
        """
        out: List[CardPriceInfo] = []
        prices_units: List[float] = []
        currency: Optional[str] = None

        lang_filter = (os.getenv("CARDTRADER_LANGUAGE", "en") or "").strip().lower()

        for p in products:
            if not isinstance(p, dict):
                continue

            # Optional language filtering: keep only products matching mtg_language (if present)
            if lang_filter:
                props = p.get("properties_hash") or p.get("properties") or {}
                prod_lang = (props.get("mtg_language") or "").strip().lower()
                if prod_lang and prod_lang != lang_filter:
                    continue

            price_obj = p.get("price")
            if not isinstance(price_obj, dict):
                continue
            cents = price_obj.get("cents")
            if cents is not None:
                try:
                    prices_units.append(float(cents) / 100.0)
                except (TypeError, ValueError):
                    pass
            if currency is None and isinstance(price_obj.get("currency"), str):
                currency = (price_obj.get("currency") or "").strip() or None

        if not prices_units:
            logger.info("cardtrader: no prices after filtering (lang=%s)", lang_filter or "any")
            return out

        cur = currency or "EUR"

        # Robust stats: trimmed mean to avoid extreme outliers
        vals = sorted(prices_units)
        n = len(vals)
        trim_ratio = 0.1
        k = int(n * trim_ratio)
        if n - 2 * k > 0:
            core = vals[k : n - k]
        else:
            core = vals

        low = core[0]
        high = core[-1]
        avg = sum(core) / len(core)

        logger.debug(
            "cardtrader: %d raw prices (units), %d after trim -> low=%.2f avg=%.2f high=%.2f (currency=%s, lang=%s)",
            n,
            len(core),
            low,
            avg,
            high,
            cur,
            lang_filter or "any",
        )

        out.append(
            CardPriceInfo(
                source="cardtrader",
                currency=cur,
                price_low=low,
                price_avg=avg,
                price_high=high,
                trend_price=None,
            )
        )
        return out
