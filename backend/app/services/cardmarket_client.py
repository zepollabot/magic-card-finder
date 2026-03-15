"""
Cardmarket.com API client for trend and condition-based pricing.
Requires OAuth 1.0 credentials. Enable with CARDMARKET_API_ENABLED=1.
"""
import asyncio
import logging
import os
from typing import Any, List, Optional

from requests_oauthlib import OAuth1Session

from ..schemas import CardPriceInfo

logger = logging.getLogger(__name__)


CARDMARKET_BASE = "https://apiv2.cardmarket.com/ws/v2.0"


def _is_enabled() -> bool:
    v = os.getenv("CARDMARKET_API_ENABLED", "0").strip().lower()
    return v in ("1", "true", "yes")


def _has_credentials() -> bool:
    return bool(
        os.getenv("CARDMARKET_APP_TOKEN")
        and os.getenv("CARDMARKET_APP_SECRET")
        and os.getenv("CARDMARKET_ACCESS_TOKEN")
        and os.getenv("CARDMARKET_ACCESS_TOKEN_SECRET")
    )


class CardmarketClient:
    """
    Fetches MTG card prices from Cardmarket API (OAuth 1.0).
    Implements PriceSource protocol; enable with CARDMARKET_API_ENABLED=1 and OAuth env vars.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        app_token: Optional[str] = None,
        app_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
    ) -> None:
        self._enabled = enabled if enabled is not None else (_is_enabled() and _has_credentials())
        self._app_token = app_token or os.getenv("CARDMARKET_APP_TOKEN", "")
        self._app_secret = app_secret or os.getenv("CARDMARKET_APP_SECRET", "")
        self._access_token = access_token or os.getenv("CARDMARKET_ACCESS_TOKEN", "")
        self._access_secret = access_secret or os.getenv("CARDMARKET_ACCESS_TOKEN_SECRET", "")

    def is_enabled(self) -> bool:
        return (
            self._enabled
            and bool(self._app_token and self._app_secret and self._access_token and self._access_secret)
        )

    def get_source_name(self) -> str:
        return "cardmarket"

    def get_prices_for_card_sync(self, scryfall_card: dict) -> List[CardPriceInfo]:
        """Synchronous version (Cardmarket uses requests_oauthlib)."""
        if not self.is_enabled():
            return []

        card_name = (scryfall_card.get("name") or "").strip()
        if not card_name:
            return []

        try:
            session = OAuth1Session(
                client_key=self._app_token,
                client_secret=self._app_secret,
                resource_owner_key=self._access_token,
                resource_owner_secret=self._access_secret,
                realm=CARDMARKET_BASE,
            )
            url_find = f"{CARDMARKET_BASE}/products/find"
            r = session.get(
                url_find,
                params={"search": card_name, "idGame": 1, "exact": "true"},
                timeout=15,
            )
            logger.info("cardmarket: GET %s?search=... -> %d", url_find, r.status_code)
            r.raise_for_status()
            data = r.json()

            product_id = self._extract_product_id(data)
            if product_id is None:
                logger.info("cardmarket: no product_id for card=%s", card_name)
                return []

            url_product = f"{CARDMARKET_BASE}/products/{product_id}"
            r2 = session.get(url_product, timeout=15)
            logger.info("cardmarket: GET %s -> %d", url_product, r2.status_code)
            r2.raise_for_status()
            product = r2.json()

            result = self._parse_product_prices(product)
            logger.info("cardmarket: card=%s product_id=%s -> %d price(s)", card_name, product_id, len(result))
            return result
        except Exception as e:
            logger.warning("cardmarket: failed for card=%s: %s", card_name, e, exc_info=True)
            return []

    async def get_prices_for_card(self, scryfall_card: dict) -> List[CardPriceInfo]:
        """Async wrapper (runs sync OAuth in thread pool)."""
        return await asyncio.to_thread(self.get_prices_for_card_sync, scryfall_card)

    def _extract_product_id(self, data: Any) -> Optional[int]:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                return first.get("idProduct") or first.get("id")
        if isinstance(data, dict):
            products = data.get("product") or data.get("products") or data.get("results")
            if isinstance(products, list) and len(products) > 0:
                first = products[0]
                if isinstance(first, dict):
                    return first.get("idProduct") or first.get("id")
            return data.get("idProduct") or data.get("id")
        return None

    def _parse_product_prices(self, product: dict) -> List[CardPriceInfo]:
        out: List[CardPriceInfo] = []
        if not isinstance(product, dict):
            return out

        # Common Cardmarket price fields (vary by response)
        price = product.get("priceGuide") or product.get("price") or product
        if isinstance(price, (int, float)):
            out.append(
                CardPriceInfo(
                    source="cardmarket",
                    currency="EUR",
                    price_low=float(price),
                    price_avg=float(price),
                    price_high=float(price),
                    trend_price=None,
                )
            )
            return out

        if isinstance(price, dict):
            avg = self._to_float(price.get("avg") or price.get("average") or price.get("sell"))
            low = self._to_float(price.get("low") or price.get("min") or price.get("lowPrice"))
            trend = self._to_float(price.get("trend") or price.get("trendPrice"))
            val = avg or low
            if val is not None:
                out.append(
                    CardPriceInfo(
                        source="cardmarket",
                        currency="EUR",
                        price_low=low or val,
                        price_avg=val,
                        price_high=self._to_float(price.get("high") or price.get("max")) or val,
                        trend_price=trend,
                    )
                )
        return out

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
