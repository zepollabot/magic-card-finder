from typing import Any, Dict, List, Optional

import httpx


class ScryfallClient:
    """
    Scryfall API client: resolve card names (fuzzy) and perform searches.
    """

    BASE_URL = "https://api.scryfall.com"

    async def named(self, fuzzy_name: str, set_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Resolve a card by fuzzy name (and optional set).
        Returns one printing or None if not found (404).
        """
        params = {"fuzzy": fuzzy_name}
        if set_code:
            params["set"] = set_code
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.BASE_URL}/cards/named", params=params)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    async def search_printings(self, exact_name: str) -> List[Dict[str, Any]]:
        """
        Return all printings of a card by exact name (unique=prints).

        Uses the Scryfall search syntax:
        - !"Name" = exact card name
        - unique=prints = all distinct printings
        """
        query = f'!"{exact_name}"'
        all_cards: List[Dict[str, Any]] = []
        page = 1
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                resp = await client.get(
                    f"{self.BASE_URL}/cards/search",
                    params={"q": query, "unique": "prints", "page": page},
                )
                resp.raise_for_status()
                data = resp.json()
                cards = data.get("data") or []
                all_cards.extend(cards)
                if not data.get("has_more", False):
                    break
                page += 1
        return all_cards

    async def search_by_name(self, raw_name: str) -> List[Dict[str, Any]]:
        """
        Broader search for cards matching the given name fragment.

        This is a thin wrapper over /cards/search. Selection of the
        \"best\" card is left to higher-level services.
        """
        query_name = (raw_name or "").strip()
        if not query_name:
            return []

        # name:\"Foo Bar\" performs a phrase search on the card name, which works
        # reasonably well for slightly noisy OCR inputs.
        query = f'name:\"{query_name}\"'

        all_cards: List[Dict[str, Any]] = []
        page = 1
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                resp = await client.get(
                    f"{self.BASE_URL}/cards/search",
                    params={"q": query, "page": page},
                )
                resp.raise_for_status()
                data = resp.json()
                cards = data.get("data") or []
                all_cards.extend(cards)
                if not data.get("has_more", False):
                    break
                page += 1
        return all_cards

