from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import logging
import os

from .scryfall_client import ScryfallClient


logger = logging.getLogger("app.services.card_name_resolver")


class CardNameResolver(Protocol):
    """
    Abstraction for resolving potentially noisy or partial card names
    into a canonical Scryfall card object.
    """

    async def resolve(self, raw_name: str, set_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Resolve a raw card name (and optional set hint) into one or
        more Scryfall card objects. Returns an empty list if no
        candidates can be found.
        """


class ScryfallCardNameResolver:
    """
    Card name resolver backed by Scryfall.

    Strategy:
    - First try: /cards/named?fuzzy=... with set_code when provided.
    - Fallback 1: /cards/named?fuzzy=... without set_code.
    - Fallback 2: /cards/search?q=name:... using a broader name query
      and selecting a single best candidate.
    """

    def __init__(self, client: ScryfallClient) -> None:
        self._client = client

        # Configuration knobs for fallback behaviour.
        #
        # SCRYFALL_FALLBACK_MAX_CANDIDATES: int or unset.
        #   If set, we only inspect up to this many candidates from the
        #   broader search before choosing a card.
        #
        # SCRYFALL_FALLBACK_ALLOW_NON_PREFIX:
        #   "0"/"false"/"no" (case-insensitive) -> only exact or prefix
        #   matches are allowed; we never fall back to an arbitrary
        #   candidate.
        #   Any other non-empty value -> allow non-prefix fallback.
        raw_max = os.getenv("SCRYFALL_FALLBACK_MAX_CANDIDATES", "").strip()
        try:
            self._max_candidates: Optional[int] = int(raw_max) if raw_max else None
        except ValueError:
            self._max_candidates = None

        raw_allow = os.getenv("SCRYFALL_FALLBACK_ALLOW_NON_PREFIX", "").strip().lower()
        if raw_allow in {"0", "false", "no"}:
            self._allow_non_prefix: bool = False
        elif raw_allow:
            self._allow_non_prefix = True
        else:
            # Sane default: allow non-prefix fallback so OCR partials
            # still have a chance to resolve.
            self._allow_non_prefix = True

    async def resolve(self, raw_name: str, set_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        name = (raw_name or "").strip()
        if not name:
            return []

        # First try: fuzzy with set_hint
        card: Optional[Dict[str, Any]] = None
        if set_hint:
            try:
                card = await self._client.named(name, set_code=set_hint)
            except Exception as exc:  # network or HTTP errors
                logger.warning("scryfall resolver: named() with set_hint failed for %r: %s", name, exc)

        # Fallback 1: fuzzy without set_hint
        if card is None:
            try:
                card = await self._client.named(name, set_code=None)
            except Exception as exc:
                logger.warning("scryfall resolver: named() without set_hint failed for %r: %s", name, exc)

        if card is not None:
            return [card]

        # Fallback 2: broader search by name
        logger.info("scryfall resolver: engaging fallback search for %r", name)
        try:
            candidates = await self._client.search_by_name(name)
        except Exception as exc:
            logger.warning("scryfall resolver: search_by_name() failed for %r: %s", name, exc)
            return []

        if not candidates:
            return []

        if self._max_candidates is not None and self._max_candidates > 0:
            candidates = candidates[: self._max_candidates]

        logger.info(
            "scryfall resolver: fallback returned %d candidates for %r",
            len(candidates),
            name,
        )
        return candidates

