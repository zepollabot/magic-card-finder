"""Protocol and HTTP client for card name extraction from images (e.g. extraction service)."""
from __future__ import annotations

import base64
import logging
import os
from typing import List, Protocol

import httpx

logger = logging.getLogger(__name__)


class CardNameExtractor(Protocol):
    """Extracts card names from raw image bytes via an external service."""

    async def extract_names_from_images(self, images: List[bytes]) -> List[List[str]]:
        """
        Returns one list of card names per input image (same order as input).
        Each inner list is the card names detected in that image.
        """
        ...


def _default_extraction_timeout() -> float:
    """Default timeout in seconds for extraction service (many cards = many OCR runs)."""
    raw = os.getenv("EXTRACTION_SERVICE_TIMEOUT", "").strip()
    if raw:
        try:
            return max(30.0, float(raw))
        except ValueError:
            pass
    return 300.0  # 5 minutes for images with many cards


class ExtractionServiceClient:
    """HTTP client for the card extraction service (OpenCV + Tesseract)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._client = client
        self._base_url = (base_url or os.getenv("EXTRACTION_SERVICE_URL", "")).rstrip("/")
        self._timeout = timeout if timeout is not None else _default_extraction_timeout()

    async def extract_names_from_images(self, images: List[bytes]) -> List[List[str]]:
        if not self._base_url:
            return [[] for _ in images]
        if not images:
            return []
        logger.info("extraction service: sending %d image(s) to %s/v1/extract/cards", len(images), self._base_url)
        payload = {
            "images": [base64.b64encode(img).decode("ascii") for img in images],
        }
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/extract/cards",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, KeyError) as e:
            logger.warning(
                "extraction service: request failed (%s: %s), returning empty names per image",
                type(e).__name__,
                e,
            )
            return [[] for _ in images]

        results = data.get("results", [])
        # Preserve order by image_index; fill gaps with empty list
        max_idx = max((r.get("image_index", 0) for r in results), default=-1)
        out: List[List[str]] = [[] for _ in range(max_idx + 1)]
        for r in results:
            idx = r.get("image_index", 0)
            names = r.get("card_names", [])
            if isinstance(names, list):
                out[idx] = [str(n) for n in names]
            else:
                out[idx] = []
        # If we have more images than results (e.g. some failed), pad
        while len(out) < len(images):
            out.append([])
        result = out[: len(images)]
        total_names = sum(len(n) for n in result)
        logger.info(
            "extraction service: got %d result(s), total card names: %d (per image: %s)",
            len(result),
            total_names,
            [len(n) for n in result],
        )
        return result
