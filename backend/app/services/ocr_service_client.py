"""Protocol and HTTP client for the Tesseract OCR service."""
from __future__ import annotations

import base64
import logging
import os
from typing import List, Protocol

import httpx

logger = logging.getLogger(__name__)


class OcrClient(Protocol):
    """Abstraction for text recognition from name-crop images."""

    async def recognize(self, images: List[bytes]) -> List[str]: ...


def _default_timeout() -> float:
    raw = os.getenv("OCR_SERVICE_TIMEOUT", "").strip()
    if raw:
        try:
            return max(10.0, float(raw))
        except ValueError:
            pass
    return 120.0


class OcrServiceClient:
    """HTTP client implementing OcrClient via the OCR microservice."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout: float | None = None,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else _default_timeout()

    async def recognize(self, images: List[bytes]) -> List[str]:
        if not images:
            return []

        payload = {
            "images": [base64.b64encode(img).decode("ascii") for img in images],
        }

        logger.info(
            "ocr service: sending %d image(s) to %s/v1/recognize",
            len(images),
            self._base_url,
        )

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/recognize",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, KeyError) as e:
            logger.warning(
                "ocr service: request failed (%s: %s)",
                type(e).__name__,
                e,
            )
            return ["" for _ in images]

        texts: List[str] = []
        for item in data.get("results", []):
            text = item.get("text", "").strip()
            texts.append(text)

        while len(texts) < len(images):
            texts.append("")

        recognized = sum(1 for t in texts if t)
        logger.info(
            "ocr service: got %d result(s), %d recognized",
            len(texts),
            recognized,
        )
        return texts
