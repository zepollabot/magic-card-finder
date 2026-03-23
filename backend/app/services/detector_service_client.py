"""Protocol and HTTP client for the YOLO detector service."""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import List, Protocol

import httpx

logger = logging.getLogger(__name__)


@dataclass
class NameCrop:
    """A single detected name region."""

    bbox: tuple[int, int, int, int]
    confidence: float
    image_bytes: bytes


@dataclass
class DetectionResult:
    """Detection results for a single input image."""

    image_index: int
    name_crops: List[NameCrop] = field(default_factory=list)


class DetectorClient(Protocol):
    """Abstraction for card name detection from raw images."""

    async def detect(self, images: List[bytes]) -> List[DetectionResult]: ...


def _default_timeout() -> float:
    raw = os.getenv("DETECTOR_SERVICE_TIMEOUT", "").strip()
    if raw:
        try:
            return max(10.0, float(raw))
        except ValueError:
            pass
    return 120.0


class DetectorServiceClient:
    """HTTP client implementing DetectorClient via the detector microservice."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout: float | None = None,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else _default_timeout()

    async def detect(self, images: List[bytes]) -> List[DetectionResult]:
        if not images:
            return []

        payload = {
            "images": [base64.b64encode(img).decode("ascii") for img in images],
        }

        logger.info(
            "detector service: sending %d image(s) to %s/v1/detect",
            len(images),
            self._base_url,
        )

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/detect",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, KeyError) as e:
            logger.warning(
                "detector service: request failed (%s: %s)",
                type(e).__name__,
                e,
            )
            return [DetectionResult(image_index=i) for i in range(len(images))]

        results: List[DetectionResult] = []
        for item in data.get("results", []):
            image_index = item.get("image_index", 0)
            crops = []
            for det in item.get("detections", []):
                try:
                    crop_bytes = base64.b64decode(det.get("crop_b64", ""))
                    bbox_list = det.get("bbox", [0, 0, 0, 0])
                    crops.append(
                        NameCrop(
                            bbox=tuple(bbox_list),
                            confidence=det.get("confidence", 0.0),
                            image_bytes=crop_bytes,
                        )
                    )
                except Exception:
                    continue
            results.append(DetectionResult(image_index=image_index, name_crops=crops))

        while len(results) < len(images):
            results.append(DetectionResult(image_index=len(results)))

        total_crops = sum(len(r.name_crops) for r in results)
        logger.info(
            "detector service: got %d result(s), %d name crop(s) total",
            len(results),
            total_crops,
        )
        return results
