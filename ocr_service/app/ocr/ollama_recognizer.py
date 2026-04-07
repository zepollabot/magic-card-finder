"""Ollama vision LLM text recognizer."""
import base64
import logging

import httpx

logger = logging.getLogger(__name__)

_PROMPT = (
    "I want you to extract the text from this image. "
    "Give me only the text as response"
)


class OllamaTextRecognizer:
    """Recognizes card name text by sending the image to a local Ollama vision model."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    async def recognize(self, image: bytes) -> str:
        """Send *image* (raw file bytes) to Ollama and return the extracted text."""
        image_b64 = base64.b64encode(image).decode("ascii")

        payload = {
            "model": self._model,
            "prompt": _PROMPT,
            "images": [image_b64],
            "stream": False,
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "ollama request failed (%s: %s)", type(exc).__name__, exc
            )
            return ""

        text = data.get("response", "").strip()
        logger.debug("ollama recognized: %r", text)
        return text
