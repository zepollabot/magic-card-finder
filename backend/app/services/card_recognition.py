from typing import List
import os
import base64

import httpx


class CardRecognitionResult:
    def __init__(
        self,
        card_name: str | None,
        set_name: str | None = None,
        collector_number: str | None = None,
    ):
        self.card_name = card_name
        self.set_name = set_name
        self.collector_number = collector_number


class CardRecognitionService:
    """
    Wrapper for Llama 3.2 Vision via Ollama or alternative MTG-specific models.
    """

    def __init__(self, model_name: str | None = None, ollama_host: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama-3.2-vision")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

    async def recognize_cards(self, card_images: List[bytes]) -> List[CardRecognitionResult]:
        results: List[CardRecognitionResult] = []
        async with httpx.AsyncClient(base_url=self.ollama_host, timeout=60.0) as client:
            for img in card_images:
                b64_img = base64.b64encode(img).decode("ascii")
                prompt = (
                    "You see a Magic: The Gathering trading card. "
                    "Respond with strictly JSON using keys card_name (string or null), "
                    "set_name (string or null), and collector_number (string or null). "
                    "Do not include any explanation or text outside the JSON."
                )
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [b64_img],
                    "stream": False,
                }
                try:
                    resp = await client.post("/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data.get("response", "").strip()
                    parsed = self._safe_parse_json(raw_text)
                    results.append(
                        CardRecognitionResult(
                            card_name=parsed.get("card_name"),
                            set_name=parsed.get("set_name"),
                            collector_number=parsed.get("collector_number"),
                        )
                    )
                except Exception:
                    results.append(CardRecognitionResult(card_name=None))
        return results

    @staticmethod
    def _safe_parse_json(text: str) -> dict:
        """
        Best-effort JSON extraction: tries to locate the first JSON object in the text.
        """
        import json

        text = text.strip()
        if not text:
            return {}
        if text[0] == "{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


