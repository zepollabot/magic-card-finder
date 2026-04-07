"""Protocols for text recognition (dependency inversion)."""
from typing import Protocol


class TextRecognizer(Protocol):
    """Recognizes card name text from a raw image."""

    async def recognize(self, image: bytes) -> str:
        """Return recognized text, or empty string on failure."""
        ...
