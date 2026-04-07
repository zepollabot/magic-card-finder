"""OCR module: name-crop image -> card name string via Ollama vision LLM."""
from .protocols import TextRecognizer
from .ollama_recognizer import OllamaTextRecognizer

__all__ = [
    "TextRecognizer",
    "OllamaTextRecognizer",
]
