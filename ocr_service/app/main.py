"""FastAPI application for the Ollama-based OCR service."""
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.routes import router as v1_router
from .api.schemas import ErrorResponse
from .ocr import OllamaTextRecognizer

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(levelname)s: [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").strip()
    model = os.getenv("OLLAMA_MODEL", "llava").strip()

    http_client = httpx.AsyncClient()
    app.state._http_client = http_client

    recognizer = OllamaTextRecognizer(
        client=http_client,
        base_url=ollama_url,
        model=model,
    )
    app.state.recognizer = recognizer

    logger.info(
        "ocr service started (ollama_url=%s, model=%s)",
        ollama_url,
        model,
    )

    yield

    await http_client.aclose()


app = FastAPI(
    title="MTG Card OCR Service",
    description="Ollama vision LLM card name recognition from name-crop images",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(v1_router, prefix="/v1", tags=["v1"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )
