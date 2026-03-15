"""FastAPI application for the card extraction service."""
import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.routes import router as v1_router
from .api.schemas import ErrorResponse
from .detection import OpenCVCardDetector
from .extract import ExtractCardNamesService
from .ocr import TesseractCardRecognizer

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(levelname)s: [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Card Extraction Service",
    description="OpenCV + Tesseract card name extraction from MTG images",
    version="0.1.0",
)

app.include_router(v1_router, prefix="/v1", tags=["v1"])


def _parse_worker_threads() -> int | None:
    raw = os.environ.get("WORKER_THREADS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


@app.on_event("startup")
def startup():
    detector = OpenCVCardDetector()
    tess_langs = os.getenv("TESSERACT_LANGS", "").strip() or None
    recognizer = TesseractCardRecognizer(**({"lang": tess_langs} if tess_langs else {}))
    max_workers = _parse_worker_threads()
    app.state.extract_service = ExtractCardNamesService(
        detector, recognizer, max_workers=max_workers
    )
    workers = getattr(app.state.extract_service, "_max_workers", "?")
    logger.info(
        "extract service started (worker_threads=%s, tesseract_langs=%s)",
        workers,
        recognizer.lang,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=str(exc), code="internal_error").model_dump(),
    )
