"""FastAPI application for the Tesseract OCR service."""
import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.routes import router as v1_router
from .api.schemas import ErrorResponse
from .ocr import NameCropPreprocessor, TesseractCardRecognizer

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(levelname)s: [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MTG Card OCR Service",
    description="Tesseract-based card name recognition from name-crop images",
    version="0.1.0",
)

app.include_router(v1_router, prefix="/v1", tags=["v1"])


@app.on_event("startup")
def startup():
    preprocessor = NameCropPreprocessor()
    lang = os.getenv("TESSERACT_LANGS", "").strip() or None
    debug_dir = os.getenv("OCR_DEBUG_DIR", "").strip()

    recognizer = TesseractCardRecognizer(
        preprocessor=preprocessor,
        **({"lang": lang} if lang else {}),
        debug_dir=debug_dir,
    )
    app.state.recognizer = recognizer

    logger.info(
        "ocr service started (lang=%s, debug_dir=%s)",
        recognizer.lang,
        debug_dir or "disabled",
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )
