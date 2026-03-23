"""FastAPI application for the YOLO card detector service."""
import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.routes import router as v1_router
from .api.schemas import ErrorResponse
from .detection import YOLOCardDetector, FileDetectorDebugger, NullDebugger

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(levelname)s: [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MTG Card Detector Service",
    description="YOLO26-based card name region detection",
    version="0.1.0",
)

app.include_router(v1_router, prefix="/v1", tags=["v1"])


@app.on_event("startup")
def startup():
    from ultralytics import YOLO

    model_path = os.getenv(
        "MODEL_PATH", "/app/models/mtg_detector/weights/best.pt"
    )
    if not os.path.isfile(model_path):
        logger.warning(
            "%s not found, falling back to yolo26n.pt base model",
            model_path,
        )
        model_path = "yolo26n.pt"

    model = YOLO(model_path)
    app.state.detector = YOLOCardDetector(model)
    app.state.model_name = os.path.basename(model_path)

    conf_raw = os.getenv("YOLO_CONFIDENCE", "0.4")
    try:
        app.state.confidence = float(conf_raw)
    except ValueError:
        app.state.confidence = 0.4

    debug_dir = os.getenv("DETECTOR_DEBUG_DIR", "").strip()
    if debug_dir:
        app.state.debugger = FileDetectorDebugger(debug_dir)
        logger.info("detector debug enabled: %s", debug_dir)
    else:
        app.state.debugger = NullDebugger()

    logger.info(
        "detector started (model=%s, confidence=%.2f)",
        app.state.model_name,
        app.state.confidence,
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
