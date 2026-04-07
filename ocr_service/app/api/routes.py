"""API routes for the OCR service."""
import base64
import logging

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    RecognizeRequest,
    RecognizeResponse,
    RecognizeResultItem,
    ResponseMeta,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: Request, body: RecognizeRequest) -> RecognizeResponse:
    """Recognize card names from base64-encoded name-crop images."""
    recognizer = getattr(request.app.state, "recognizer", None)
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not initialized")

    results = []
    for img_idx, b64 in enumerate(body.images):
        try:
            image_bytes = base64.b64decode(b64, validate=True)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 at image index {img_idx}: {e!s}",
            ) from e

        text = await recognizer.recognize(image_bytes)
        results.append(RecognizeResultItem(image_index=img_idx, text=text))

    total = sum(1 for r in results if r.text)
    logger.info(
        "recognize: %d image(s), %d name(s) recognized",
        len(results),
        total,
    )

    return RecognizeResponse(
        results=results,
        meta=ResponseMeta(processor="ollama", version="1.0"),
    )
