"""API routes for the OCR service."""
import base64
import logging

import cv2
import numpy as np
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
def recognize(request: Request, body: RecognizeRequest) -> RecognizeResponse:
    """Recognize card names from base64-encoded name-crop images."""
    recognizer = getattr(request.app.state, "recognizer", None)
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not initialized")

    results = []
    for img_idx, b64 in enumerate(body.images):
        try:
            image_bytes = base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 at image index {img_idx}: {e!s}",
            ) from e

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("recognize: could not decode image at index %d", img_idx)
            results.append(RecognizeResultItem(image_index=img_idx, text=""))
            continue

        text = recognizer.recognize(image)
        results.append(RecognizeResultItem(image_index=img_idx, text=text))

    total = sum(1 for r in results if r.text)
    logger.info(
        "recognize: %d image(s), %d name(s) recognized",
        len(results),
        total,
    )

    return RecognizeResponse(
        results=results,
        meta=ResponseMeta(processor="tesseract", version="1.0"),
    )
