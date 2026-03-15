"""API routes. Business logic is delegated to the extract orchestrator."""
import base64
import logging

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    ExtractCardsRequest,
    ExtractCardsResponse,
    ExtractResultItem,
    ResponseMeta,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/extract/cards", response_model=ExtractCardsResponse)
def extract_cards(request: Request, body: ExtractCardsRequest) -> ExtractCardsResponse:
    """
    Extract card names from base64-encoded images.
    Returns one result per image with card names in detection order.
    """
    service = getattr(request.app.state, "extract_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Extract service not initialized")

    image_bytes_list = []
    for i, b64 in enumerate(body.images):
        try:
            image_bytes_list.append(base64.b64decode(b64))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 at image index {i}: {e!s}",
            ) from e

    logger.info("extract/cards: processing %d image(s)", len(image_bytes_list))
    try:
        results_tuples = service.extract(image_bytes_list)
    except Exception as e:
        logger.exception("extract/cards: extraction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    results = [
        ExtractResultItem(image_index=idx, card_names=names)
        for idx, names in results_tuples
    ]
    total_names = sum(len(r.card_names) for r in results)
    logger.info(
        "extract/cards: done — %d result(s), %d card name(s) total (per image: %s)",
        len(results),
        total_names,
        [len(r.card_names) for r in results],
    )
    return ExtractCardsResponse(
        results=results,
        meta=ResponseMeta(processor="opencv+tesseract", version="1.0"),
    )
