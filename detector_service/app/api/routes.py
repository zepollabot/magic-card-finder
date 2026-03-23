"""API routes for the detector service."""
import base64
import logging
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    DetectRequest,
    DetectResponse,
    DetectResultItem,
    DetectionItem,
    ResponseMeta,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
def detect(request: Request, body: DetectRequest) -> DetectResponse:
    """Detect card name regions in base64-encoded images."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    debugger = getattr(request.app.state, "debugger", None)
    conf = getattr(request.app.state, "confidence", 0.4)
    model_name = getattr(request.app.state, "model_name", "unknown")
    batch_id = uuid.uuid4().hex[:8]

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
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode image at index {img_idx}",
            )

        detections = detector.detect(image, conf=conf)

        if debugger is not None:
            debugger.save(batch_id, img_idx, image, detections)

        detection_items = []
        for det in detections:
            if det.cls_name != "name":
                continue
            _, buf = cv2.imencode(".jpg", det.crop)
            crop_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            detection_items.append(
                DetectionItem(
                    cls=det.cls_name,
                    confidence=round(det.confidence, 3),
                    bbox=list(det.bbox),
                    crop_b64=crop_b64,
                )
            )

        results.append(
            DetectResultItem(
                image_index=img_idx,
                detections=detection_items,
            )
        )

    total_names = sum(len(r.detections) for r in results)
    logger.info(
        "detect: %d image(s), %d name region(s) found",
        len(results),
        total_names,
    )

    return DetectResponse(
        results=results,
        meta=ResponseMeta(processor="yolo26", model=model_name),
    )
