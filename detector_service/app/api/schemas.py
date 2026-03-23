"""Pydantic schemas for the detector API."""
from typing import List

from pydantic import BaseModel


class DetectRequest(BaseModel):
    images: List[str]


class DetectionItem(BaseModel):
    cls: str
    confidence: float
    bbox: List[int]
    crop_b64: str


class DetectResultItem(BaseModel):
    image_index: int
    detections: List[DetectionItem]


class ResponseMeta(BaseModel):
    processor: str
    model: str


class DetectResponse(BaseModel):
    results: List[DetectResultItem]
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
