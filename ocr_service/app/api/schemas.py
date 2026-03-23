"""Pydantic schemas for the OCR API."""
from typing import List

from pydantic import BaseModel


class RecognizeRequest(BaseModel):
    images: List[str]


class RecognizeResultItem(BaseModel):
    image_index: int
    text: str


class ResponseMeta(BaseModel):
    processor: str
    version: str


class RecognizeResponse(BaseModel):
    results: List[RecognizeResultItem]
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
