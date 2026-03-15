"""Request/response models matching the OpenAPI spec."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractCardsRequest(BaseModel):
    """Request body for POST /v1/extract/cards."""

    images: List[str] = Field(..., description="Base64-encoded image bytes (JPEG/PNG)")


class ExtractResultItem(BaseModel):
    """One result entry per input image."""

    image_index: int = Field(..., ge=0, description="Index of the input image")
    card_names: List[str] = Field(..., description="Card names in detection order")


class ResponseMeta(BaseModel):
    """Optional metadata for debugging and multi-engine support."""

    processor: Optional[str] = None
    version: Optional[str] = None


class ExtractCardsResponse(BaseModel):
    """Response for POST /v1/extract/cards."""

    results: List[ExtractResultItem]
    meta: Optional[ResponseMeta] = None


class ErrorResponse(BaseModel):
    """Shared error shape."""

    detail: str
    code: Optional[str] = None
