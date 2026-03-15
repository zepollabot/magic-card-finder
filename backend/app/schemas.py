from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class AnalyzeRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None


class AnalyzeNamesRequest(BaseModel):
    names: List[str]


class CardPriceInfo(BaseModel):
    source: str
    currency: str
    price_low: Optional[float] = None
    price_avg: Optional[float] = None
    price_high: Optional[float] = None
    trend_price: Optional[float] = None
    set_name: Optional[str] = None  # expansion/set for this price when available
    collector_number: Optional[str] = None  # collector number for this set when available
    image_url: Optional[str] = None  # full-size image for this printing when available
    thumbnail_url: Optional[str] = None  # thumbnail for this printing when available


class CardReportItem(BaseModel):
    card_name: str
    set_name: Optional[str] = None
    collector_number: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    prices: List[CardPriceInfo]


class AnalysisResponse(BaseModel):
    analysis_id: str
    cards: List[CardReportItem]
    price_sources: List[str] = []  # enabled source names for table columns (e.g. scryfall, cardtrader)


class StepInfo(BaseModel):
    id: str
    label: str
    index: int


