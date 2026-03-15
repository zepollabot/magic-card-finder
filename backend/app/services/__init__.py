"""
Service layer package for image ingest, card detection, recognition, Scryfall access,
and pricing aggregation.

Pricing is pluggable: implement PriceSource and add to get_default_price_sources().
"""
from .price_sources import PriceSource, get_default_price_sources

__all__ = ["PriceSource", "get_default_price_sources"]

