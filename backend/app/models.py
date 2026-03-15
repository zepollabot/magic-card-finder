from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .db import Base


class Card(Base):
    __tablename__ = "cards"

    id = Column(Integer, primary_key=True, index=True)
    scryfall_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    set_code = Column(String, index=True, nullable=True)
    set_name = Column(String, nullable=True)  # display name of set (e.g. "Dominaria")
    collector_number = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    prices = relationship("Price", back_populates="card")
    analysis_links = relationship("AnalysisCard", back_populates="card")


class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, index=True)
    card_id = Column(Integer, ForeignKey("cards.id"), nullable=False)
    source = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    price_low = Column(Float, nullable=True)
    price_avg = Column(Float, nullable=True)
    price_high = Column(Float, nullable=True)
    trend_price = Column(Float, nullable=True)
    retrieved_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    card = relationship("Card", back_populates="prices")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source_urls = Column(String, nullable=True)

    cards = relationship("AnalysisCard", back_populates="analysis", cascade="all, delete-orphan")


class AnalysisCard(Base):
    __tablename__ = "analysis_cards"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    card_id = Column(Integer, ForeignKey("cards.id"), nullable=False)

    analysis = relationship("Analysis", back_populates="cards")
    card = relationship("Card", back_populates="analysis_links")
    prices = relationship("AnalysisPrice", back_populates="analysis_card", cascade="all, delete-orphan")


class AnalysisPrice(Base):
    """Snapshot of a price at the time of a specific analysis run."""
    __tablename__ = "analysis_prices"

    id = Column(Integer, primary_key=True, index=True)
    analysis_card_id = Column(Integer, ForeignKey("analysis_cards.id"), nullable=False)
    source = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    price_low = Column(Float, nullable=True)
    price_avg = Column(Float, nullable=True)
    price_high = Column(Float, nullable=True)
    trend_price = Column(Float, nullable=True)
    set_name = Column(String, nullable=True)
    collector_number = Column(String, nullable=True)
    retrieved_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis_card = relationship("AnalysisCard", back_populates="prices")

