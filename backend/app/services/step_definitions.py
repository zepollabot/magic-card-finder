import os
from enum import Enum
from typing import List, Optional

from ..schemas import StepInfo


class Feature(str, Enum):
    CARD_NAMES = "card_names"
    UPLOAD_IMAGES = "upload_images"
    SCRAPE_URL = "scrape_url"


def _use_extraction_service(use_extraction_service: Optional[bool] = None) -> bool:
    if use_extraction_service is not None:
        return use_extraction_service
    return bool(os.getenv("EXTRACTION_SERVICE_URL", "").strip())


def get_steps_for_feature(
    feature: Feature,
    price_source_names: List[str],
    use_extraction_service: Optional[bool] = None,
) -> List[StepInfo]:
    """
    Return the ordered list of high-level steps for the given feature.
    The price_source_names are used to expand the pricing-related steps.
    When use_extraction_service is True (or EXTRACTION_SERVICE_URL is set), image
    steps show "Card extraction (OCR)" in the status bar.
    """
    steps: List[StepInfo] = []
    use_extraction = _use_extraction_service(use_extraction_service)

    def add(step_id: str, label: str) -> None:
        steps.append(StepInfo(id=step_id, label=label, index=len(steps)))

    if feature == Feature.CARD_NAMES:
        add("scryfall_normalize", "Card normalization via Scryfall")
        for src in price_source_names:
            add(f"price_source_{src}", f"Fetch prices from {src}")
        add("report_generation", "Report generation")
    elif feature == Feature.UPLOAD_IMAGES:
        add("image_upload", "Image upload")
        if use_extraction:
            add("card_detection", "Card extraction (OCR)")
            add("card_recognition", "Recognition")
        else:
            add("card_detection", "Card detection and cropping")
            add("card_recognition", "Card recognition via LLM")
        add("scryfall_normalize", "Card normalization via Scryfall")
        for src in price_source_names:
            add(f"price_source_{src}", f"Fetch prices from {src}")
        add("report_generation", "Report generation")
    elif feature == Feature.SCRAPE_URL:
        add("scrape_urls", "Scrape URLs and download images")
        if use_extraction:
            add("card_detection", "Card extraction (OCR)")
            add("card_recognition", "Recognition")
        else:
            add("card_detection", "Card detection and cropping")
            add("card_recognition", "Card recognition via LLM")
        add("scryfall_normalize", "Card normalization via Scryfall")
        for src in price_source_names:
            add(f"price_source_{src}", f"Fetch prices from {src}")
        add("report_generation", "Report generation")

    return steps

