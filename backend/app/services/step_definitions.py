from enum import Enum
from typing import List

from ..schemas import StepInfo


class Feature(str, Enum):
    CARD_NAMES = "card_names"
    UPLOAD_IMAGES = "upload_images"
    SCRAPE_URL = "scrape_url"


def get_steps_for_feature(
    feature: Feature,
    price_source_names: List[str],
) -> List[StepInfo]:
    """
    Return the ordered list of high-level steps for the given feature.
    The price_source_names are used to expand the pricing-related steps.
    """
    steps: List[StepInfo] = []

    def add(step_id: str, label: str) -> None:
        steps.append(StepInfo(id=step_id, label=label, index=len(steps)))

    if feature == Feature.CARD_NAMES:
        add("scryfall_normalize", "Card normalization via Scryfall")
        for src in price_source_names:
            add(f"price_source_{src}", f"Fetch prices from {src}")
        add("report_generation", "Report generation")
    elif feature == Feature.UPLOAD_IMAGES:
        add("image_upload", "Image upload")
        add("card_detection", "Detecting card names (YOLO)")
        add("card_recognition", "Reading card names (OCR)")
        add("scryfall_normalize", "Card normalization via Scryfall")
        for src in price_source_names:
            add(f"price_source_{src}", f"Fetch prices from {src}")
        add("report_generation", "Report generation")
    elif feature == Feature.SCRAPE_URL:
        add("scrape_urls", "Scrape URLs and download images")
        add("card_detection", "Detecting card names (YOLO)")
        add("card_recognition", "Reading card names (OCR)")
        add("scryfall_normalize", "Card normalization via Scryfall")
        for src in price_source_names:
            add(f"price_source_{src}", f"Fetch prices from {src}")
        add("report_generation", "Report generation")

    return steps
