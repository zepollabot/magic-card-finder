from typing import Dict, List, Optional, Tuple

import logging

from ..db import SessionLocal

logger = logging.getLogger(__name__)
from ..models import Analysis, AnalysisCard, AnalysisPrice, Card
from ..schemas import AnalyzeRequest, AnalysisResponse, CardReportItem, CardPriceInfo
from .image_ingest import ImageIngestService
from .detector_service_client import DetectorClient
from .ocr_service_client import OcrClient
from .scryfall_client import ScryfallClient
from .pricing_aggregator import PricingAggregator
from .progress import ProgressReporter, NoOpProgressReporter
from .step_definitions import Feature, get_steps_for_feature
from .card_name_resolver import CardNameResolver, ScryfallCardNameResolver


class AnalysisService:
    """
    Application service that orchestrates image-based and name-based analyses.

    Detection and OCR are handled by two independent microservices,
    accessed through protocol abstractions (DIP).

    All features converge on the same resolve-price-persist pipeline via
    ``_resolve_and_price_cards``, so the pricing logic is never duplicated.
    """

    def __init__(
        self,
        detector_client: DetectorClient,
        ocr_client: OcrClient,
        image_ingest: Optional[ImageIngestService] = None,
        scryfall: Optional[ScryfallClient] = None,
        pricing: Optional[PricingAggregator] = None,
        name_resolver: Optional[CardNameResolver] = None,
    ) -> None:
        self.detector_client = detector_client
        self.ocr_client = ocr_client
        self.image_ingest = image_ingest or ImageIngestService()
        self.scryfall = scryfall or ScryfallClient()
        self.pricing = pricing or PricingAggregator()
        self.name_resolver = name_resolver or ScryfallCardNameResolver(self.scryfall)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def analyze_images_and_urls(
        self,
        request: AnalyzeRequest,
        file_bytes: List[bytes],
        progress: Optional[ProgressReporter] = None,
    ) -> AnalysisResponse:
        """Extract card names from images, then run the shared price pipeline."""
        reporter: ProgressReporter = progress or NoOpProgressReporter()
        image_bytes: List[bytes] = []

        if request.urls:
            await reporter.step_start("scrape_urls", 0, "Fetching images from URLs")
            image_bytes.extend(
                await self.image_ingest.fetch_image_bytes_from_urls([str(u) for u in request.urls])
            )
            await reporter.step_complete("scrape_urls", 0)

        image_bytes.extend(file_bytes)

        price_source_names = self.pricing.get_enabled_source_names()
        steps = get_steps_for_feature(Feature.UPLOAD_IMAGES if file_bytes else Feature.SCRAPE_URL, price_source_names)
        step_index_by_id = {s.id: s.index for s in steps}
        await reporter.start_steps(steps)

        upload_idx = step_index_by_id.get("image_upload")
        if upload_idx is not None:
            await reporter.step_start("image_upload", upload_idx, "Processing uploaded images")
            await reporter.step_complete("image_upload", upload_idx)

        card_names = await self._extract_names_from_images(
            image_bytes, step_index_by_id, reporter,
        )

        return await self._resolve_and_price_cards(
            entries=[(n, None) for n in card_names],
            price_source_names=price_source_names,
            step_index_by_id=step_index_by_id,
            reporter=reporter,
            source_urls=",".join([str(u) for u in (request.urls or [])]) or None,
        )

    async def analyze_card_names(
        self,
        names: List[str],
        progress: Optional[ProgressReporter] = None,
    ) -> AnalysisResponse:
        """Resolve card names and run the shared price pipeline."""
        entries = self._parse_name_lines(names)
        reporter: ProgressReporter = progress or NoOpProgressReporter()

        price_source_names = self.pricing.get_enabled_source_names()
        steps = get_steps_for_feature(Feature.CARD_NAMES, price_source_names)
        step_index_by_id = {s.id: s.index for s in steps}
        await reporter.start_steps(steps)

        return await self._resolve_and_price_cards(
            entries=entries,
            price_source_names=price_source_names,
            step_index_by_id=step_index_by_id,
            reporter=reporter,
            source_urls=None,
        )

    # ------------------------------------------------------------------
    # Image → card-name extraction (image-specific logic)
    # ------------------------------------------------------------------

    async def _extract_names_from_images(
        self,
        image_bytes: List[bytes],
        step_index_by_id: Dict[str, int],
        reporter: ProgressReporter,
    ) -> List[str]:
        """Return a flat list of recognised card names from raw image bytes.

        Orchestrates the detector service (YOLO) then the OCR service
        (Tesseract) as two independent steps with real progress reporting.
        """
        det_idx = step_index_by_id.get("card_detection", 1)
        await reporter.step_start("card_detection", det_idx, "Detecting card names")
        detection_results = await self.detector_client.detect(image_bytes)
        await reporter.step_complete("card_detection", det_idx)

        all_crops: List[bytes] = []
        for det_result in detection_results:
            for crop in det_result.name_crops:
                all_crops.append(crop.image_bytes)

        if not all_crops:
            rec_idx = step_index_by_id.get("card_recognition", 2)
            await reporter.step_start("card_recognition", rec_idx, "Reading card names")
            await reporter.step_complete("card_recognition", rec_idx)
            return []

        rec_idx = step_index_by_id.get("card_recognition", 2)
        await reporter.step_start("card_recognition", rec_idx, "Reading card names")
        recognized_texts = await self.ocr_client.recognize(all_crops)
        await reporter.step_complete("card_recognition", rec_idx)

        all_names = [text for text in recognized_texts if text and text.strip()]
        logger.info(
            "image analysis: %d crop(s) -> %d name(s) recognized",
            len(all_crops),
            len(all_names),
        )
        return all_names

    # ------------------------------------------------------------------
    # Shared resolve → price → persist pipeline
    # ------------------------------------------------------------------

    async def _resolve_and_price_cards(
        self,
        entries: List[Tuple[str, Optional[str]]],
        price_source_names: List[str],
        step_index_by_id: Dict[str, int],
        reporter: ProgressReporter,
        source_urls: Optional[str],
    ) -> AnalysisResponse:
        """
        Core pipeline shared by every feature.

        *entries* is a list of ``(card_name, set_hint | None)`` tuples.
        """
        db = SessionLocal()
        try:
            analysis = Analysis(source_urls=source_urls)
            db.add(analysis)
            db.flush()

            card_reports: List[CardReportItem] = []
            seen_canonical: set = set()

            for entry_idx, (line_name, set_hint) in enumerate(entries):
                scryfall_idx = step_index_by_id.get("scryfall_normalize", 0)
                await reporter.step_start(
                    "scryfall_normalize", scryfall_idx, f"Resolving '{line_name}' via Scryfall"
                )
                await reporter.progress("scryfall_normalize", entry_idx + 1, len(entries))
                cards = await self.name_resolver.resolve(line_name, set_hint=set_hint)
                if not cards:
                    await reporter.step_complete("scryfall_normalize", scryfall_idx)
                    continue

                for scry_first in cards:
                    canonical_name = (scry_first.get("name") or line_name).strip()
                    if not canonical_name or scry_first.get("id") is None:
                        continue
                    if canonical_name in seen_canonical:
                        continue
                    seen_canonical.add(canonical_name)
                    await reporter.step_complete("scryfall_normalize", scryfall_idx)

                    printings = await self.scryfall.search_printings(canonical_name)
                    if not printings:
                        printings = [scry_first]

                    preferred = scry_first
                    if set_hint:
                        set_hint_lower = set_hint.strip().lower()
                        for p in printings:
                            if (p.get("set") or "").lower() == set_hint_lower:
                                preferred = p
                                break
                            if (p.get("set_name") or "").lower() == set_hint_lower:
                                preferred = p
                                break
                    else:
                        preferred = printings[0]

                    for src_name in price_source_names:
                        step_id = f"price_source_{src_name}"
                        idx = step_index_by_id.get(step_id)
                        if idx is not None:
                            await reporter.step_start(
                                step_id, idx, f"Fetching {src_name} prices for {canonical_name}"
                            )

                    all_prices: List[CardPriceInfo] = []
                    for printing in printings:
                        sid = printing.get("id")
                        if not sid:
                            continue
                        raw_prices = await self.pricing.get_prices_for_card(printing)
                        set_name = printing.get("set_name") or printing.get("set")
                        col_num = printing.get("collector_number")
                        image_uris = printing.get("image_uris") or {}
                        print_image_url = image_uris.get("normal")
                        print_thumb_url = image_uris.get("small") or image_uris.get("normal")
                        had_price = False
                        for p in raw_prices:
                            had_price = True
                            all_prices.append(
                                CardPriceInfo(
                                    source=p.source,
                                    currency=p.currency,
                                    price_low=p.price_low,
                                    price_avg=p.price_avg,
                                    price_high=p.price_high,
                                    trend_price=p.trend_price,
                                    set_name=set_name,
                                    collector_number=col_num,
                                    image_url=print_image_url,
                                    thumbnail_url=print_thumb_url,
                                )
                            )
                        if not had_price:
                            all_prices.append(
                                CardPriceInfo(
                                    source="__noprice__",
                                    currency="",
                                    price_low=None,
                                    price_avg=None,
                                    price_high=None,
                                    trend_price=None,
                                    set_name=set_name,
                                    collector_number=col_num,
                                    image_url=print_image_url,
                                    thumbnail_url=print_thumb_url,
                                )
                            )

                        card = db.query(Card).filter_by(scryfall_id=sid).one_or_none()
                        if card is None:
                            card = Card(
                                scryfall_id=sid,
                                name=printing.get("name", canonical_name),
                                set_code=printing.get("set"),
                                set_name=printing.get("set_name"),
                                collector_number=col_num,
                                image_url=print_image_url,
                                thumbnail_url=print_thumb_url,
                            )
                            db.add(card)
                            db.flush()

                        ac = AnalysisCard(analysis_id=analysis.id, card_id=card.id)
                        db.add(ac)
                        db.flush()

                        for p in raw_prices:
                            db.add(
                                AnalysisPrice(
                                    analysis_card_id=ac.id,
                                    source=p.source,
                                    currency=p.currency,
                                    price_low=p.price_low,
                                    price_avg=p.price_avg,
                                    price_high=p.price_high,
                                    trend_price=p.trend_price,
                                    set_name=set_name,
                                    collector_number=col_num,
                                )
                            )

                    for src_name in price_source_names:
                        step_id = f"price_source_{src_name}"
                        idx = step_index_by_id.get(step_id)
                        if idx is not None:
                            await reporter.step_complete(step_id, idx)

                    image_url = (preferred.get("image_uris") or {}).get("normal")
                    collector_number = preferred.get("collector_number")

                    card_reports.append(
                        CardReportItem(
                            card_name=canonical_name,
                            set_name=None,
                            collector_number=collector_number,
                            image_url=image_url,
                            prices=all_prices,
                        )
                    )

            report_idx = step_index_by_id.get("report_generation")
            if report_idx is not None:
                await reporter.step_start("report_generation", report_idx, "Generating report")
            db.commit()
            if report_idx is not None:
                await reporter.step_complete("report_generation", report_idx)

            return AnalysisResponse(
                analysis_id=str(analysis.id),
                cards=card_reports,
                price_sources=self.pricing.get_enabled_source_names(),
            )
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_name_lines(names: List[str]) -> List[tuple]:
        """Parse list of lines into (card_name, set_hint). One line = one card; 'Name, Set' allowed."""
        out: List[tuple] = []
        for raw in names:
            line = raw.strip()
            if not line:
                continue
            if "," in line:
                parts = line.split(",", 1)
                name_part = parts[0].strip()
                set_part = parts[1].strip() or None
                if name_part:
                    out.append((name_part, set_part))
            else:
                out.append((line, None))
        return out
