from contextlib import asynccontextmanager
import logging
import os
import time
from typing import List, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import base64
import json

from .db import SessionLocal, Base, engine
from .models import Analysis, AnalysisCard, AnalysisPrice, Card, Price
from .schemas import (
    AnalyzeRequest,
    AnalyzeNamesRequest,
    AnalysisResponse,
    CardReportItem,
    CardPriceInfo,
    StepInfo,
)
from .services.analysis_service import AnalysisService
from .services.detector_service_client import DetectorServiceClient
from .services.ocr_service_client import OcrServiceClient
from .services.price_sources import get_enabled_price_source_names
from .services.step_definitions import Feature, get_steps_for_feature
from .services.progress import WebSocketProgressReporter

# Pricing and API client logs (flow, HTTP calls, response codes)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_app_log_level = getattr(logging, _log_level, logging.INFO)
_app_services_logger = logging.getLogger("app.services")
_app_services_logger.setLevel(_app_log_level)
if not _app_services_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(_app_log_level)
    _handler.setFormatter(logging.Formatter("%(levelname)s:     [%(name)s] %(message)s"))
    _app_services_logger.addHandler(_handler)
    _app_services_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Defer DB table creation to startup so we don't connect at import time.
    # Retry a few times so we can wait for the db container to be ready.
    for attempt in range(10):
        try:
            Base.metadata.create_all(bind=engine)
            # Simple in-place migrations for existing DBs.
            with engine.connect() as conn:
                from sqlalchemy import text
                try:
                    conn.execute(text("ALTER TABLE cards ADD COLUMN IF NOT EXISTS set_name VARCHAR"))
                except Exception:
                    pass  # column may already exist or DB may not support IF NOT EXISTS
                try:
                    conn.execute(text("ALTER TABLE cards ADD COLUMN IF NOT EXISTS thumbnail_url VARCHAR"))
                except Exception:
                    pass
                conn.commit()
            break
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2)

    detector_url = os.getenv("DETECTOR_SERVICE_URL", "").strip()
    ocr_url = os.getenv("OCR_SERVICE_URL", "").strip()

    http_client = httpx.AsyncClient()
    app.state._http_client = http_client

    app.state.detector_client = DetectorServiceClient(
        client=http_client, base_url=detector_url or "http://detector:8002"
    )
    app.state.ocr_client = OcrServiceClient(
        client=http_client, base_url=ocr_url or "http://ocr:8003"
    )
    _app_services_logger.info(
        "pipeline clients configured (detector=%s, ocr=%s)",
        detector_url or "http://detector:8002",
        ocr_url or "http://ocr:8003",
    )

    yield

    if getattr(app.state, "_http_client", None) is not None:
        await app.state._http_client.aclose()


app = FastAPI(title="Magic Card Finder API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def get_analysis_service(request: Request) -> AnalysisService:
    return AnalysisService(
        detector_client=request.app.state.detector_client,
        ocr_client=request.app.state.ocr_client,
    )


@app.get("/steps", response_model=List[StepInfo])
async def get_steps(feature: Feature = Query(..., description="Feature identifier, e.g. card_names")) -> List[StepInfo]:
    """
    Return the ordered list of high-level steps for the given feature.
    """
    source_names = get_enabled_price_source_names()
    return get_steps_for_feature(feature, source_names)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: str = Form(..., description="JSON-encoded AnalyzeRequest"),
    files: Optional[List[UploadFile]] = File(default=None),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    try:
        parsed = AnalyzeRequest.model_validate_json(request)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail="Invalid request: 'urls' must be valid URLs (e.g. https://...). For card names, use the Card Names tab or POST /analyze/names.",
        ) from e

    file_bytes: List[bytes] = []
    if files:
        for f in files:
            file_bytes.append(await f.read())

    return await service.analyze_images_and_urls(parsed, file_bytes)


@app.post("/analyze/names", response_model=AnalysisResponse)
async def analyze_names(body: AnalyzeNamesRequest, service: AnalysisService = Depends(get_analysis_service)) -> AnalysisResponse:
    return await service.analyze_card_names(body.names)


@app.websocket("/ws/analyze")
async def analyze_ws(websocket: WebSocket):
    await websocket.accept()
    reporter = WebSocketProgressReporter(websocket)
    service = AnalysisService(
        detector_client=websocket.app.state.detector_client,
        ocr_client=websocket.app.state.ocr_client,
    )

    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        feature_str: str = data.get("feature") or "card_names"
        payload = data.get("payload") or {}

        try:
            feature = Feature(feature_str)
        except ValueError:
            await websocket.send_json({"type": "error", "message": f"Unknown feature '{feature_str}'"})
            await websocket.close(code=1003)
            return

        source_names = get_enabled_price_source_names()
        steps = get_steps_for_feature(feature, source_names)
        await reporter.start_steps(steps)

        if feature == Feature.CARD_NAMES:
            names = payload.get("names") or []
            if not isinstance(names, list):
                await websocket.send_json({"type": "error", "message": "Invalid payload: 'names' must be a list"})
                return
            result = await service.analyze_card_names(names, progress=reporter)
        else:
            urls = payload.get("urls") or []
            files_b64 = payload.get("files") or []
            if not isinstance(urls, list) or not isinstance(files_b64, list):
                await websocket.send_json(
                    {"type": "error", "message": "Invalid payload: 'urls' and 'files' must be lists"}
                )
                return

            analyze_request = AnalyzeRequest(urls=[u for u in urls])
            file_bytes: List[bytes] = []
            for encoded in files_b64:
                try:
                    file_bytes.append(base64.b64decode(encoded))
                except Exception:
                    continue

            result = await service.analyze_images_and_urls(analyze_request, file_bytes, progress=reporter)

        await websocket.send_json({"type": "result", "result": result.model_dump()})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1011)


@app.get("/report/{analysis_id}", response_model=AnalysisResponse)
async def get_report(analysis_id: int) -> AnalysisResponse:
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter_by(id=analysis_id).one_or_none()
        if analysis is None:
            return AnalysisResponse(analysis_id=str(analysis_id), cards=[], price_sources=[])

        by_name: dict = {}
        for link in analysis.cards:
            card = link.card
            name = card.name
            if name not in by_name:
                by_name[name] = {
                    "card_name": name,
                    "set_name": None,
                    "collector_number": card.collector_number,
                    "image_url": card.image_url,
                    "thumbnail_url": getattr(card, "thumbnail_url", None),
                    "prices": [],
                }
            has_prices = False
            card_image = card.image_url
            card_thumb = getattr(card, "thumbnail_url", None) or card.image_url
            for ap in link.prices:
                has_prices = True
                by_name[name]["prices"].append(
                    CardPriceInfo(
                        source=ap.source,
                        currency=ap.currency,
                        price_low=ap.price_low,
                        price_avg=ap.price_avg,
                        price_high=ap.price_high,
                        trend_price=ap.trend_price,
                        set_name=ap.set_name or getattr(card, "set_name", None) or card.set_code,
                        collector_number=ap.collector_number or card.collector_number,
                        image_url=card_image,
                        thumbnail_url=card_thumb,
                    )
                )
            if not has_prices:
                set_display = getattr(card, "set_name", None) or card.set_code
                by_name[name]["prices"].append(
                    CardPriceInfo(
                        source="__noprice__",
                        currency="",
                        price_low=None,
                        price_avg=None,
                        price_high=None,
                        trend_price=None,
                        set_name=set_display,
                        collector_number=card.collector_number,
                        image_url=card_image,
                        thumbnail_url=card_thumb,
                    )
                )
        items = [CardReportItem(**v) for v in by_name.values()]
        return AnalysisResponse(
            analysis_id=str(analysis.id),
            cards=items,
            price_sources=get_enabled_price_source_names(),
        )
    finally:
        db.close()


@app.get("/report/{analysis_id}/csv")
async def get_report_csv(analysis_id: int):
    from fastapi.responses import PlainTextResponse
    import csv
    from io import StringIO

    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter_by(id=analysis_id).one_or_none()
        if analysis is None:
            return PlainTextResponse("", media_type="text/csv")

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "card_name",
                "set_name",
                "collector_number",
                "image_url",
                "thumbnail_url",
                "source",
                "currency",
                "price_low",
                "price_avg",
                "price_high",
                "trend_price",
            ]
        )

        for link in analysis.cards:
            card = link.card
            if not link.prices:
                set_display = getattr(card, "set_name", None) or card.set_code or ""
                writer.writerow(
                    [
                        card.name,
                        set_display,
                        card.collector_number or "",
                        card.image_url or "",
                        getattr(card, "thumbnail_url", "") or "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            else:
                for ap in link.prices:
                    set_display = ap.set_name or getattr(card, "set_name", None) or card.set_code or ""
                    writer.writerow(
                        [
                            card.name,
                            set_display,
                            ap.collector_number or card.collector_number or "",
                            card.image_url or "",
                            getattr(card, "thumbnail_url", "") or "",
                            ap.source,
                            ap.currency,
                            ap.price_low if ap.price_low is not None else "",
                            ap.price_avg if ap.price_avg is not None else "",
                            ap.price_high if ap.price_high is not None else "",
                            ap.trend_price if ap.trend_price is not None else "",
                        ]
                    )

        return PlainTextResponse(output.getvalue(), media_type="text/csv")
    finally:
        db.close()


