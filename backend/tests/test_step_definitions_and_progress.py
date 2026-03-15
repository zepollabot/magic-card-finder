import pytest
from unittest.mock import AsyncMock

from app.schemas import StepInfo
from app.services.step_definitions import Feature, get_steps_for_feature
from app.services.progress import WebSocketProgressReporter


def _ids(steps):
  return [s.id for s in steps]


def test_get_steps_for_feature_card_names_uses_price_sources():
    price_sources = ["scryfall", "cardtrader"]
    steps = get_steps_for_feature(Feature.CARD_NAMES, price_sources)
    assert _ids(steps) == [
        "scryfall_normalize",
        "price_source_scryfall",
        "price_source_cardtrader",
        "report_generation",
    ]
    assert all(isinstance(s, StepInfo) for s in steps)


def test_get_steps_for_feature_upload_images_includes_detection_and_recognition():
    steps = get_steps_for_feature(Feature.UPLOAD_IMAGES, ["scryfall"])
    ids = _ids(steps)
    assert ids[:4] == [
        "image_upload",
        "card_detection",
        "card_recognition",
        "scryfall_normalize",
    ]
    assert "price_source_scryfall" in ids
    assert ids[-1] == "report_generation"


def test_get_steps_for_feature_scrape_url_includes_scrape_first():
    steps = get_steps_for_feature(Feature.SCRAPE_URL, [])
    ids = _ids(steps)
    assert ids[0] == "scrape_urls"
    assert "card_detection" in ids
    assert "card_recognition" in ids
    assert "scryfall_normalize" in ids
    assert ids[-1] == "report_generation"


@pytest.mark.asyncio
async def test_websocket_progress_reporter_sends_expected_payloads():
    sent = []

    class FakeWebSocket:
        async def send_json(self, payload):
            sent.append(payload)

    ws = FakeWebSocket()
    reporter = WebSocketProgressReporter(ws)  # type: ignore[arg-type]

    steps = [
        StepInfo(id="step1", label="Step 1", index=0),
        StepInfo(id="step2", label="Step 2", index=1),
    ]

    await reporter.start_steps(steps)
    await reporter.step_start("step1", 0, "starting")
    await reporter.step_complete("step1", 0)
    await reporter.progress("step1", 1, 2)

    assert sent[0]["type"] == "steps"
    assert sent[0]["steps"][0]["id"] == "step1"
    assert sent[1] == {
        "type": "step_start",
        "step_id": "step1",
        "step_index": 0,
        "message": "starting",
    }
    assert sent[2] == {
        "type": "step_complete",
        "step_id": "step1",
        "step_index": 0,
    }
    assert sent[3] == {
        "type": "progress",
        "step_id": "step1",
        "current": 1,
        "total": 2,
    }

