import asyncio
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_analysis_service
from app.schemas import AnalyzeRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService


class FakeAnalysisService(AnalysisService):
    async def analyze_images_and_urls(
        self, request: AnalyzeRequest, file_bytes: List[bytes], *args, **kwargs
    ) -> AnalysisResponse:
        return AnalysisResponse(analysis_id="1", cards=[])

    async def analyze_card_names(self, names: List[str], *args, **kwargs) -> AnalysisResponse:
        cards = [
            {
                "card_name": name,
                "set_name": None,
                "collector_number": None,
                "image_url": None,
                "prices": [],
            }
            for name in names
        ]
        return AnalysisResponse(analysis_id="2", cards=[  # type: ignore[arg-type]
            *cards
        ])


@pytest.fixture(autouse=True)
def override_analysis_service(monkeypatch: pytest.MonkeyPatch):
    def _get_fake() -> AnalysisService:
        return FakeAnalysisService()

    app.dependency_overrides[get_analysis_service] = _get_fake
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_names(client: TestClient):
    payload = {"names": ["Tarmogoyf", "Lightning Bolt"]}
    resp = client.post("/analyze/names", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_id"] == "2"
    assert len(data["cards"]) == 2
    assert data["cards"][0]["card_name"] == "Tarmogoyf"


def test_get_steps_endpoint_returns_steps_for_card_names(client: TestClient):
    resp = client.get("/steps", params={"feature": "card_names"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(step["id"] == "scryfall_normalize" for step in data)


def test_ws_analyze_card_names_basic_flow(client: TestClient):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"feature": "card_names", "payload": {"names": ["Tarmogoyf"]}})
        first = ws.receive_json()
        assert first["type"] == "steps"
        result_msg = None
        for _ in range(10):
            msg = ws.receive_json()
            if msg.get("type") == "result":
                result_msg = msg
                break
        assert result_msg is not None
        assert result_msg["result"]["analysis_id"] == "2"

