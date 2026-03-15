from __future__ import annotations

from typing import List, Optional, Protocol

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder

from ..schemas import StepInfo


class ProgressReporter(Protocol):
    async def start_steps(self, steps: List[StepInfo]) -> None: ...

    async def step_start(self, step_id: str, step_index: int, message: Optional[str] = None) -> None: ...

    async def step_complete(self, step_id: str, step_index: int) -> None: ...

    async def progress(self, step_id: str, current: int, total: int) -> None: ...


class NoOpProgressReporter:
    async def start_steps(self, steps: List[StepInfo]) -> None:  # type: ignore[override]
        return None

    async def step_start(self, step_id: str, step_index: int, message: Optional[str] = None) -> None:  # type: ignore[override]
        return None

    async def step_complete(self, step_id: str, step_index: int) -> None:  # type: ignore[override]
        return None

    async def progress(self, step_id: str, current: int, total: int) -> None:  # type: ignore[override]
        return None


class WebSocketProgressReporter:
    """
    Progress reporter that sends JSON messages over a WebSocket to the frontend.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def start_steps(self, steps: List[StepInfo]) -> None:
        payload = {"type": "steps", "steps": jsonable_encoder(steps)}
        await self._ws.send_json(payload)

    async def step_start(self, step_id: str, step_index: int, message: Optional[str] = None) -> None:
        payload = {
            "type": "step_start",
            "step_id": step_id,
            "step_index": step_index,
            "message": message,
        }
        await self._ws.send_json(payload)

    async def step_complete(self, step_id: str, step_index: int) -> None:
        payload = {
            "type": "step_complete",
            "step_id": step_id,
            "step_index": step_index,
        }
        await self._ws.send_json(payload)

    async def progress(self, step_id: str, current: int, total: int) -> None:
        payload = {
            "type": "progress",
            "step_id": step_id,
            "current": current,
            "total": total,
        }
        await self._ws.send_json(payload)

