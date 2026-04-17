from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from app.domain.models import Point


@dataclass
class TrafficSnapshot:
    congestion_matrix: list[list[float]]
    source: str
    source_url: str | None
    observed_at: datetime


class TrafficRepository(Protocol):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> TrafficSnapshot:
        raise NotImplementedError


class DisabledTrafficRepository(TrafficRepository):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> TrafficSnapshot:
        size = len(points)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        return TrafficSnapshot(
            congestion_matrix=matrix,
            source="traffic-disabled",
            source_url=None,
            observed_at=departure_at or datetime.now(timezone.utc),
        )


class CompositeTrafficRepository(TrafficRepository):
    def __init__(self, primary: TrafficRepository | None, fallback: TrafficRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> TrafficSnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points, departure_at)
            except Exception:
                return await self._fallback.fetch(points, departure_at)
        return await self._fallback.fetch(points, departure_at)
