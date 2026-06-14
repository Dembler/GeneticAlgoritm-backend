from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Protocol

import httpx

from app.domain.models import Point
from app.services.distance import haversine_km

logger = logging.getLogger(__name__)


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


class JsonTrafficRepository(TrafficRepository):
    def __init__(self, source_path: str) -> None:
        self._source_path = Path(source_path)

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> TrafficSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        payload = json.loads(self._source_path.read_text(encoding="utf-8"))

        if isinstance(payload, dict) and isinstance(payload.get("matrix"), list):
            matrix = self._coerce_matrix(payload["matrix"], size)
        else:
            edges = payload.get("edges", payload) if isinstance(payload, dict) else payload
            if not isinstance(edges, list):
                raise ValueError("Traffic JSON must contain a matrix or edges list")
            labels = {point.label: idx for idx, point in enumerate(points) if point.label}
            for raw_edge in edges:
                if not isinstance(raw_edge, dict):
                    continue
                from_idx = self._resolve_edge_index(raw_edge, "from", "from_index", "from_label", labels)
                to_idx = self._resolve_edge_index(raw_edge, "to", "to_index", "to_label", labels)
                if from_idx is None or to_idx is None:
                    continue
                if not (0 <= from_idx < size and 0 <= to_idx < size) or from_idx == to_idx:
                    continue
                value = self._bounded_value(raw_edge, "congestion", "traffic", "traffic_index", "value")
                matrix[from_idx][to_idx] = value
                if bool(raw_edge.get("bidirectional", False)):
                    matrix[to_idx][from_idx] = value

        return TrafficSnapshot(
            congestion_matrix=matrix,
            source="traffic-json",
            source_url=str(self._source_path),
            observed_at=observed_at,
        )

    @staticmethod
    def _coerce_matrix(raw_matrix: list, size: int) -> list[list[float]]:
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(min(size, len(raw_matrix))):
            row = raw_matrix[i] if isinstance(raw_matrix[i], list) else []
            for j in range(min(size, len(row))):
                if i != j:
                    matrix[i][j] = max(0.0, min(1.0, float(row[j])))
        return matrix

    @staticmethod
    def _resolve_edge_index(
        edge: dict[str, object],
        short_key: str,
        index_key: str,
        label_key: str,
        labels: dict[str, int],
    ) -> int | None:
        raw_index = edge.get(index_key, edge.get(short_key))
        if isinstance(raw_index, int):
            return raw_index
        if isinstance(raw_index, str) and raw_index.isdigit():
            return int(raw_index)
        raw_label = edge.get(label_key)
        if isinstance(raw_label, str):
            return labels.get(raw_label)
        if isinstance(raw_index, str):
            return labels.get(raw_index)
        return None

    @staticmethod
    def _bounded_value(edge: dict[str, object], *keys: str) -> float:
        for key in keys:
            value = edge.get(key)
            if isinstance(value, int | float):
                return max(0.0, min(1.0, float(value)))
        return 0.0


class SyntheticTrafficRepository(TrafficRepository):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> TrafficSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        if size == 0:
            return TrafficSnapshot(
                congestion_matrix=[],
                source="synthetic-traffic",
                source_url=None,
                observed_at=observed_at,
            )

        centroid = Point(
            lat=sum(point.lat for point in points) / size,
            lon=sum(point.lon for point in points) / size,
        )
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        peak_factor = self._peak_factor(observed_at.hour)

        for i, start in enumerate(points):
            for j, end in enumerate(points):
                if i == j:
                    continue
                distance = haversine_km(start, end)
                midpoint = Point(lat=(start.lat + end.lat) / 2.0, lon=(start.lon + end.lon) / 2.0)
                local_density = self._local_density(points, midpoint)
                centroid_proximity = 1.0 - min(1.0, haversine_km(midpoint, centroid) / 35.0)
                short_segment_pressure = 1.0 - min(1.0, distance / 45.0)
                directional_noise = 0.5 + 0.5 * math.sin((start.lat * 17.0) + (end.lon * 13.0))

                congestion = (
                    0.03
                    + (0.34 * peak_factor)
                    + (0.26 * local_density)
                    + (0.16 * centroid_proximity)
                    + (0.13 * short_segment_pressure)
                    + (0.08 * directional_noise)
                )
                matrix[i][j] = max(0.0, min(0.95, congestion))

        return TrafficSnapshot(
            congestion_matrix=matrix,
            source="synthetic-traffic",
            source_url=None,
            observed_at=observed_at,
        )

    @staticmethod
    def _peak_factor(hour: int) -> float:
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            return 1.0
        if hour in {6, 11, 16, 21}:
            return 0.55
        if 0 <= hour <= 5:
            return 0.12
        return 0.25

    @staticmethod
    def _local_density(points: list[Point], midpoint: Point) -> float:
        if len(points) <= 1:
            return 0.0
        nearby = 0
        for point in points:
            if haversine_km(point, midpoint) <= 10.0:
                nearby += 1
        return min(1.0, nearby / max(3.0, len(points) * 0.5))


class TomTomTrafficRepository(TrafficRepository):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        max_concurrency: int = 8,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_concurrency = max(1, max_concurrency)

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> TrafficSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        if size < 2:
            return TrafficSnapshot(
                congestion_matrix=matrix,
                source="tomtom-traffic",
                source_url=self._endpoint,
                observed_at=observed_at,
            )

        semaphore = asyncio.Semaphore(self._max_concurrency)
        async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True) as client:
            tasks = [
                asyncio.create_task(
                    self._edge_congestion(
                        client=client,
                        semaphore=semaphore,
                        start=points[i],
                        end=points[j],
                        i=i,
                        j=j,
                    )
                )
                for i in range(size)
                for j in range(i + 1, size)
            ]
            edge_values = await asyncio.gather(*tasks)

        for i, j, congestion in edge_values:
            matrix[i][j] = congestion
            matrix[j][i] = congestion

        logger.warning(
            "TomTom traffic matrix: points=%d shape=%sx%s row0=%s",
            size,
            len(matrix),
            len(matrix[0]) if matrix else 0,
            [round(float(value), 3) for value in (matrix[0][: min(4, len(matrix[0]))] if matrix else [])],
        )
        return TrafficSnapshot(
            congestion_matrix=matrix,
            source="tomtom-traffic",
            source_url=self._endpoint,
            observed_at=observed_at,
        )

    async def _edge_congestion(
        self,
        *,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        start: Point,
        end: Point,
        i: int,
        j: int,
    ) -> tuple[int, int, float]:
        midpoint = Point(lat=(start.lat + end.lat) / 2.0, lon=(start.lon + end.lon) / 2.0)
        params = {
            "point": f"{midpoint.lat:.7f},{midpoint.lon:.7f}",
            "unit": "KMPH",
            "key": self._api_key,
        }
        async with semaphore:
            response = await client.get(self._endpoint, params=params)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("flowSegmentData") or {}
        congestion = self._congestion_from_speeds(
            current_speed=data.get("currentSpeed"),
            free_flow_speed=data.get("freeFlowSpeed"),
        )
        return i, j, congestion

    @property
    def _endpoint(self) -> str:
        return f"{self._base_url}/traffic/services/4/flowSegmentData/absolute/10/json"

    @staticmethod
    def _congestion_from_speeds(current_speed: object, free_flow_speed: object) -> float:
        if not isinstance(current_speed, int | float) or not isinstance(free_flow_speed, int | float):
            return 0.0
        current = max(0.0, float(current_speed))
        free_flow = max(0.0, float(free_flow_speed))
        if free_flow <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (current / free_flow)))


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
