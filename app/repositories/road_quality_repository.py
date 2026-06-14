from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol

from app.domain.models import Point
from app.repositories.osm_overpass_repository import OverpassRoadDataClient
from app.services.distance import haversine_km


@dataclass
class RoadQualitySnapshot:
    surface_quality_matrix: list[list[float]]
    source: str
    source_url: str | None
    observed_at: datetime


class RoadQualityRepository(Protocol):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadQualitySnapshot:
        raise NotImplementedError


class FullQualityRoadQualityRepository(RoadQualityRepository):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadQualitySnapshot:
        size = len(points)
        matrix = [[1.0 for _ in range(size)] for _ in range(size)]
        for idx in range(size):
            matrix[idx][idx] = 1.0
        return RoadQualitySnapshot(
            surface_quality_matrix=matrix,
            source="road-quality-disabled",
            source_url=None,
            observed_at=departure_at or datetime.now(timezone.utc),
        )


class JsonRoadQualityRepository(RoadQualityRepository):
    def __init__(self, source_path: str) -> None:
        self._source_path = Path(source_path)

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadQualitySnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        matrix = [[1.0 for _ in range(size)] for _ in range(size)]
        payload = json.loads(self._source_path.read_text(encoding="utf-8"))

        if isinstance(payload, dict) and isinstance(payload.get("matrix"), list):
            matrix = self._coerce_matrix(payload["matrix"], size)
        else:
            edges = payload.get("edges", payload) if isinstance(payload, dict) else payload
            if not isinstance(edges, list):
                raise ValueError("Road quality JSON must contain a matrix or edges list")
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
                value = self._bounded_value(
                    raw_edge,
                    "surface_quality",
                    "road_quality",
                    "quality",
                    "value",
                )
                matrix[from_idx][to_idx] = value
                if bool(raw_edge.get("bidirectional", False)):
                    matrix[to_idx][from_idx] = value

        return RoadQualitySnapshot(
            surface_quality_matrix=matrix,
            source="road-quality-json",
            source_url=str(self._source_path),
            observed_at=observed_at,
        )

    @staticmethod
    def _coerce_matrix(raw_matrix: list, size: int) -> list[list[float]]:
        matrix = [[1.0 for _ in range(size)] for _ in range(size)]
        for i in range(min(size, len(raw_matrix))):
            row = raw_matrix[i] if isinstance(raw_matrix[i], list) else []
            for j in range(min(size, len(row))):
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
        return 1.0


class SyntheticRoadQualityRepository(RoadQualityRepository):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadQualitySnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        matrix = [[1.0 for _ in range(size)] for _ in range(size)]

        for i, start in enumerate(points):
            for j, end in enumerate(points):
                if i == j:
                    continue
                distance = haversine_km(start, end)
                midpoint_lat = (start.lat + end.lat) / 2.0
                midpoint_lon = (start.lon + end.lon) / 2.0
                terrain_noise = abs(math.sin(midpoint_lat * 9.7) * math.cos(midpoint_lon * 7.9))
                short_local_penalty = 1.0 - min(1.0, distance / 18.0)
                long_corridor_bonus = min(0.08, distance / 450.0)
                seasonal_penalty = 0.04 if observed_at.month in {1, 2, 3, 11, 12} else 0.0

                quality = (
                    0.88
                    + long_corridor_bonus
                    - (0.16 * terrain_noise)
                    - (0.10 * short_local_penalty)
                    - seasonal_penalty
                )
                matrix[i][j] = max(0.35, min(1.0, quality))

        return RoadQualitySnapshot(
            surface_quality_matrix=matrix,
            source="synthetic-road-quality",
            source_url=None,
            observed_at=observed_at,
        )


class OverpassRoadQualityRepository(RoadQualityRepository):
    def __init__(self, client: OverpassRoadDataClient) -> None:
        self._client = client

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadQualitySnapshot:
        attributes = await self._client.fetch(points, departure_at)
        size = len(points)
        matrix = [[1.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                matrix[i][j] = min(
                    attributes.surface_quality_by_point[i],
                    attributes.surface_quality_by_point[j],
                )
        return RoadQualitySnapshot(
            surface_quality_matrix=matrix,
            source=attributes.source if attributes.source != "overpass-osm" else "overpass-osm-road-quality",
            source_url=attributes.source_url,
            observed_at=attributes.observed_at,
        )


class CompositeRoadQualityRepository(RoadQualityRepository):
    def __init__(self, primary: RoadQualityRepository | None, fallback: RoadQualityRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadQualitySnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points, departure_at)
            except Exception:
                return await self._fallback.fetch(points, departure_at)
        return await self._fallback.fetch(points, departure_at)
