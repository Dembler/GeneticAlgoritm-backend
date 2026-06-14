from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol

from app.domain.models import Point, TransportProfile
from app.repositories.osm_overpass_repository import OverpassRoadDataClient
from app.services.distance import haversine_km


@dataclass
class InfrastructureSnapshot:
    height_clearance_matrix_m: list[list[float | None]]
    weight_limit_matrix_t: list[list[float | None]]
    width_limit_matrix_m: list[list[float | None]]
    length_limit_matrix_m: list[list[float | None]]
    access_matrix: list[list[bool]]
    source: str
    source_url: str | None
    observed_at: datetime


class InfrastructureRepository(Protocol):
    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        departure_at: datetime | None = None,
    ) -> InfrastructureSnapshot:
        raise NotImplementedError


class UnrestrictedInfrastructureRepository(InfrastructureRepository):
    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        departure_at: datetime | None = None,
    ) -> InfrastructureSnapshot:
        size = len(points)
        optional_matrix = [[None for _ in range(size)] for _ in range(size)]
        access_matrix = [[True for _ in range(size)] for _ in range(size)]
        return InfrastructureSnapshot(
            height_clearance_matrix_m=[row.copy() for row in optional_matrix],
            weight_limit_matrix_t=[row.copy() for row in optional_matrix],
            width_limit_matrix_m=[row.copy() for row in optional_matrix],
            length_limit_matrix_m=[row.copy() for row in optional_matrix],
            access_matrix=access_matrix,
            source="infrastructure-unrestricted",
            source_url=None,
            observed_at=departure_at or datetime.now(timezone.utc),
        )


class JsonInfrastructureRepository(InfrastructureRepository):
    def __init__(self, source_path: str) -> None:
        self._source_path = Path(source_path)

    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        departure_at: datetime | None = None,
    ) -> InfrastructureSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        height_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        weight_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        width_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        length_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        access_matrix = [[True for _ in range(size)] for _ in range(size)]

        if profile != TransportProfile.driving:
            return InfrastructureSnapshot(
                height_clearance_matrix_m=height_matrix,
                weight_limit_matrix_t=weight_matrix,
                width_limit_matrix_m=width_matrix,
                length_limit_matrix_m=length_matrix,
                access_matrix=access_matrix,
                source="infrastructure-json",
                source_url=str(self._source_path),
                observed_at=observed_at,
            )

        payload = json.loads(self._source_path.read_text(encoding="utf-8"))
        edges = payload.get("edges", payload) if isinstance(payload, dict) else payload
        if not isinstance(edges, list):
            raise ValueError("Infrastructure JSON must contain an edges list")

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
            self._apply_edge(
                raw_edge,
                from_idx,
                to_idx,
                height_matrix,
                weight_matrix,
                width_matrix,
                length_matrix,
                access_matrix,
            )
            if bool(raw_edge.get("bidirectional", False)):
                self._apply_edge(
                    raw_edge,
                    to_idx,
                    from_idx,
                    height_matrix,
                    weight_matrix,
                    width_matrix,
                    length_matrix,
                    access_matrix,
                )

        return InfrastructureSnapshot(
            height_clearance_matrix_m=height_matrix,
            weight_limit_matrix_t=weight_matrix,
            width_limit_matrix_m=width_matrix,
            length_limit_matrix_m=length_matrix,
            access_matrix=access_matrix,
            source="infrastructure-json",
            source_url=str(self._source_path),
            observed_at=observed_at,
        )

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

    @classmethod
    def _apply_edge(
        cls,
        edge: dict[str, object],
        from_idx: int,
        to_idx: int,
        height_matrix: list[list[float | None]],
        weight_matrix: list[list[float | None]],
        width_matrix: list[list[float | None]],
        length_matrix: list[list[float | None]],
        access_matrix: list[list[bool]],
    ) -> None:
        height_matrix[from_idx][to_idx] = cls._positive_number(
            edge,
            "height_clearance_m",
            "height_m",
            "max_height_m",
        )
        weight_matrix[from_idx][to_idx] = cls._positive_number(edge, "weight_limit_t", "weight_t", "max_weight_t")
        width_matrix[from_idx][to_idx] = cls._positive_number(edge, "width_limit_m", "width_m", "max_width_m")
        length_matrix[from_idx][to_idx] = cls._positive_number(edge, "length_limit_m", "length_m", "max_length_m")
        if "accessible" in edge:
            access_matrix[from_idx][to_idx] = bool(edge["accessible"])

    @staticmethod
    def _positive_number(edge: dict[str, object], *keys: str) -> float | None:
        for key in keys:
            value = edge.get(key)
            if isinstance(value, int | float) and value > 0:
                return float(value)
        return None


class SyntheticInfrastructureRepository(InfrastructureRepository):
    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        departure_at: datetime | None = None,
    ) -> InfrastructureSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        height_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        weight_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        width_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        length_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        access_matrix = [[True for _ in range(size)] for _ in range(size)]

        if profile != TransportProfile.driving:
            return InfrastructureSnapshot(
                height_clearance_matrix_m=height_matrix,
                weight_limit_matrix_t=weight_matrix,
                width_limit_matrix_m=width_matrix,
                length_limit_matrix_m=length_matrix,
                access_matrix=access_matrix,
                source="synthetic-infrastructure",
                source_url=None,
                observed_at=observed_at,
            )

        for i, start in enumerate(points):
            for j, end in enumerate(points):
                if i == j:
                    continue
                distance = haversine_km(start, end)
                midpoint_lat = (start.lat + end.lat) / 2.0
                midpoint_lon = (start.lon + end.lon) / 2.0
                local_pressure = 1.0 - min(1.0, distance / 28.0)
                corridor_relief = min(1.0, distance / 160.0)
                bridge_signal = 0.5 + 0.5 * math.sin((midpoint_lat * 11.0) + (midpoint_lon * 5.0))
                freight_signal = 0.5 + 0.5 * math.cos((start.lat * 7.0) - (end.lon * 9.0))

                height_limit = 4.65 - (0.85 * local_pressure) - (0.55 * bridge_signal) + (0.25 * corridor_relief)
                weight_limit = 32.0 - (14.0 * local_pressure) - (8.0 * freight_signal) + (8.0 * corridor_relief)
                width_limit = 3.25 - (0.45 * local_pressure) - (0.28 * bridge_signal)
                length_limit = 18.5 - (5.5 * local_pressure) - (2.8 * freight_signal) + (2.0 * corridor_relief)

                height_matrix[i][j] = round(max(2.9, min(5.0, height_limit)), 2)
                weight_matrix[i][j] = round(max(6.0, min(44.0, weight_limit)), 1)
                width_matrix[i][j] = round(max(2.2, min(3.8, width_limit)), 2)
                length_matrix[i][j] = round(max(7.0, min(22.0, length_limit)), 1)

        return InfrastructureSnapshot(
            height_clearance_matrix_m=height_matrix,
            weight_limit_matrix_t=weight_matrix,
            width_limit_matrix_m=width_matrix,
            length_limit_matrix_m=length_matrix,
            access_matrix=access_matrix,
            source="synthetic-infrastructure",
            source_url=None,
            observed_at=observed_at,
        )


class OverpassInfrastructureRepository(InfrastructureRepository):
    def __init__(self, client: OverpassRoadDataClient) -> None:
        self._client = client

    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        departure_at: datetime | None = None,
    ) -> InfrastructureSnapshot:
        attributes = await self._client.fetch(points, departure_at)
        size = len(points)
        height_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        weight_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        width_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        length_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
        access_matrix = [[True for _ in range(size)] for _ in range(size)]

        if profile != TransportProfile.driving:
            return InfrastructureSnapshot(
                height_clearance_matrix_m=height_matrix,
                weight_limit_matrix_t=weight_matrix,
                width_limit_matrix_m=width_matrix,
                length_limit_matrix_m=length_matrix,
                access_matrix=access_matrix,
                source=attributes.source if attributes.source != "overpass-osm" else "overpass-osm-infrastructure",
                source_url=attributes.source_url,
                observed_at=attributes.observed_at,
            )

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                height_matrix[i][j] = self._min_optional(
                    attributes.height_clearance_m_by_point[i],
                    attributes.height_clearance_m_by_point[j],
                )
                weight_matrix[i][j] = self._min_optional(
                    attributes.weight_limit_t_by_point[i],
                    attributes.weight_limit_t_by_point[j],
                )
                width_matrix[i][j] = self._min_optional(
                    attributes.width_limit_m_by_point[i],
                    attributes.width_limit_m_by_point[j],
                )
                length_matrix[i][j] = self._min_optional(
                    attributes.length_limit_m_by_point[i],
                    attributes.length_limit_m_by_point[j],
                )
                access_matrix[i][j] = attributes.access_by_point[i] and attributes.access_by_point[j]

        return InfrastructureSnapshot(
            height_clearance_matrix_m=height_matrix,
            weight_limit_matrix_t=weight_matrix,
            width_limit_matrix_m=width_matrix,
            length_limit_matrix_m=length_matrix,
            access_matrix=access_matrix,
            source=attributes.source if attributes.source != "overpass-osm" else "overpass-osm-infrastructure",
            source_url=attributes.source_url,
            observed_at=attributes.observed_at,
        )

    @staticmethod
    def _min_optional(first: float | None, second: float | None) -> float | None:
        values = [value for value in (first, second) if value is not None]
        return min(values) if values else None


class CompositeInfrastructureRepository(InfrastructureRepository):
    def __init__(
        self,
        primary: InfrastructureRepository | None,
        fallback: InfrastructureRepository,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        departure_at: datetime | None = None,
    ) -> InfrastructureSnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points, profile, departure_at)
            except Exception:
                return await self._fallback.fetch(points, profile, departure_at)
        return await self._fallback.fetch(points, profile, departure_at)
