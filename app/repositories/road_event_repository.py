from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol

import httpx

from app.domain.models import Point
from app.repositories.osm_overpass_repository import OverpassRoadDataClient
from app.services.distance import haversine_km


@dataclass
class RoadEventSnapshot:
    incident_risk_matrix: list[list[float]]
    roadwork_risk_matrix: list[list[float]]
    temporal_access_matrix: list[list[bool]]
    source: str
    source_url: str | None
    observed_at: datetime


class RoadEventRepository(Protocol):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        raise NotImplementedError


class DisabledRoadEventRepository(RoadEventRepository):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        size = len(points)
        return RoadEventSnapshot(
            incident_risk_matrix=[[0.0 for _ in range(size)] for _ in range(size)],
            roadwork_risk_matrix=[[0.0 for _ in range(size)] for _ in range(size)],
            temporal_access_matrix=[[True for _ in range(size)] for _ in range(size)],
            source="road-events-disabled",
            source_url=None,
            observed_at=departure_at or datetime.now(timezone.utc),
        )


class JsonRoadEventRepository(RoadEventRepository):
    def __init__(self, source_path: str) -> None:
        self._source_path = Path(source_path)

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        incident_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        roadwork_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        temporal_access_matrix = [[True for _ in range(size)] for _ in range(size)]

        payload = json.loads(self._source_path.read_text(encoding="utf-8"))
        edges = payload.get("edges", payload) if isinstance(payload, dict) else payload
        if not isinstance(edges, list):
            raise ValueError("Road event JSON must contain an edges list")

        labels = {point.label: idx for idx, point in enumerate(points) if point.label}
        for raw_edge in edges:
            if not isinstance(raw_edge, dict) or not self._is_active(raw_edge, observed_at):
                continue
            from_idx = self._resolve_edge_index(raw_edge, "from", "from_index", "from_label", labels)
            to_idx = self._resolve_edge_index(raw_edge, "to", "to_index", "to_label", labels)
            if from_idx is None or to_idx is None:
                continue
            if not (0 <= from_idx < size and 0 <= to_idx < size) or from_idx == to_idx:
                continue
            self._apply_edge(raw_edge, from_idx, to_idx, incident_matrix, roadwork_matrix, temporal_access_matrix)
            if bool(raw_edge.get("bidirectional", False)):
                self._apply_edge(raw_edge, to_idx, from_idx, incident_matrix, roadwork_matrix, temporal_access_matrix)

        return RoadEventSnapshot(
            incident_risk_matrix=incident_matrix,
            roadwork_risk_matrix=roadwork_matrix,
            temporal_access_matrix=temporal_access_matrix,
            source="road-events-json",
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
        incident_matrix: list[list[float]],
        roadwork_matrix: list[list[float]],
        temporal_access_matrix: list[list[bool]],
    ) -> None:
        incident_matrix[from_idx][to_idx] = cls._bounded_risk(
            edge,
            "incident_risk",
            "accident_risk",
            "incident",
        )
        roadwork_matrix[from_idx][to_idx] = cls._bounded_risk(
            edge,
            "roadwork_risk",
            "repair_risk",
            "roadwork",
        )
        if "temporal_accessible" in edge:
            temporal_access_matrix[from_idx][to_idx] = bool(edge["temporal_accessible"])
        elif "accessible" in edge:
            temporal_access_matrix[from_idx][to_idx] = bool(edge["accessible"])
        elif "closed" in edge:
            temporal_access_matrix[from_idx][to_idx] = not bool(edge["closed"])

    @staticmethod
    def _bounded_risk(edge: dict[str, object], *keys: str) -> float:
        for key in keys:
            value = edge.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, int | float):
                return max(0.0, min(1.0, float(value)))
        return 0.0

    @staticmethod
    def _is_active(edge: dict[str, object], observed_at: datetime) -> bool:
        active_from = JsonRoadEventRepository._parse_datetime(edge.get("active_from", edge.get("start_at")))
        active_to = JsonRoadEventRepository._parse_datetime(edge.get("active_to", edge.get("end_at")))
        if active_from is not None and observed_at < active_from:
            return False
        if active_to is not None and observed_at > active_to:
            return False
        hours = edge.get("hours")
        if isinstance(hours, list) and hours:
            try:
                return observed_at.hour in {int(hour) for hour in hours}
            except (TypeError, ValueError):
                return True
        return True

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed


class SyntheticRoadEventRepository(RoadEventRepository):
    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        incident_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        roadwork_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        temporal_access_matrix = [[True for _ in range(size)] for _ in range(size)]
        if size == 0:
            return RoadEventSnapshot(
                incident_risk_matrix=incident_matrix,
                roadwork_risk_matrix=roadwork_matrix,
                temporal_access_matrix=temporal_access_matrix,
                source="synthetic-road-events",
                source_url=None,
                observed_at=observed_at,
            )

        centroid = Point(
            lat=sum(point.lat for point in points) / size,
            lon=sum(point.lon for point in points) / size,
        )
        peak_factor = self._peak_factor(observed_at.hour)
        night_factor = 1.0 if observed_at.hour >= 22 or observed_at.hour <= 5 else 0.0
        roadwork_season = 1.0 if observed_at.month in {4, 5, 6, 7, 8, 9, 10} else 0.25
        daylight_roadwork = 1.0 if 8 <= observed_at.hour <= 19 else 0.25

        for i, start in enumerate(points):
            for j, end in enumerate(points):
                if i == j:
                    continue
                distance = haversine_km(start, end)
                midpoint = Point(lat=(start.lat + end.lat) / 2.0, lon=(start.lon + end.lon) / 2.0)
                centroid_pressure = 1.0 - min(1.0, haversine_km(midpoint, centroid) / 40.0)
                local_segment = 1.0 - min(1.0, distance / 32.0)
                long_corridor = min(1.0, distance / 180.0)
                incident_noise = 0.5 + 0.5 * math.sin((start.lat * 21.0) - (end.lon * 15.0))
                roadwork_noise = 0.5 + 0.5 * math.cos((midpoint.lat * 12.0) + (midpoint.lon * 4.0))

                incident = (
                    0.02
                    + (0.16 * peak_factor)
                    + (0.11 * centroid_pressure)
                    + (0.08 * local_segment)
                    + (0.08 * night_factor)
                    + (0.06 * incident_noise)
                )
                roadwork = (
                    0.02
                    + (0.14 * roadwork_season * daylight_roadwork)
                    + (0.08 * long_corridor)
                    + (0.06 * roadwork_noise)
                )
                incident_matrix[i][j] = max(0.0, min(0.85, incident))
                roadwork_matrix[i][j] = max(0.0, min(0.75, roadwork))

        return RoadEventSnapshot(
            incident_risk_matrix=incident_matrix,
            roadwork_risk_matrix=roadwork_matrix,
            temporal_access_matrix=temporal_access_matrix,
            source="synthetic-road-events",
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
            return 0.20
        return 0.30


class OverpassRoadEventRepository(RoadEventRepository):
    def __init__(self, client: OverpassRoadDataClient) -> None:
        self._client = client

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        attributes = await self._client.fetch(points, departure_at)
        size = len(points)
        incident_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        roadwork_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        temporal_access_matrix = [[True for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                incident_matrix[i][j] = max(
                    attributes.incident_risk_by_point[i],
                    attributes.incident_risk_by_point[j],
                )
                roadwork_matrix[i][j] = max(
                    attributes.roadwork_risk_by_point[i],
                    attributes.roadwork_risk_by_point[j],
                )
                temporal_access_matrix[i][j] = attributes.access_by_point[i] and attributes.access_by_point[j]
        return RoadEventSnapshot(
            incident_risk_matrix=incident_matrix,
            roadwork_risk_matrix=roadwork_matrix,
            temporal_access_matrix=temporal_access_matrix,
            source=attributes.source if attributes.source != "overpass-osm" else "overpass-osm-road-events",
            source_url=attributes.source_url,
            observed_at=attributes.observed_at,
        )


class TomTomIncidentRepository(RoadEventRepository):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        bbox_padding_km: float = 4.0,
        match_radius_km: float = 2.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._bbox_padding_km = max(0.0, bbox_padding_km)
        self._match_radius_km = max(0.1, match_radius_km)

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        size = len(points)
        incident_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        roadwork_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        temporal_access_matrix = [[True for _ in range(size)] for _ in range(size)]
        if size < 2:
            return RoadEventSnapshot(
                incident_risk_matrix=incident_matrix,
                roadwork_risk_matrix=roadwork_matrix,
                temporal_access_matrix=temporal_access_matrix,
                source="tomtom-incidents",
                source_url=self._endpoint,
                observed_at=observed_at,
            )

        params = {
            "key": self._api_key,
            "bbox": self._bbox(points),
            "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description,code}}}}",
            "language": "en-GB",
            "timeValidityFilter": "present",
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True) as client:
            response = await client.get(self._endpoint, params=params)
        response.raise_for_status()
        incidents = response.json().get("incidents") or []
        parsed = [self._parse_incident(item) for item in incidents if isinstance(item, dict)]
        parsed = [item for item in parsed if item is not None]

        for i, start in enumerate(points):
            for j, end in enumerate(points):
                if i == j:
                    continue
                midpoint = Point(lat=(start.lat + end.lat) / 2.0, lon=(start.lon + end.lon) / 2.0)
                for incident_points, risk, is_roadwork, is_closed in parsed:
                    nearest = self._nearest_distance_km(midpoint, incident_points)
                    if nearest > self._match_radius_km:
                        continue
                    scaled_risk = risk * (1.0 - min(1.0, nearest / self._match_radius_km))
                    if is_roadwork:
                        roadwork_matrix[i][j] = max(roadwork_matrix[i][j], scaled_risk)
                    else:
                        incident_matrix[i][j] = max(incident_matrix[i][j], scaled_risk)
                    if is_closed:
                        temporal_access_matrix[i][j] = False

        return RoadEventSnapshot(
            incident_risk_matrix=incident_matrix,
            roadwork_risk_matrix=roadwork_matrix,
            temporal_access_matrix=temporal_access_matrix,
            source="tomtom-incidents",
            source_url=self._endpoint,
            observed_at=observed_at,
        )

    @property
    def _endpoint(self) -> str:
        return f"{self._base_url}/traffic/services/5/incidentDetails"

    def _bbox(self, points: list[Point]) -> str:
        min_lat = min(point.lat for point in points)
        max_lat = max(point.lat for point in points)
        min_lon = min(point.lon for point in points)
        max_lon = max(point.lon for point in points)
        mean_lat = sum(point.lat for point in points) / len(points)
        lat_pad = self._bbox_padding_km / 111.0
        lon_pad = self._bbox_padding_km / max(1.0, 111.0 * math.cos(math.radians(mean_lat)))
        return ",".join(
            [
                f"{min_lon - lon_pad:.7f}",
                f"{min_lat - lat_pad:.7f}",
                f"{max_lon + lon_pad:.7f}",
                f"{max_lat + lat_pad:.7f}",
            ]
        )

    @classmethod
    def _parse_incident(cls, incident: dict[str, object]) -> tuple[list[Point], float, bool, bool] | None:
        points = cls._geometry_points(incident.get("geometry"))
        if not points:
            return None
        properties = incident.get("properties") if isinstance(incident.get("properties"), dict) else {}
        text = cls._incident_text(properties)
        magnitude = properties.get("magnitudeOfDelay") if isinstance(properties, dict) else None
        risk = cls._risk_from_magnitude(magnitude)
        is_roadwork = any(token in text for token in ("roadwork", "road works", "construction", "maintenance"))
        is_closed = any(token in text for token in ("closed", "closure", "blocked"))
        return points, risk, is_roadwork, is_closed

    @classmethod
    def _geometry_points(cls, geometry: object) -> list[Point]:
        if not isinstance(geometry, dict):
            return []
        coordinates = geometry.get("coordinates")
        return cls._points_from_coordinates(coordinates)

    @classmethod
    def _points_from_coordinates(cls, value: object) -> list[Point]:
        if not isinstance(value, list):
            return []
        if len(value) >= 2 and isinstance(value[0], int | float) and isinstance(value[1], int | float):
            return [Point(lat=float(value[1]), lon=float(value[0]))]
        points: list[Point] = []
        for item in value:
            points.extend(cls._points_from_coordinates(item))
        return points

    @staticmethod
    def _incident_text(properties: object) -> str:
        if not isinstance(properties, dict):
            return ""
        chunks: list[str] = []
        icon = properties.get("iconCategory")
        if icon is not None:
            chunks.append(str(icon))
        events = properties.get("events")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    chunks.extend(str(value) for value in event.values() if value is not None)
        return " ".join(chunks).lower()

    @staticmethod
    def _risk_from_magnitude(magnitude: object) -> float:
        if isinstance(magnitude, int | float):
            return max(0.2, min(1.0, float(magnitude) / 4.0))
        return 0.55

    @staticmethod
    def _nearest_distance_km(point: Point, incident_points: list[Point]) -> float:
        if not incident_points:
            return float("inf")
        return min(haversine_km(point, incident_point) for incident_point in incident_points)


class CompositeRoadEventRepository(RoadEventRepository):
    def __init__(self, primary: RoadEventRepository | None, fallback: RoadEventRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, points: list[Point], departure_at: datetime | None = None) -> RoadEventSnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points, departure_at)
            except Exception:
                return await self._fallback.fetch(points, departure_at)
        return await self._fallback.fetch(points, departure_at)
