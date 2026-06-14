from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import re

import httpx

from app.domain.models import Point
from app.services.distance import haversine_km

logger = logging.getLogger(__name__)


@dataclass
class OverpassRoadAttributesSnapshot:
    surface_quality_by_point: list[float]
    incident_risk_by_point: list[float]
    roadwork_risk_by_point: list[float]
    access_by_point: list[bool]
    height_clearance_m_by_point: list[float | None]
    weight_limit_t_by_point: list[float | None]
    width_limit_m_by_point: list[float | None]
    length_limit_m_by_point: list[float | None]
    source: str
    source_url: str | None
    observed_at: datetime


class OverpassRoadDataClient:
    def __init__(
        self,
        base_url: str,
        *,
        radius_m: float = 900.0,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 900,
        user_agent: str = "RouteOptimizationLab/1.0 (contact: local)",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._radius_m = max(100.0, float(radius_m))
        self._timeout_seconds = timeout_seconds
        self._cache_ttl = timedelta(seconds=max(0, int(cache_ttl_seconds)))
        self._headers = {"User-Agent": user_agent}
        self._cache: dict[str, tuple[datetime, OverpassRoadAttributesSnapshot]] = {}
        self._lock = asyncio.Lock()

    async def fetch(
        self,
        points: list[Point],
        departure_at: datetime | None = None,
    ) -> OverpassRoadAttributesSnapshot:
        observed_at = departure_at or datetime.now(timezone.utc)
        if not points:
            return self._empty([], observed_at, source="overpass-osm-empty")

        key = self._cache_key(points)
        async with self._lock:
            cached = self._cache.get(key)
            now = datetime.now(timezone.utc)
            if cached is not None and now - cached[0] <= self._cache_ttl:
                return cached[1]

            snapshot = await self._fetch_uncached(points, observed_at)
            self._cache[key] = (now, snapshot)
            return snapshot

    async def _fetch_uncached(
        self,
        points: list[Point],
        observed_at: datetime,
    ) -> OverpassRoadAttributesSnapshot:
        query = self._build_query(points)
        try:
            async with httpx.AsyncClient(
                timeout=max(self._timeout_seconds, 20.0),
                headers=self._headers,
                follow_redirects=True,
            ) as client:
                response = await client.post(self._base_url, data={"data": query})
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Overpass road data fetch failed: %s", repr(exc))
            return self._empty(points, observed_at, source="overpass-osm-unavailable")

        elements = payload.get("elements") or []
        point_tags = self._tags_near_points(points, elements)
        snapshot = OverpassRoadAttributesSnapshot(
            surface_quality_by_point=[
                self._surface_quality(tags_list) for tags_list in point_tags
            ],
            incident_risk_by_point=[
                self._incident_risk(tags_list) for tags_list in point_tags
            ],
            roadwork_risk_by_point=[
                self._roadwork_risk(tags_list) for tags_list in point_tags
            ],
            access_by_point=[
                self._access_allowed(tags_list) for tags_list in point_tags
            ],
            height_clearance_m_by_point=[
                self._min_limit(tags_list, "maxheight", "maxheight:physical")
                for tags_list in point_tags
            ],
            weight_limit_t_by_point=[
                self._min_limit(tags_list, "maxweight", "maxgcweight")
                for tags_list in point_tags
            ],
            width_limit_m_by_point=[
                self._min_limit(tags_list, "maxwidth")
                for tags_list in point_tags
            ],
            length_limit_m_by_point=[
                self._min_limit(tags_list, "maxlength")
                for tags_list in point_tags
            ],
            source="overpass-osm",
            source_url=self._base_url,
            observed_at=observed_at,
        )
        logger.warning(
            "Overpass road data: points=%d elements=%d quality=%s roadwork=%s access=%s",
            len(points),
            len(elements),
            [round(value, 3) for value in snapshot.surface_quality_by_point[:4]],
            [round(value, 3) for value in snapshot.roadwork_risk_by_point[:4]],
            snapshot.access_by_point[:4],
        )
        return snapshot

    def _build_query(self, points: list[Point]) -> str:
        selectors = "\n".join(
            f'  way["highway"](around:{int(self._radius_m)},{point.lat:.6f},{point.lon:.6f});'
            for point in points
        )
        timeout = int(max(8.0, self._timeout_seconds))
        return f"""[out:json][timeout:{timeout}];
(
{selectors}
);
out tags center;"""

    def _tags_near_points(self, points: list[Point], elements: list[dict]) -> list[list[dict[str, str]]]:
        tags_by_point: list[list[dict[str, str]]] = [[] for _ in points]
        for element in elements:
            tags = element.get("tags") or {}
            center = element.get("center") or {}
            lat = center.get("lat")
            lon = center.get("lon")
            if not isinstance(tags, dict) or lat is None or lon is None:
                continue
            road_point = Point(lat=float(lat), lon=float(lon))
            normalized_tags = {str(key): str(value) for key, value in tags.items()}
            for idx, point in enumerate(points):
                if haversine_km(point, road_point) * 1000.0 <= self._radius_m * 1.75:
                    tags_by_point[idx].append(normalized_tags)
        return tags_by_point

    def _empty(
        self,
        points: list[Point],
        observed_at: datetime,
        *,
        source: str,
    ) -> OverpassRoadAttributesSnapshot:
        size = len(points)
        return OverpassRoadAttributesSnapshot(
            surface_quality_by_point=[1.0 for _ in range(size)],
            incident_risk_by_point=[0.0 for _ in range(size)],
            roadwork_risk_by_point=[0.0 for _ in range(size)],
            access_by_point=[True for _ in range(size)],
            height_clearance_m_by_point=[None for _ in range(size)],
            weight_limit_t_by_point=[None for _ in range(size)],
            width_limit_m_by_point=[None for _ in range(size)],
            length_limit_m_by_point=[None for _ in range(size)],
            source=source,
            source_url=self._base_url if source != "overpass-osm-empty" else None,
            observed_at=observed_at,
        )

    def _cache_key(self, points: list[Point]) -> str:
        coords = "|".join(f"{point.lat:.5f},{point.lon:.5f}" for point in points)
        return f"{int(self._radius_m)}:{coords}"

    @classmethod
    def _surface_quality(cls, tags_list: list[dict[str, str]]) -> float:
        if not tags_list:
            return 1.0
        road_tags = cls._driving_tags(tags_list) or tags_list
        values = [cls._surface_quality_for_tags(tags) for tags in road_tags]
        return max(0.0, min(1.0, sum(values) / len(values)))

    @staticmethod
    def _surface_quality_for_tags(tags: dict[str, str]) -> float:
        surface = tags.get("surface", "").lower()
        smoothness = tags.get("smoothness", "").lower()
        tracktype = tags.get("tracktype", "").lower()
        highway = tags.get("highway", "").lower()

        surface_scores = {
            "asphalt": 0.98,
            "concrete": 0.96,
            "concrete:plates": 0.9,
            "paved": 0.92,
            "paving_stones": 0.78,
            "sett": 0.72,
            "compacted": 0.72,
            "fine_gravel": 0.66,
            "gravel": 0.58,
            "pebblestone": 0.55,
            "ground": 0.46,
            "dirt": 0.4,
            "earth": 0.4,
            "sand": 0.32,
            "mud": 0.24,
            "unpaved": 0.45,
        }
        smoothness_scores = {
            "excellent": 1.0,
            "good": 0.92,
            "intermediate": 0.78,
            "bad": 0.52,
            "very_bad": 0.36,
            "horrible": 0.24,
            "very_horrible": 0.18,
            "impassable": 0.05,
        }
        track_scores = {
            "grade1": 0.86,
            "grade2": 0.68,
            "grade3": 0.5,
            "grade4": 0.34,
            "grade5": 0.22,
        }

        values: list[float] = []
        if surface in surface_scores:
            values.append(surface_scores[surface])
        if smoothness in smoothness_scores:
            values.append(smoothness_scores[smoothness])
        if tracktype in track_scores:
            values.append(track_scores[tracktype])
        if highway == "construction":
            values.append(0.22)
        return sum(values) / len(values) if values else 0.9

    @classmethod
    def _incident_risk(cls, tags_list: list[dict[str, str]]) -> float:
        risk = 0.0
        for tags in cls._driving_tags(tags_list):
            if "hazard" in tags:
                risk = max(risk, 0.35)
            if tags.get("traffic_calming"):
                risk = max(risk, 0.12)
            if tags.get("junction") in {"roundabout", "circular"}:
                risk = max(risk, 0.08)
            maxspeed = OverpassRoadDataClient._parse_number(tags.get("maxspeed"))
            if maxspeed is not None and 0 < maxspeed <= 30:
                risk = max(risk, 0.1)
        return risk

    @classmethod
    def _roadwork_risk(cls, tags_list: list[dict[str, str]]) -> float:
        risk = 0.0
        for tags in cls._driving_tags(tags_list):
            if tags.get("highway") == "construction" or tags.get("construction"):
                risk = max(risk, 0.8)
            if tags.get("access") in {"no", "private"} or tags.get("motor_vehicle") == "no":
                risk = max(risk, 0.45)
        return risk

    @classmethod
    def _access_allowed(cls, tags_list: list[dict[str, str]]) -> bool:
        road_tags = cls._driving_tags(tags_list)
        if not road_tags:
            return True
        return any(not cls._is_access_denied(tags) for tags in road_tags)

    @classmethod
    def _min_limit(cls, tags_list: list[dict[str, str]], *keys: str) -> float | None:
        values: list[float] = []
        for tags in tags_list:
            for key in keys:
                parsed = cls._parse_number(tags.get(key))
                if parsed is not None and parsed > 0:
                    values.append(parsed)
        return min(values) if values else None

    @staticmethod
    def _parse_number(value: str | None) -> float | None:
        if not value:
            return None
        normalized = value.lower().replace(",", ".")
        if normalized in {"none", "unsigned", "signals", "variable"}:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)", normalized)
        if match is None:
            return None
        return float(match.group(1))

    @classmethod
    def _driving_tags(cls, tags_list: list[dict[str, str]]) -> list[dict[str, str]]:
        return [tags for tags in tags_list if cls._is_driving_highway(tags)]

    @staticmethod
    def _is_driving_highway(tags: dict[str, str]) -> bool:
        highway = tags.get("highway", "").lower()
        return highway in {
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "unclassified",
            "residential",
            "living_street",
            "service",
            "road",
            "construction",
        }

    @staticmethod
    def _is_access_denied(tags: dict[str, str]) -> bool:
        return (
            tags.get("access") in {"no", "private"}
            or tags.get("motor_vehicle") in {"no", "private"}
            or tags.get("vehicle") in {"no", "private"}
        )
