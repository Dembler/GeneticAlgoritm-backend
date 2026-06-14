from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol
from urllib.parse import quote

import httpx

from app.domain.models import Point, TransportProfile
from app.services.distance import haversine_km, path_distance_km

logger = logging.getLogger(__name__)


class RoutingProviderError(RuntimeError):
    pass


@dataclass
class RouteProviderResult:
    geometry: list[list[float]]
    distance_km: float
    duration_min: float | None
    provider: str


@dataclass
class MatrixProviderResult:
    distance_km: list[list[float]]
    duration_min: list[list[float]]
    provider: str


class RoutingRepository(Protocol):
    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        raise NotImplementedError

    async def matrix(self, points: list[Point], profile: TransportProfile) -> MatrixProviderResult:
        raise NotImplementedError


class OsrmRoutingRepository(RoutingRepository):
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        if len(points) < 2:
            raise RoutingProviderError("At least two points required for routing.")

        coords = ";".join([f"{point.lon},{point.lat}" for point in points])
        url = f"{self._base_url}/route/v1/{profile.value}/{coords}"
        params = {"overview": "full", "geometries": "geojson"}

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(url, params=params)

        if response.status_code != 200:
            raise RoutingProviderError(f"OSRM returned status {response.status_code}.")

        payload = response.json()
        routes = payload.get("routes")
        if not routes:
            raise RoutingProviderError("OSRM returned no routes.")

        route = routes[0]
        geometry = route.get("geometry", {}).get("coordinates", [])
        if not geometry:
            raise RoutingProviderError("OSRM geometry is empty.")

        latlon = [[coord[1], coord[0]] for coord in geometry]
        distance_km = float(route.get("distance", 0.0)) / 1000
        duration_min = float(route.get("duration", 0.0)) / 60 if "duration" in route else None
        logger.warning(
            "OSRM route raw: profile=%s points=%d geometry_points=%d distance_km=%.3f duration_min=%s sample_geometry=%s",
            profile.value,
            len(points),
            len(latlon),
            distance_km,
            None if duration_min is None else round(duration_min, 3),
            latlon[: min(5, len(latlon))],
        )

        return RouteProviderResult(
            geometry=latlon,
            distance_km=distance_km,
            duration_min=duration_min,
            provider="osrm",
        )

    async def matrix(self, points: list[Point], profile: TransportProfile) -> MatrixProviderResult:
        if not points:
            return MatrixProviderResult(distance_km=[], duration_min=[], provider="osrm")

        coords = ";".join([f"{point.lon},{point.lat}" for point in points])
        url = f"{self._base_url}/table/v1/{profile.value}/{coords}"
        params = {"annotations": "distance,duration"}

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            raise RoutingProviderError(f"OSRM table returned status {response.status_code}.")

        payload = response.json()
        distance_m = payload.get("distances")
        duration_s = payload.get("durations")
        if not distance_m or not duration_s:
            raise RoutingProviderError("OSRM table has empty distance or duration matrix.")

        distance_km = [[float(value or 0.0) / 1000 for value in row] for row in distance_m]
        duration_min = [[float(value or 0.0) / 60 for value in row] for row in duration_s]
        logger.warning(
            "OSRM table raw: profile=%s points=%d shape=%sx%s distance_row0=%s duration_row0=%s",
            profile.value,
            len(points),
            len(distance_km),
            len(distance_km[0]) if distance_km else 0,
            [round(float(value), 3) for value in (distance_km[0][: min(4, len(distance_km[0]))] if distance_km else [])],
            [round(float(value), 3) for value in (duration_min[0][: min(4, len(duration_min[0]))] if duration_min else [])],
        )
        return MatrixProviderResult(distance_km=distance_km, duration_min=duration_min, provider="osrm-table")


class OpenRouteServiceRoutingRepository(RoutingRepository):
    _PROFILE_MAP = {
        TransportProfile.driving: "driving-car",
        TransportProfile.walking: "foot-walking",
        TransportProfile.cycling: "cycling-regular",
    }

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        if len(points) < 2:
            raise RoutingProviderError("At least two points required for routing.")

        ors_profile = self._profile(profile)
        url = f"{self._base_url}/v2/directions/{ors_profile}/geojson"
        payload = {"coordinates": [[point.lon, point.lat] for point in points]}
        headers = {"Authorization": self._api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RoutingProviderError(f"OpenRouteService returned status {response.status_code}.")

        data = response.json()
        features = data.get("features") or []
        if not features:
            raise RoutingProviderError("OpenRouteService returned no route features.")

        feature = features[0]
        coordinates = feature.get("geometry", {}).get("coordinates") or []
        if not coordinates:
            raise RoutingProviderError("OpenRouteService geometry is empty.")
        summary = feature.get("properties", {}).get("summary") or {}
        latlon = [[coord[1], coord[0]] for coord in coordinates]
        distance_km = float(summary.get("distance", 0.0)) / 1000.0
        duration_min = float(summary.get("duration", 0.0)) / 60.0 if "duration" in summary else None

        logger.warning(
            "OpenRouteService route raw: profile=%s points=%d geometry_points=%d distance_km=%.3f duration_min=%s",
            ors_profile,
            len(points),
            len(latlon),
            distance_km,
            None if duration_min is None else round(duration_min, 3),
        )
        return RouteProviderResult(
            geometry=latlon,
            distance_km=distance_km,
            duration_min=duration_min,
            provider="openrouteservice",
        )

    async def matrix(self, points: list[Point], profile: TransportProfile) -> MatrixProviderResult:
        if not points:
            return MatrixProviderResult(distance_km=[], duration_min=[], provider="openrouteservice")

        ors_profile = self._profile(profile)
        url = f"{self._base_url}/v2/matrix/{ors_profile}"
        payload = {
            "locations": [[point.lon, point.lat] for point in points],
            "metrics": ["distance", "duration"],
            "units": "km",
        }
        headers = {"Authorization": self._api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RoutingProviderError(f"OpenRouteService matrix returned status {response.status_code}.")

        data = response.json()
        raw_distances = data.get("distances")
        raw_durations = data.get("durations")
        if not raw_distances or not raw_durations:
            raise RoutingProviderError("OpenRouteService matrix has empty distance or duration matrix.")

        distance_km = [[float(value or 0.0) for value in row] for row in raw_distances]
        duration_min = [[float(value or 0.0) / 60.0 for value in row] for row in raw_durations]
        logger.warning(
            "OpenRouteService matrix raw: profile=%s points=%d shape=%sx%s distance_row0=%s duration_row0=%s",
            ors_profile,
            len(points),
            len(distance_km),
            len(distance_km[0]) if distance_km else 0,
            [round(float(value), 3) for value in (distance_km[0][: min(4, len(distance_km[0]))] if distance_km else [])],
            [round(float(value), 3) for value in (duration_min[0][: min(4, len(duration_min[0]))] if duration_min else [])],
        )
        return MatrixProviderResult(
            distance_km=distance_km,
            duration_min=duration_min,
            provider="openrouteservice-matrix",
        )

    @classmethod
    def _profile(cls, profile: TransportProfile) -> str:
        return cls._PROFILE_MAP.get(profile, "driving-car")


class TomTomRoutingRepository(RoutingRepository):
    _PROFILE_MAP = {
        TransportProfile.driving: "car",
        TransportProfile.walking: "pedestrian",
        TransportProfile.cycling: "bicycle",
    }
    _MATRIX_PROFILE_MAP = {
        TransportProfile.driving: "car",
        TransportProfile.walking: "pedestrian",
    }

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        matrix_max_cells: int = 200,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._matrix_max_cells = max(1, matrix_max_cells)

    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        if len(points) < 2:
            raise RoutingProviderError("At least two points required for routing.")

        locations = ":".join(f"{point.lat},{point.lon}" for point in points)
        url = f"{self._base_url}/routing/1/calculateRoute/{quote(locations, safe=':,')}/json"
        params = {
            "key": self._api_key,
            "travelMode": self._profile(profile),
            "traffic": "true",
            "routeRepresentation": "polyline",
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True) as client:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            raise RoutingProviderError(f"TomTom routing returned status {response.status_code}.")

        payload = response.json()
        routes = payload.get("routes") or []
        if not routes:
            raise RoutingProviderError("TomTom routing returned no routes.")

        route = routes[0]
        summary = route.get("summary") or {}
        geometry = self._route_points(route)
        if not geometry:
            raise RoutingProviderError("TomTom routing geometry is empty.")

        distance_km = float(summary.get("lengthInMeters", 0.0)) / 1000.0
        duration_min = (
            float(summary["travelTimeInSeconds"]) / 60.0
            if isinstance(summary.get("travelTimeInSeconds"), int | float)
            else None
        )
        logger.warning(
            "TomTom route raw: profile=%s points=%d geometry_points=%d distance_km=%.3f duration_min=%s",
            profile.value,
            len(points),
            len(geometry),
            distance_km,
            None if duration_min is None else round(duration_min, 3),
        )
        return RouteProviderResult(
            geometry=geometry,
            distance_km=distance_km,
            duration_min=duration_min,
            provider="tomtom-routing",
        )

    async def matrix(self, points: list[Point], profile: TransportProfile) -> MatrixProviderResult:
        if not points:
            return MatrixProviderResult(distance_km=[], duration_min=[], provider="tomtom-matrix")
        if profile not in self._MATRIX_PROFILE_MAP:
            raise RoutingProviderError(f"TomTom matrix does not support profile {profile.value}.")
        if len(points) * len(points) > self._matrix_max_cells:
            raise RoutingProviderError("TomTom matrix request exceeds configured max cell count.")

        url = f"{self._base_url}/routing/matrix/2"
        payload = {
            "origins": [{"point": {"latitude": point.lat, "longitude": point.lon}} for point in points],
            "destinations": [{"point": {"latitude": point.lat, "longitude": point.lon}} for point in points],
            "options": {
                "departAt": "now",
                "routeType": "fastest",
                "traffic": "live",
                "travelMode": self._MATRIX_PROFILE_MAP[profile],
            },
        }
        async with httpx.AsyncClient(timeout=max(self._timeout_seconds, 20.0), follow_redirects=True) as client:
            response = await client.post(url, params={"key": self._api_key}, json=payload)
        if response.status_code != 200:
            raise RoutingProviderError(f"TomTom matrix returned status {response.status_code}.")

        data = response.json().get("data") or []
        size = len(points)
        distance_km = [[0.0 for _ in range(size)] for _ in range(size)]
        duration_min = [[0.0 for _ in range(size)] for _ in range(size)]
        completed_cells: set[tuple[int, int]] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            origin = item.get("originIndex")
            destination = item.get("destinationIndex")
            summary = item.get("routeSummary") or {}
            if not isinstance(origin, int) or not isinstance(destination, int):
                continue
            if not (0 <= origin < size and 0 <= destination < size):
                continue
            if "lengthInMeters" not in summary or "travelTimeInSeconds" not in summary:
                continue
            distance_km[origin][destination] = float(summary.get("lengthInMeters") or 0.0) / 1000.0
            duration_min[origin][destination] = float(summary.get("travelTimeInSeconds") or 0.0) / 60.0
            completed_cells.add((origin, destination))

        expected = {(i, j) for i in range(size) for j in range(size)}
        if not expected.issubset(completed_cells):
            raise RoutingProviderError("TomTom matrix returned incomplete matrix.")

        logger.warning(
            "TomTom matrix raw: profile=%s points=%d shape=%sx%s distance_row0=%s duration_row0=%s",
            profile.value,
            size,
            len(distance_km),
            len(distance_km[0]) if distance_km else 0,
            [round(float(value), 3) for value in (distance_km[0][: min(4, len(distance_km[0]))] if distance_km else [])],
            [round(float(value), 3) for value in (duration_min[0][: min(4, len(duration_min[0]))] if duration_min else [])],
        )
        return MatrixProviderResult(distance_km=distance_km, duration_min=duration_min, provider="tomtom-matrix")

    @classmethod
    def _profile(cls, profile: TransportProfile) -> str:
        return cls._PROFILE_MAP.get(profile, "car")

    @staticmethod
    def _route_points(route: dict[str, object]) -> list[list[float]]:
        geometry: list[list[float]] = []
        for leg in route.get("legs") or []:
            if not isinstance(leg, dict):
                continue
            for point in leg.get("points") or []:
                if not isinstance(point, dict):
                    continue
                lat = point.get("latitude")
                lon = point.get("longitude")
                if isinstance(lat, int | float) and isinstance(lon, int | float):
                    geometry.append([float(lat), float(lon)])
        return geometry


class FallbackRoutingRepository(RoutingRepository):
    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        geometry = [[point.lat, point.lon] for point in points]
        duration_min = self._fallback_duration_minutes(path_distance_km(points), profile)
        logger.warning(
            "Fallback route used: profile=%s points=%d distance_km=%.3f duration_min=%.3f geometry=%s",
            profile.value,
            len(points),
            path_distance_km(points),
            duration_min,
            geometry,
        )
        return RouteProviderResult(
            geometry=geometry,
            distance_km=path_distance_km(points),
            duration_min=duration_min,
            provider="fallback",
        )

    async def matrix(self, points: list[Point], profile: TransportProfile) -> MatrixProviderResult:
        size = len(points)
        distance = [[0.0 for _ in range(size)] for _ in range(size)]
        duration = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                d = haversine_km(points[i], points[j])
                distance[i][j] = d
                duration[i][j] = self._fallback_duration_minutes(d, profile)
        logger.warning(
            "Fallback matrix used: profile=%s points=%d shape=%sx%s distance_row0=%s duration_row0=%s",
            profile.value,
            size,
            len(distance),
            len(distance[0]) if distance else 0,
            [round(float(value), 3) for value in (distance[0][: min(4, len(distance[0]))] if distance else [])],
            [round(float(value), 3) for value in (duration[0][: min(4, len(duration[0]))] if duration else [])],
        )
        return MatrixProviderResult(distance_km=distance, duration_min=duration, provider="fallback-matrix")

    @staticmethod
    def _fallback_duration_minutes(distance_km: float, profile: TransportProfile) -> float:
        if profile == TransportProfile.walking:
            speed_kph = 5.0
        elif profile == TransportProfile.cycling:
            speed_kph = 15.0
        else:
            speed_kph = 42.0
        if distance_km <= 0:
            return 0.0
        return (distance_km / speed_kph) * 60.0


class CompositeRoutingRepository(RoutingRepository):
    def __init__(
        self,
        primary: RoutingRepository | None,
        fallback: RoutingRepository,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        if self._primary is not None:
            try:
                return await self._primary.route(points, profile)
            except Exception:
                return await self._fallback.route(points, profile)
        return await self._fallback.route(points, profile)

    async def matrix(self, points: list[Point], profile: TransportProfile) -> MatrixProviderResult:
        if self._primary is not None:
            try:
                return await self._primary.matrix(points, profile)
            except Exception:
                return await self._fallback.matrix(points, profile)
        return await self._fallback.matrix(points, profile)
