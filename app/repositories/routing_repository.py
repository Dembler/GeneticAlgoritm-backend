from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from app.domain.models import Point, TransportProfile
from app.services.distance import haversine_km, path_distance_km


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
        return MatrixProviderResult(distance_km=distance_km, duration_min=duration_min, provider="osrm-table")


class FallbackRoutingRepository(RoutingRepository):
    async def route(self, points: list[Point], profile: TransportProfile) -> RouteProviderResult:
        geometry = [[point.lat, point.lon] for point in points]
        duration_min = self._fallback_duration_minutes(path_distance_km(points), profile)
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
