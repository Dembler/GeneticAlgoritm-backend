from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.models import DataSourceInfo, Point, RouteRequest
from app.repositories.elevation_repository import ElevationProfile, ElevationRepository
from app.repositories.routing_repository import RoutingRepository
from app.repositories.toll_repository import TollRepository
from app.repositories.traffic_repository import TrafficRepository
from app.repositories.weather_repository import WeatherRepository, WeatherSnapshot


@dataclass
class OptimizationContext:
    points: list[Point]
    distance_matrix_km: list[list[float]]
    duration_matrix_min: list[list[float]]
    traffic_matrix: list[list[float]]
    toll_matrix: list[list[float]]
    weather: WeatherSnapshot
    elevation: ElevationProfile
    departure_at: datetime
    data_sources: DataSourceInfo
    matrix_provider: str

    def mean_congestion(self) -> float:
        values: list[float] = []
        for row in self.traffic_matrix:
            values.extend([v for v in row if v > 0])
        if not values:
            return 0.0
        return sum(values) / len(values)


class ContextService:
    def __init__(
        self,
        routing_repository: RoutingRepository,
        weather_repository: WeatherRepository,
        elevation_repository: ElevationRepository,
        traffic_repository: TrafficRepository,
        toll_repository: TollRepository,
    ) -> None:
        self._routing_repository = routing_repository
        self._weather_repository = weather_repository
        self._elevation_repository = elevation_repository
        self._traffic_repository = traffic_repository
        self._toll_repository = toll_repository

    async def build(self, request: RouteRequest) -> OptimizationContext:
        departure_at = request.departure_at or datetime.now(timezone.utc)
        points = list(request.points)
        if not points:
            empty_weather = await self._weather_repository.fetch(0.0, 0.0, departure_at)
            empty_elevation = await self._elevation_repository.fetch(points)
            return OptimizationContext(
                points=points,
                distance_matrix_km=[],
                duration_matrix_min=[],
                traffic_matrix=[],
                toll_matrix=[],
                weather=empty_weather,
                elevation=empty_elevation,
                departure_at=departure_at,
                data_sources=DataSourceInfo(
                    routing="unknown",
                    matrix="unknown",
                    weather=empty_weather.source,
                    elevation=empty_elevation.source,
                    traffic="unknown",
                    tolls="unknown",
                    fuel_prices="unknown",
                ),
                matrix_provider="unknown",
            )

        lat = sum([p.lat for p in points]) / len(points)
        lon = sum([p.lon for p in points]) / len(points)

        matrix_task = self._routing_repository.matrix(points, request.profile)
        weather_task = self._weather_repository.fetch(lat, lon, departure_at)
        elevation_task = self._elevation_repository.fetch(points)
        traffic_task = self._traffic_repository.fetch(points, departure_at)
        toll_task = self._toll_repository.fetch(points, request.profile, request.vehicle_class, departure_at)

        matrix_result, weather, elevation, traffic, tolls = await asyncio.gather(
            matrix_task, weather_task, elevation_task, traffic_task, toll_task
        )
        traffic_matrix = self._coerce_size(traffic.congestion_matrix, len(points))
        toll_matrix = self._coerce_non_negative_size(tolls.toll_matrix, len(points))

        return OptimizationContext(
            points=points,
            distance_matrix_km=matrix_result.distance_km,
            duration_matrix_min=matrix_result.duration_min,
            traffic_matrix=traffic_matrix,
            toll_matrix=toll_matrix,
            weather=weather,
            elevation=elevation,
            departure_at=departure_at,
            data_sources=DataSourceInfo(
                routing="osrm+fallback",
                matrix=matrix_result.provider,
                weather=weather.source,
                elevation=elevation.source,
                traffic=traffic.source,
                tolls=tolls.source,
                fuel_prices="rosstat+fallback",
            ),
            matrix_provider=matrix_result.provider,
        )

    @staticmethod
    def _coerce_size(matrix: list[list[float]], size: int) -> list[list[float]]:
        if not matrix:
            return [[0.0 for _ in range(size)] for _ in range(size)]
        normalized: list[list[float]] = []
        for i in range(size):
            if i >= len(matrix):
                normalized.append([0.0 for _ in range(size)])
                continue
            row = matrix[i]
            row_fixed = [0.0 for _ in range(size)]
            for j in range(size):
                if j < len(row):
                    row_fixed[j] = max(0.0, min(1.0, float(row[j])))
            normalized.append(row_fixed)
        return normalized

    @staticmethod
    def _coerce_non_negative_size(matrix: list[list[float]], size: int) -> list[list[float]]:
        if not matrix:
            return [[0.0 for _ in range(size)] for _ in range(size)]
        normalized: list[list[float]] = []
        for i in range(size):
            if i >= len(matrix):
                normalized.append([0.0 for _ in range(size)])
                continue
            row = matrix[i]
            row_fixed = [0.0 for _ in range(size)]
            for j in range(size):
                if j < len(row):
                    row_fixed[j] = max(0.0, float(row[j]))
            normalized.append(row_fixed)
        return normalized
