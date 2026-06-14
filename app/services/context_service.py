from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

from app.domain.models import DataSourceInfo, Point, RouteRequest, SegmentAlternativeSet, SegmentAlternativesSummary, SegmentCandidate
from app.repositories.elevation_repository import ElevationProfile, ElevationRepository
from app.repositories.infrastructure_repository import InfrastructureRepository
from app.repositories.road_event_repository import RoadEventRepository
from app.repositories.road_quality_repository import RoadQualityRepository
from app.repositories.routing_repository import RoutingRepository
from app.repositories.toll_repository import TollRepository
from app.repositories.traffic_repository import TrafficRepository
from app.repositories.weather_repository import WeatherProfile, WeatherRepository, WeatherSnapshot
from app.services.terrain_profile_service import TerrainProfileService

logger = logging.getLogger(__name__)


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
    weather_profiles: list[WeatherProfile] = field(default_factory=list)
    elevation_gain_matrix_m: list[list[float]] = field(default_factory=list)
    elevation_loss_matrix_m: list[list[float]] = field(default_factory=list)
    mean_elevation_matrix_m: list[list[float]] = field(default_factory=list)
    height_clearance_matrix_m: list[list[float | None]] = field(default_factory=list)
    weight_limit_matrix_t: list[list[float | None]] = field(default_factory=list)
    width_limit_matrix_m: list[list[float | None]] = field(default_factory=list)
    length_limit_matrix_m: list[list[float | None]] = field(default_factory=list)
    infrastructure_access_matrix: list[list[bool]] = field(default_factory=list)
    surface_quality_matrix: list[list[float]] = field(default_factory=list)
    incident_risk_matrix: list[list[float]] = field(default_factory=list)
    roadwork_risk_matrix: list[list[float]] = field(default_factory=list)
    temporal_access_matrix: list[list[bool]] = field(default_factory=list)
    segment_alternatives: dict[tuple[int, int], SegmentAlternativeSet] = field(default_factory=dict)
    best_segment_score_matrix: list[list[float]] = field(default_factory=list)
    best_segment_distance_matrix_km: list[list[float]] = field(default_factory=list)
    best_segment_duration_matrix_min: list[list[float]] = field(default_factory=list)
    best_segment_choice_matrix: list[list[SegmentCandidate | None]] = field(default_factory=list)
    segment_alternatives_enabled: bool = False
    segment_alternatives_summary: SegmentAlternativesSummary = field(default_factory=SegmentAlternativesSummary)

    def mean_congestion(self) -> float:
        values: list[float] = []
        for row in self.traffic_matrix:
            values.extend([v for v in row if v > 0])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def mean_dynamic_event_risk(self) -> float:
        values: list[float] = []
        for incident_row, roadwork_row in zip(self.incident_risk_matrix, self.roadwork_risk_matrix, strict=False):
            for incident, roadwork in zip(incident_row, roadwork_row, strict=False):
                value = max(float(incident), float(roadwork))
                if value > 0:
                    values.append(value)
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
        road_quality_repository: RoadQualityRepository,
        road_event_repository: RoadEventRepository,
        infrastructure_repository: InfrastructureRepository,
        terrain_profile_service: TerrainProfileService,
    ) -> None:
        self._routing_repository = routing_repository
        self._weather_repository = weather_repository
        self._elevation_repository = elevation_repository
        self._traffic_repository = traffic_repository
        self._toll_repository = toll_repository
        self._road_quality_repository = road_quality_repository
        self._road_event_repository = road_event_repository
        self._infrastructure_repository = infrastructure_repository
        self._terrain_profile_service = terrain_profile_service

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
                weather_profiles=[],
                elevation=empty_elevation,
                elevation_gain_matrix_m=[],
                elevation_loss_matrix_m=[],
                mean_elevation_matrix_m=[],
                height_clearance_matrix_m=[],
                weight_limit_matrix_t=[],
                width_limit_matrix_m=[],
                length_limit_matrix_m=[],
                infrastructure_access_matrix=[],
                surface_quality_matrix=[],
                incident_risk_matrix=[],
                roadwork_risk_matrix=[],
                temporal_access_matrix=[],
                departure_at=departure_at,
                data_sources=DataSourceInfo(
                    routing="unknown",
                    matrix="unknown",
                    weather=empty_weather.source,
                    elevation=empty_elevation.source,
                    traffic="unknown",
                    tolls="unknown",
                    fuel_prices="unknown",
                    infrastructure="unknown",
                    road_quality="unknown",
                    road_events="unknown",
                ),
                matrix_provider="unknown",
            )

        matrix_task = self._routing_repository.matrix(points, request.profile)
        weather_profiles_task = asyncio.gather(
            *(self._weather_repository.fetch_profile(point.lat, point.lon, departure_at) for point in points)
        )
        elevation_task = self._elevation_repository.fetch(points)
        terrain_task = self._terrain_profile_service.build_edge_matrices(points)
        traffic_task = self._traffic_repository.fetch(points, departure_at)
        toll_task = self._toll_repository.fetch(points, request.profile, request.vehicle_class, departure_at)
        road_quality_task = self._road_quality_repository.fetch(points, departure_at)
        road_events_task = self._road_event_repository.fetch(points, departure_at)
        infrastructure_task = self._infrastructure_repository.fetch(points, request.profile, departure_at)

        (
            matrix_result,
            weather_profiles,
            elevation,
            terrain_matrices,
            traffic,
            tolls,
            road_quality,
            road_events,
            infrastructure,
        ) = await asyncio.gather(
            matrix_task,
            weather_profiles_task,
            elevation_task,
            terrain_task,
            traffic_task,
            toll_task,
            road_quality_task,
            road_events_task,
            infrastructure_task,
        )
        weather = self._aggregate_weather_profiles(weather_profiles, departure_at)
        elevation_gain_matrix, elevation_loss_matrix, mean_elevation_matrix = terrain_matrices
        traffic_matrix = self._coerce_size(traffic.congestion_matrix, len(points))
        toll_matrix = self._coerce_non_negative_size(tolls.toll_matrix, len(points))
        surface_quality_matrix = self._coerce_quality_size(road_quality.surface_quality_matrix, len(points))
        incident_risk_matrix = self._coerce_size(road_events.incident_risk_matrix, len(points))
        roadwork_risk_matrix = self._coerce_size(road_events.roadwork_risk_matrix, len(points))
        temporal_access_matrix = self._coerce_bool_size(road_events.temporal_access_matrix, len(points))
        height_clearance_matrix = self._coerce_optional_positive_size(
            infrastructure.height_clearance_matrix_m,
            len(points),
        )
        weight_limit_matrix = self._coerce_optional_positive_size(infrastructure.weight_limit_matrix_t, len(points))
        width_limit_matrix = self._coerce_optional_positive_size(infrastructure.width_limit_matrix_m, len(points))
        length_limit_matrix = self._coerce_optional_positive_size(infrastructure.length_limit_matrix_m, len(points))
        infrastructure_access_matrix = self._coerce_bool_size(infrastructure.access_matrix, len(points))
        logger.warning(
            "Context debug summary: points=%d departure_at=%s matrix_provider=%s weather_source=%s weather_profile_sources=%s elevation_source=%s traffic_source=%s toll_source=%s road_quality_source=%s road_events_source=%s infrastructure_source=%s weather_profiles=%d weather_severity=%.3f temp=%s precip=%s wind=%s elevation_count=%d elevation_sample=%s distance_shape=%s duration_shape=%s traffic_shape=%s toll_shape=%s road_quality_shape=%s incident_shape=%s roadwork_shape=%s distance_row0=%s duration_row0=%s traffic_row0=%s toll_row0=%s road_quality_row0=%s incident_row0=%s roadwork_row0=%s gain_row0=%s loss_row0=%s mean_elev_row0=%s height_limit_row0=%s weight_limit_row0=%s access_row0=%s temporal_access_row0=%s",
            len(points),
            departure_at.isoformat(),
            matrix_result.provider,
            weather.source,
            [profile.source for profile in weather_profiles],
            elevation.source,
            traffic.source,
            tolls.source,
            road_quality.source,
            road_events.source,
            infrastructure.source,
            len(weather_profiles),
            weather.severity,
            weather.temperature_c,
            weather.precipitation_mm,
            weather.wind_speed_kph,
            len(elevation.elevations_m),
            elevation.elevations_m[: min(6, len(elevation.elevations_m))],
            self._shape(matrix_result.distance_km),
            self._shape(matrix_result.duration_min),
            self._shape(traffic_matrix),
            self._shape(toll_matrix),
            self._shape(surface_quality_matrix),
            self._shape(incident_risk_matrix),
            self._shape(roadwork_risk_matrix),
            self._sample_row(matrix_result.distance_km),
            self._sample_row(matrix_result.duration_min),
            self._sample_row(traffic_matrix),
            self._sample_row(toll_matrix),
            self._sample_row(surface_quality_matrix),
            self._sample_row(incident_risk_matrix),
            self._sample_row(roadwork_risk_matrix),
            self._sample_row(elevation_gain_matrix),
            self._sample_row(elevation_loss_matrix),
            self._sample_row(mean_elevation_matrix),
            self._sample_optional_row(height_clearance_matrix),
            self._sample_optional_row(weight_limit_matrix),
            infrastructure_access_matrix[0][:4] if infrastructure_access_matrix else [],
            temporal_access_matrix[0][:4] if temporal_access_matrix else [],
        )

        return OptimizationContext(
            points=points,
            distance_matrix_km=matrix_result.distance_km,
            duration_matrix_min=matrix_result.duration_min,
            traffic_matrix=traffic_matrix,
            toll_matrix=toll_matrix,
            weather=weather,
            weather_profiles=weather_profiles,
            elevation=elevation,
            elevation_gain_matrix_m=elevation_gain_matrix,
            elevation_loss_matrix_m=elevation_loss_matrix,
            mean_elevation_matrix_m=mean_elevation_matrix,
            height_clearance_matrix_m=height_clearance_matrix,
            weight_limit_matrix_t=weight_limit_matrix,
            width_limit_matrix_m=width_limit_matrix,
            length_limit_matrix_m=length_limit_matrix,
            infrastructure_access_matrix=infrastructure_access_matrix,
            surface_quality_matrix=surface_quality_matrix,
            incident_risk_matrix=incident_risk_matrix,
            roadwork_risk_matrix=roadwork_risk_matrix,
            temporal_access_matrix=temporal_access_matrix,
            departure_at=departure_at,
            data_sources=DataSourceInfo(
                routing="osrm+fallback",
                matrix=matrix_result.provider,
                weather=weather.source,
                elevation=elevation.source,
                traffic=traffic.source,
                tolls=tolls.source,
                fuel_prices="rosstat+fallback",
                infrastructure=infrastructure.source,
                road_quality=road_quality.source,
                road_events=road_events.source,
            ),
            matrix_provider=matrix_result.provider,
        )

    @staticmethod
    def _aggregate_weather_profiles(profiles: list[WeatherProfile], at: datetime) -> WeatherSnapshot:
        if not profiles:
            return WeatherSnapshot(
                severity=0.0,
                temperature_c=None,
                precipitation_mm=None,
                wind_speed_kph=None,
                source="unknown",
                source_url=None,
                observed_at=at,
            )

        snapshots = [profile.snapshot_at(at) for profile in profiles]
        severity = sum(item.severity for item in snapshots) / len(snapshots)
        temperatures = [item.temperature_c for item in snapshots if item.temperature_c is not None]
        precipitations = [item.precipitation_mm for item in snapshots if item.precipitation_mm is not None]
        winds = [item.wind_speed_kph for item in snapshots if item.wind_speed_kph is not None]
        sources = sorted({item.source for item in snapshots if item.source})
        source_urls = sorted({item.source_url for item in snapshots if item.source_url})
        return WeatherSnapshot(
            severity=severity,
            temperature_c=(sum(temperatures) / len(temperatures)) if temperatures else None,
            precipitation_mm=(sum(precipitations) / len(precipitations)) if precipitations else None,
            wind_speed_kph=(sum(winds) / len(winds)) if winds else None,
            source=("+".join(sources) if sources else "unknown"),
            source_url=source_urls[0] if len(source_urls) == 1 else None,
            observed_at=at,
        )

    @staticmethod
    def _shape(matrix: list[list[float]]) -> tuple[int, int]:
        if not matrix:
            return 0, 0
        return len(matrix), len(matrix[0]) if matrix[0] else 0

    @staticmethod
    def _sample_row(matrix: list[list[float]], row_index: int = 0, limit: int = 4) -> list[float]:
        if not matrix or row_index >= len(matrix):
            return []
        row = matrix[row_index] or []
        return [round(float(value), 4) for value in row[:limit]]

    @staticmethod
    def _sample_optional_row(
        matrix: list[list[float | None]],
        row_index: int = 0,
        limit: int = 4,
    ) -> list[float | None]:
        if not matrix or row_index >= len(matrix):
            return []
        row = matrix[row_index] or []
        return [None if value is None else round(float(value), 4) for value in row[:limit]]

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

    @staticmethod
    def _coerce_quality_size(matrix: list[list[float]], size: int) -> list[list[float]]:
        if not matrix:
            return [[1.0 for _ in range(size)] for _ in range(size)]
        normalized: list[list[float]] = []
        for i in range(size):
            if i >= len(matrix):
                normalized.append([1.0 for _ in range(size)])
                continue
            row = matrix[i]
            row_fixed = [1.0 for _ in range(size)]
            for j in range(size):
                if j < len(row):
                    row_fixed[j] = max(0.0, min(1.0, float(row[j])))
            normalized.append(row_fixed)
        return normalized

    @staticmethod
    def _coerce_optional_positive_size(matrix: list[list[float | None]], size: int) -> list[list[float | None]]:
        if not matrix:
            return [[None for _ in range(size)] for _ in range(size)]
        normalized: list[list[float | None]] = []
        for i in range(size):
            if i >= len(matrix):
                normalized.append([None for _ in range(size)])
                continue
            row = matrix[i]
            row_fixed: list[float | None] = [None for _ in range(size)]
            for j in range(size):
                if j >= len(row) or row[j] is None:
                    continue
                value = float(row[j])
                row_fixed[j] = value if value > 0 else None
            normalized.append(row_fixed)
        return normalized

    @staticmethod
    def _coerce_bool_size(matrix: list[list[bool]], size: int) -> list[list[bool]]:
        if not matrix:
            return [[True for _ in range(size)] for _ in range(size)]
        normalized: list[list[bool]] = []
        for i in range(size):
            if i >= len(matrix):
                normalized.append([True for _ in range(size)])
                continue
            row = matrix[i]
            row_fixed = [True for _ in range(size)]
            for j in range(size):
                if j < len(row):
                    row_fixed[j] = bool(row[j])
            normalized.append(row_fixed)
        return normalized
