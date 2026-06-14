from __future__ import annotations

import math
import logging
from dataclasses import dataclass

from app.domain.models import Point, RouteTerrainProfile, RouteTerrainSegment, TerrainTrend
from app.repositories.elevation_repository import ElevationRepository
from app.services.distance import haversine_km


logger = logging.getLogger(__name__)


@dataclass
class EdgeTerrainStats:
    gain_m: float
    loss_m: float
    mean_elevation_m: float | None


@dataclass
class PolylineSample:
    point: Point
    edge_index: int
    edge_ratio: float


class TerrainProfileService:
    _EDGE_SAMPLE_SPACING_KM = 10.0
    _EDGE_SAMPLE_MAX_POINTS = 80
    _ROUTE_SAMPLE_SPACING_KM = 1.5
    _ROUTE_SAMPLE_MAX_POINTS = 240
    _FETCH_BATCH_SIZE = 80
    _FLAT_GRADE_THRESHOLD_PCT = 0.05
    _MERGED_SEGMENT_MAX_DISTANCE_KM = 0.28
    _MERGED_SEGMENT_MAX_GRADE_DELTA_PCT = 0.04

    def __init__(self, elevation_repository: ElevationRepository) -> None:
        self._elevation_repository = elevation_repository

    async def build_edge_matrices(
        self,
        points: list[Point],
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        size = len(points)
        gain_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        loss_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        mean_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        if size < 2:
            return gain_matrix, loss_matrix, mean_matrix
        spacing_km, max_points = self._edge_sampling_budget(size)

        point_index_lookup: dict[tuple[float, float], int] = {}
        unique_points: list[Point] = []
        edge_keys: dict[tuple[int, int], list[tuple[float, float]]] = {}

        for i in range(size):
            for j in range(i + 1, size):
                sample_points = self._sample_between_points(
                    points[i],
                    points[j],
                    spacing_km=spacing_km,
                    max_points=max_points,
                )
                keys: list[tuple[float, float]] = []
                for point in sample_points:
                    key = self._point_key(point)
                    if key not in point_index_lookup:
                        point_index_lookup[key] = len(unique_points)
                        unique_points.append(Point(lat=point.lat, lon=point.lon))
                    keys.append(key)
                edge_keys[(i, j)] = keys

        elevation_lookup, _source = await self._fetch_lookup(unique_points)

        for (i, j), keys in edge_keys.items():
            sample_points = [Point(lat=lat, lon=lon) for lat, lon in keys]
            elevations = [elevation_lookup.get(key, 0.0) for key in keys]
            stats = self._summarize_edge(sample_points, elevations)
            gain_matrix[i][j] = stats.gain_m
            loss_matrix[i][j] = stats.loss_m
            mean_matrix[i][j] = stats.mean_elevation_m or 0.0
            gain_matrix[j][i] = stats.loss_m
            loss_matrix[j][i] = stats.gain_m
            mean_matrix[j][i] = stats.mean_elevation_m or 0.0

        return gain_matrix, loss_matrix, mean_matrix

    async def build_route_profile(self, geometry: list[list[float]]) -> RouteTerrainProfile:
        if len(geometry) < 2:
            return RouteTerrainProfile(source="unknown")

        route_points = [Point(lat=float(lat), lon=float(lon)) for lat, lon in geometry]
        sampled_points = self._sample_geometry(
            route_points,
            spacing_km=self._ROUTE_SAMPLE_SPACING_KM,
            max_points=self._ROUTE_SAMPLE_MAX_POINTS,
        )
        elevation_lookup, source = await self._fetch_lookup([item.point for item in sampled_points])
        elevations = [elevation_lookup.get(self._point_key(item.point), 0.0) for item in sampled_points]
        logger.warning(
            "Terrain debug route profile input: geometry_points=%d sampled_points=%d source=%s sample_geometry=%s sample_elev=%s",
            len(route_points),
            len(sampled_points),
            source,
            [[round(point.lat, 6), round(point.lon, 6)] for point in route_points[:5]],
            [round(value, 2) for value in elevations[:8]],
        )
        return self._build_route_profile_from_samples(route_points, sampled_points, elevations, source)

    async def _fetch_lookup(self, points: list[Point]) -> tuple[dict[tuple[float, float], float], str]:
        if not points:
            return {}, "unknown"

        lookup: dict[tuple[float, float], float] = {}
        source = "unknown"
        for start in range(0, len(points), self._FETCH_BATCH_SIZE):
            chunk = points[start : start + self._FETCH_BATCH_SIZE]
            profile = await self._elevation_repository.fetch(chunk)
            if source == "unknown":
                source = profile.source
            for point, elevation in zip(chunk, profile.elevations_m, strict=False):
                lookup[self._point_key(point)] = float(elevation)
        return lookup, source

    def _build_route_profile_from_samples(
        self,
        route_points: list[Point],
        samples: list[PolylineSample],
        elevations: list[float],
        source: str,
    ) -> RouteTerrainProfile:
        if len(samples) < 2 or len(elevations) < 2:
            return RouteTerrainProfile(sampled_points=len(samples), source=source)

        smoothed_elevations = self._smooth_elevations(samples, elevations)
        logger.warning(
            "Terrain debug smoothing: samples=%d raw=%s smoothed=%s",
            len(samples),
            [round(float(value), 2) for value in elevations[:10]],
            [round(float(value), 2) for value in smoothed_elevations[:10]],
        )
        merged_segments: list[RouteTerrainSegment] = []
        total_gain = 0.0
        total_loss = 0.0
        max_uphill_grade = 0.0
        max_downhill_grade = 0.0

        for index in range(len(samples) - 1):
            start_sample = samples[index]
            end_sample = samples[index + 1]
            start = start_sample.point
            end = end_sample.point
            distance_km = haversine_km(start, end)
            if distance_km <= 0:
                continue
            raw_delta = float(elevations[index + 1]) - float(elevations[index])
            gain = max(raw_delta, 0.0)
            loss = max(-raw_delta, 0.0)
            trend_delta = float(smoothed_elevations[index + 1]) - float(smoothed_elevations[index])
            grade_pct = (trend_delta / (distance_km * 1000.0)) * 100.0
            trend = self._classify_trend(grade_pct)

            total_gain += gain
            total_loss += loss
            max_uphill_grade = max(max_uphill_grade, max(grade_pct, 0.0))
            max_downhill_grade = max(max_downhill_grade, max(-grade_pct, 0.0))

            segment = RouteTerrainSegment(
                trend=trend,
                geometry=self._slice_route_geometry(route_points, start_sample, end_sample),
                distance_km=distance_km,
                elevation_delta_m=raw_delta,
                elevation_gain_m=gain,
                elevation_loss_m=loss,
                grade_pct=grade_pct,
            )
            if merged_segments and self._should_merge_segments(merged_segments[-1], segment):
                previous = merged_segments[-1]
                merged_segments[-1] = previous.model_copy(
                    update={
                        "geometry": [*previous.geometry, *segment.geometry[1:]],
                        "distance_km": previous.distance_km + segment.distance_km,
                        "elevation_delta_m": previous.elevation_delta_m + segment.elevation_delta_m,
                        "elevation_gain_m": previous.elevation_gain_m + segment.elevation_gain_m,
                        "elevation_loss_m": previous.elevation_loss_m + segment.elevation_loss_m,
                        "grade_pct": (
                            (previous.elevation_delta_m + segment.elevation_delta_m)
                            / max((previous.distance_km + segment.distance_km) * 1000.0, 1e-9)
                        )
                        * 100.0,
                    }
                )
            else:
                merged_segments.append(segment)

        profile = RouteTerrainProfile(
            sampled_points=len(samples),
            total_gain_m=total_gain,
            total_loss_m=total_loss,
            max_uphill_grade_pct=max_uphill_grade,
            max_downhill_grade_pct=max_downhill_grade,
            source=source,
            segments=merged_segments,
        )
        logger.warning(
            "Terrain debug output: sampled=%d gain=%.2f loss=%.2f max_up=%.4f max_down=%.4f segments=%d first_segments=%s",
            profile.sampled_points,
            profile.total_gain_m,
            profile.total_loss_m,
            profile.max_uphill_grade_pct,
            profile.max_downhill_grade_pct,
            len(profile.segments),
            [
                {
                    "trend": segment.trend,
                    "dist_km": round(segment.distance_km, 3),
                    "delta_m": round(segment.elevation_delta_m, 2),
                    "grade_pct": round(segment.grade_pct, 4),
                    "geom_points": len(segment.geometry),
                }
                for segment in profile.segments[:12]
            ],
        )
        return profile

    def _smooth_elevations(self, samples: list[PolylineSample], elevations: list[float]) -> list[float]:
        if len(samples) != len(elevations) or len(elevations) < 3:
            return [float(value) for value in elevations]

        total_distance = 0.0
        for index in range(len(samples) - 1):
            total_distance += haversine_km(samples[index].point, samples[index + 1].point)

        average_spacing_km = total_distance / max(len(samples) - 1, 1)
        # For city routes, keep the smoothing window local so short rises and drops
        # are not flattened into one long monotonic segment.
        radius = max(1, min(2, round(0.18 / max(average_spacing_km, 0.03))))
        smoothed: list[float] = []

        for index in range(len(elevations)):
            left = max(0, index - radius)
            right = min(len(elevations), index + radius + 1)
            window = elevations[left:right]
            if not window:
                smoothed.append(float(elevations[index]))
                continue
            smoothed.append(sum(float(value) for value in window) / len(window))
        return smoothed

    def _should_merge_segments(self, previous: RouteTerrainSegment, current: RouteTerrainSegment) -> bool:
        if previous.trend != current.trend:
            return False

        combined_distance = previous.distance_km + current.distance_km
        if combined_distance > self._MERGED_SEGMENT_MAX_DISTANCE_KM:
            return False

        grade_delta = abs(previous.grade_pct - current.grade_pct)
        return grade_delta <= self._MERGED_SEGMENT_MAX_GRADE_DELTA_PCT

    def _edge_sampling_budget(self, points_count: int) -> tuple[float, int]:
        pair_count = max(0, points_count * (points_count - 1) // 2)
        if pair_count <= 28:
            return self._EDGE_SAMPLE_SPACING_KM, self._EDGE_SAMPLE_MAX_POINTS
        if pair_count <= 66:
            return 14.0, 64
        if pair_count <= 153:
            return 20.0, 48
        if pair_count <= 300:
            return 28.0, 32
        return 36.0, 24

    def _summarize_edge(self, points: list[Point], elevations: list[float]) -> EdgeTerrainStats:
        if len(points) < 2 or len(elevations) < 2:
            return EdgeTerrainStats(gain_m=0.0, loss_m=0.0, mean_elevation_m=None)

        gain = 0.0
        loss = 0.0
        weighted_sum = 0.0
        weighted_distance = 0.0
        for index in range(len(points) - 1):
            distance_km = haversine_km(points[index], points[index + 1])
            if distance_km <= 0:
                continue
            delta = float(elevations[index + 1]) - float(elevations[index])
            gain += max(delta, 0.0)
            loss += max(-delta, 0.0)
            segment_mean = (float(elevations[index]) + float(elevations[index + 1])) / 2.0
            weighted_sum += segment_mean * distance_km
            weighted_distance += distance_km

        mean_elevation = weighted_sum / weighted_distance if weighted_distance > 0 else None
        return EdgeTerrainStats(gain_m=gain, loss_m=loss, mean_elevation_m=mean_elevation)

    @staticmethod
    def _sample_between_points(
        start: Point,
        end: Point,
        spacing_km: float,
        max_points: int,
    ) -> list[Point]:
        distance_km = haversine_km(start, end)
        if distance_km <= 0:
            return [start, end]
        sample_count = min(max_points, max(2, math.ceil(distance_km / max(spacing_km, 0.5)) + 1))
        points: list[Point] = []
        for index in range(sample_count):
            ratio = 0.0 if sample_count == 1 else index / (sample_count - 1)
            points.append(
                Point(
                    lat=start.lat + (end.lat - start.lat) * ratio,
                    lon=start.lon + (end.lon - start.lon) * ratio,
                )
            )
        return points

    @staticmethod
    def _sample_geometry(
        points: list[Point],
        spacing_km: float,
        max_points: int,
    ) -> list[PolylineSample]:
        if len(points) <= 2:
            last_edge = max(0, len(points) - 2)
            if len(points) == 1:
                return [PolylineSample(point=points[0], edge_index=0, edge_ratio=0.0)]
            return [
                PolylineSample(point=points[0], edge_index=0, edge_ratio=0.0),
                PolylineSample(point=points[-1], edge_index=last_edge, edge_ratio=1.0),
            ]

        total_distance = 0.0
        edge_distances: list[float] = []
        for index in range(len(points) - 1):
            distance = haversine_km(points[index], points[index + 1])
            edge_distances.append(distance)
            total_distance += distance

        if total_distance <= 0:
            return [
                PolylineSample(point=points[0], edge_index=0, edge_ratio=0.0),
                PolylineSample(point=points[-1], edge_index=max(0, len(points) - 2), edge_ratio=1.0),
            ]

        target_count = min(max_points, max(2, math.ceil(total_distance / max(spacing_km, 0.5)) + 1))
        interval = total_distance / max(target_count - 1, 1)
        sampled = [PolylineSample(point=points[0], edge_index=0, edge_ratio=0.0)]
        target_distance = interval
        traversed = 0.0

        for index, edge_distance in enumerate(edge_distances):
            if edge_distance <= 0:
                continue
            start = points[index]
            end = points[index + 1]
            while target_distance < traversed + edge_distance and len(sampled) < target_count - 1:
                ratio = (target_distance - traversed) / edge_distance
                sampled.append(
                    PolylineSample(
                        point=Point(
                            lat=start.lat + (end.lat - start.lat) * ratio,
                            lon=start.lon + (end.lon - start.lon) * ratio,
                        ),
                        edge_index=index,
                        edge_ratio=ratio,
                    )
                )
                target_distance += interval
            traversed += edge_distance

        sampled.append(PolylineSample(point=points[-1], edge_index=len(points) - 2, edge_ratio=1.0))
        return sampled

    @staticmethod
    def _slice_route_geometry(
        route_points: list[Point],
        start_sample: PolylineSample,
        end_sample: PolylineSample,
    ) -> list[list[float]]:
        if not route_points:
            return []
        if start_sample.edge_index == end_sample.edge_index:
            return [
                [start_sample.point.lat, start_sample.point.lon],
                [end_sample.point.lat, end_sample.point.lon],
            ]

        geometry: list[list[float]] = [[start_sample.point.lat, start_sample.point.lon]]
        start_vertex = start_sample.edge_index + 1
        end_vertex = min(end_sample.edge_index, len(route_points) - 2)

        for vertex_index in range(start_vertex, end_vertex + 1):
            point = route_points[vertex_index]
            if not TerrainProfileService._same_coordinate(geometry[-1], point):
                geometry.append([point.lat, point.lon])

        if not TerrainProfileService._same_coordinate(geometry[-1], end_sample.point):
            geometry.append([end_sample.point.lat, end_sample.point.lon])
        return geometry

    @staticmethod
    def _same_coordinate(candidate: list[float], point: Point) -> bool:
        return abs(float(candidate[0]) - float(point.lat)) <= 1e-9 and abs(float(candidate[1]) - float(point.lon)) <= 1e-9

    def _classify_trend(self, grade_pct: float) -> TerrainTrend:
        if grade_pct > self._FLAT_GRADE_THRESHOLD_PCT:
            return TerrainTrend.uphill
        if grade_pct < -self._FLAT_GRADE_THRESHOLD_PCT:
            return TerrainTrend.downhill
        return TerrainTrend.flat

    @staticmethod
    def _point_key(point: Point) -> tuple[float, float]:
        return round(float(point.lat), 6), round(float(point.lon), 6)
