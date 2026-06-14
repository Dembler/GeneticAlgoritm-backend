from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import time

from app.domain.models import (
    CriteriaWeights,
    Point,
    RouteRequest,
    SegmentAlternativeSet,
    SegmentAlternativesSummary,
    SegmentCandidate,
)
from app.repositories.routing_repository import RouteProviderResult, RoutingRepository
from app.services.distance import haversine_km
from app.services.fuel_cost import FuelCostService, FuelPriceSnapshot


@dataclass
class SegmentAlternativeMatrix:
    segment_alternatives: dict[tuple[int, int], SegmentAlternativeSet]
    best_segment_score_matrix: list[list[float]]
    best_segment_distance_matrix_km: list[list[float]]
    best_segment_duration_matrix_min: list[list[float]]
    best_segment_choice_matrix: list[list[SegmentCandidate | None]]
    summary: SegmentAlternativesSummary
    enabled: bool


@dataclass
class _CacheEntry:
    expires_at: float
    value: SegmentAlternativeSet


class SegmentAlternativeService:
    _HARD_RESTRICTION_PENALTY = 100_000.0

    def __init__(
        self,
        *,
        routing_repository: RoutingRepository,
        fuel_cost_service: FuelCostService,
        enabled: bool = True,
        max_candidates_per_edge: int = 5,
        max_points: int = 20,
        max_concurrency: int = 8,
        max_detour_ratio: float = 1.15,
        cache_ttl_sec: int = 300,
    ) -> None:
        self._routing_repository = routing_repository
        self._fuel_cost_service = fuel_cost_service
        self._enabled = enabled
        self._max_candidates_per_edge = max(1, max_candidates_per_edge)
        self._max_points = max(2, max_points)
        self._max_concurrency = max(1, max_concurrency)
        self._max_detour_ratio = max(1.0, max_detour_ratio)
        self._cache_ttl_sec = max(0, cache_ttl_sec)
        self._cache: dict[str, _CacheEntry] = {}

    async def build_matrix(
        self,
        *,
        request: RouteRequest,
        points: list[Point],
        context,
        weights: CriteriaWeights,
        fuel_prices: FuelPriceSnapshot,
    ) -> SegmentAlternativeMatrix:
        size = len(points)
        fallback = self._fallback_matrix(context, enabled=False)
        if not self._enabled or size < 2 or size > self._max_points:
            return fallback

        semaphore = asyncio.Semaphore(self._max_concurrency)
        pairs = [(i, j) for i in range(size) for j in range(size) if i != j]

        async def build_pair(i: int, j: int) -> tuple[tuple[int, int], SegmentAlternativeSet]:
            async with semaphore:
                return (i, j), await self._build_pair_set(
                    request=request,
                    points=points,
                    context=context,
                    weights=weights,
                    fuel_prices=fuel_prices,
                    from_index=i,
                    to_index=j,
                )

        results = await asyncio.gather(*(build_pair(i, j) for i, j in pairs))
        alternatives = dict(results)
        score_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        distance_matrix = [list(row) for row in context.distance_matrix_km]
        duration_matrix = [list(row) for row in context.duration_matrix_min]
        choice_matrix: list[list[SegmentCandidate | None]] = [[None for _ in range(size)] for _ in range(size)]
        total_candidates = 0
        used_candidates = 0
        detours: list[float] = []
        gain_values: list[float] = []

        for (i, j), alt_set in alternatives.items():
            total_candidates += len(alt_set.candidates)
            best = alt_set.best_candidate
            baseline = alt_set.baseline_candidate
            choice_matrix[i][j] = best
            score_matrix[i][j] = best.objective_score
            distance_matrix[i][j] = best.distance_km
            duration_matrix[i][j] = best.duration_min
            if best.variant_id != baseline.variant_id:
                used_candidates += 1
                detours.append(best.detour_ratio)
                gain_values.append(self._improvement_pct(baseline.objective_score, best.objective_score))

        summary = SegmentAlternativesSummary(
            enabled=True,
            total_pairs=len(alternatives),
            total_candidates=total_candidates,
            used_candidates=used_candidates,
            average_candidates_per_pair=total_candidates / max(len(alternatives), 1),
            average_detour_ratio=sum(detours) / len(detours) if detours else 0.0,
            estimated_gain_pct=sum(gain_values) / len(gain_values) if gain_values else 0.0,
        )
        return SegmentAlternativeMatrix(
            segment_alternatives=alternatives,
            best_segment_score_matrix=score_matrix,
            best_segment_distance_matrix_km=distance_matrix,
            best_segment_duration_matrix_min=duration_matrix,
            best_segment_choice_matrix=choice_matrix,
            summary=summary,
            enabled=True,
        )

    async def _build_pair_set(
        self,
        *,
        request: RouteRequest,
        points: list[Point],
        context,
        weights: CriteriaWeights,
        fuel_prices: FuelPriceSnapshot,
        from_index: int,
        to_index: int,
    ) -> SegmentAlternativeSet:
        cache_key = self._cache_key(request, points[from_index], points[to_index], from_index, to_index)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        specs = self._candidate_specs(points[from_index], points[to_index])[: self._candidate_limit(len(points))]
        route_results: list[tuple[str, list[Point], RouteProviderResult | None]] = []
        for variant_type, route_points in specs:
            try:
                route_results.append(
                    (
                        variant_type,
                        route_points,
                        await self._routing_repository.route(route_points, request.profile),
                    )
                )
            except Exception:
                route_results.append((variant_type, route_points, None))

        baseline_result = route_results[0][2]
        baseline_distance = (
            baseline_result.distance_km
            if baseline_result is not None and baseline_result.distance_km > 0
            else self._matrix_distance(context, points, from_index, to_index)
        )
        baseline_duration = (
            baseline_result.duration_min
            if baseline_result is not None and baseline_result.duration_min is not None
            else self._matrix_duration(context, from_index, to_index, baseline_distance)
        )
        candidates = [
            self._candidate_from_route(
                request=request,
                context=context,
                weights=weights,
                fuel_prices=fuel_prices,
                from_index=from_index,
                to_index=to_index,
                variant_type=variant_type,
                route_points=route_points,
                route_result=route_result,
                baseline_distance=baseline_distance,
                baseline_duration=baseline_duration,
            )
            for variant_type, route_points, route_result in route_results
        ]
        self._assign_scores(candidates, weights)
        baseline = candidates[0]
        viable = [
            candidate
            for candidate in candidates
            if candidate.detour_ratio <= self._max_detour_ratio or candidate.variant_type in {"baseline", "fastest"}
        ]
        best = min(viable or candidates, key=lambda item: item.objective_score)
        alt_set = SegmentAlternativeSet(
            from_index=from_index,
            to_index=to_index,
            candidates=candidates,
            best_candidate=best,
            baseline_candidate=baseline,
        )
        self._cache_set(cache_key, alt_set)
        return alt_set

    def _candidate_from_route(
        self,
        *,
        request: RouteRequest,
        context,
        weights: CriteriaWeights,
        fuel_prices: FuelPriceSnapshot,
        from_index: int,
        to_index: int,
        variant_type: str,
        route_points: list[Point],
        route_result: RouteProviderResult | None,
        baseline_distance: float,
        baseline_duration: float,
    ) -> SegmentCandidate:
        if route_result is None:
            distance = baseline_distance
            duration = baseline_duration
            geometry = [[route_points[0].lat, route_points[0].lon], [route_points[-1].lat, route_points[-1].lon]]
            provider = "fallback-candidate"
            explanation = "provider_error_fallback"
        else:
            distance = max(0.0, route_result.distance_km)
            duration = route_result.duration_min
            if duration is None:
                duration = self._fallback_duration_minutes(distance)
            geometry = route_result.geometry
            provider = route_result.provider
            explanation = self._explanation(variant_type)

        consumption = self._fuel_cost_service.resolve_consumption_l_per_100km(request)
        fuel_liters = distance * consumption / 100.0
        fuel_cost = fuel_liters * self._fuel_cost_service.price_per_liter(fuel_prices, request.fuel_type)
        risk_values = self._risk_values(
            context=context,
            geometry=geometry,
            from_index=from_index,
            to_index=to_index,
            baseline_distance=baseline_distance,
        )
        restriction_penalty = self._restriction_penalty(context, from_index, to_index, risk_values["relief"])
        detour_ratio = distance / max(baseline_distance, 0.001)
        return SegmentCandidate(
            variant_id=f"{from_index}:{to_index}:{variant_type}",
            variant_type=variant_type,
            from_index=from_index,
            to_index=to_index,
            distance_km=distance,
            duration_min=duration,
            fuel_liters=fuel_liters,
            fuel_cost=fuel_cost,
            risk_exposure=risk_values["risk_exposure"],
            road_quality_risk=risk_values["road_quality_risk"],
            weather_risk=risk_values["weather_risk"],
            dynamic_event_risk=risk_values["dynamic_event_risk"],
            safety_risk=risk_values["safety_risk"],
            cargo_risk=risk_values["cargo_risk"],
            detour_ratio=detour_ratio,
            restriction_penalty=restriction_penalty,
            objective_score=0.0,
            geometry=geometry,
            data_source=provider,
            explanation=explanation,
        )

    def _assign_scores(self, candidates: list[SegmentCandidate], weights: CriteriaWeights) -> None:
        bounds = {
            "distance": self._min_max([item.distance_km for item in candidates]),
            "duration": self._min_max([item.duration_min for item in candidates]),
            "detour": self._min_max([item.detour_ratio for item in candidates]),
            "penalty": self._min_max([item.restriction_penalty for item in candidates]),
        }
        objective_total = weights.distance + weights.duration + weights.operational_cost
        if objective_total <= 0:
            distance_weight = duration_weight = operational_weight = 1.0 / 3.0
        else:
            distance_weight = weights.distance / objective_total
            duration_weight = weights.duration / objective_total
            operational_weight = weights.operational_cost / objective_total
        for candidate in candidates:
            detour_penalty = 0.0
            if candidate.detour_ratio > self._max_detour_ratio:
                detour_penalty = 2.0 + (candidate.detour_ratio - self._max_detour_ratio) * 10.0
            distance = self._normalize(candidate.distance_km, bounds["distance"])
            duration = self._normalize(candidate.duration_min, bounds["duration"])
            operational_proxy = (distance + duration) / 2.0
            candidate.objective_score = (
                distance_weight * distance
                + duration_weight * duration
                + operational_weight * operational_proxy
                + 0.08 * self._normalize(candidate.detour_ratio, bounds["detour"])
                + self._normalize(candidate.restriction_penalty, bounds["penalty"])
                + detour_penalty
            )

    def _candidate_specs(self, start: Point, end: Point) -> list[tuple[str, list[Point]]]:
        return [
            ("fastest", [start, end]),
            ("shortest_proxy", [start, self._offset_midpoint(start, end, 0.35), end]),
            ("safe_detour", [start, self._offset_midpoint(start, end, 0.75), end]),
            ("avoid_events", [start, self._offset_midpoint(start, end, -0.75), end]),
            ("eco_proxy", [start, self._offset_midpoint(start, end, -0.35), end]),
        ]

    def _candidate_limit(self, points_count: int) -> int:
        if points_count > 12:
            return min(self._max_candidates_per_edge, 3)
        return self._max_candidates_per_edge

    def _risk_values(self, *, context, geometry: list[list[float]], from_index: int, to_index: int, baseline_distance: float) -> dict[str, float]:
        relief = self._corridor_relief(context, geometry, from_index, to_index, baseline_distance)
        road_quality_risk = (1.0 - self._quality_value(context.surface_quality_matrix, from_index, to_index)) * (1.0 - relief)
        incident = self._matrix_value(context.incident_risk_matrix, from_index, to_index) * (1.0 - relief)
        roadwork = self._matrix_value(context.roadwork_risk_matrix, from_index, to_index) * (1.0 - relief)
        dynamic_event_risk = max(incident, roadwork)
        weather_risk = context.weather.severity * (1.0 - (relief * 0.35))
        traffic = self._matrix_value(context.traffic_matrix, from_index, to_index) * (1.0 - (relief * 0.5))
        safety_risk = self._clamp01(
            0.30 * weather_risk
            + 0.22 * traffic
            + 0.18 * road_quality_risk
            + 0.16 * incident
            + 0.14 * roadwork
        )
        cargo_risk = self._clamp01((road_quality_risk + dynamic_event_risk + weather_risk + safety_risk) / 4.0)
        risk_exposure = self._clamp01(
            (road_quality_risk + weather_risk + dynamic_event_risk + safety_risk + cargo_risk) / 5.0
        )
        return {
            "relief": relief,
            "road_quality_risk": road_quality_risk,
            "weather_risk": weather_risk,
            "dynamic_event_risk": dynamic_event_risk,
            "safety_risk": safety_risk,
            "cargo_risk": cargo_risk,
            "risk_exposure": risk_exposure,
        }

    def _restriction_penalty(self, context, from_index: int, to_index: int, relief: float) -> float:
        penalty = 0.0
        relief_factor = 1.0 - min(0.90, max(0.0, relief))
        if not self._bool_matrix_value(context.infrastructure_access_matrix, from_index, to_index):
            penalty += self._HARD_RESTRICTION_PENALTY * relief_factor
        if not self._bool_matrix_value(context.temporal_access_matrix, from_index, to_index):
            penalty += self._HARD_RESTRICTION_PENALTY * relief_factor
        return penalty

    def _corridor_relief(self, context, geometry: list[list[float]], from_index: int, to_index: int, baseline_distance: float) -> float:
        if len(geometry) < 3 or from_index >= len(context.points) or to_index >= len(context.points):
            return 0.0
        start = context.points[from_index]
        end = context.points[to_index]
        clearances = [
            self._point_to_segment_distance_km(float(point[0]), float(point[1]), start, end)
            for point in geometry[1:-1]
            if len(point) >= 2
        ]
        if not clearances:
            return 0.0
        normalized = max(clearances) / max(baseline_distance, 0.01)
        return min(0.75, normalized * 2.2)

    @staticmethod
    def _point_to_segment_distance_km(lat: float, lon: float, start: Point, end: Point) -> float:
        mean_lat = math.radians((start.lat + end.lat + lat) / 3.0)
        x = lon * math.cos(mean_lat) * 111.320
        y = lat * 110.574
        start_x = start.lon * math.cos(mean_lat) * 111.320
        start_y = start.lat * 110.574
        end_x = end.lon * math.cos(mean_lat) * 111.320
        end_y = end.lat * 110.574
        dx = end_x - start_x
        dy = end_y - start_y
        length_squared = (dx * dx) + (dy * dy)
        if length_squared <= 1e-12:
            return haversine_km(Point(lat=lat, lon=lon), start)
        t = max(0.0, min(1.0, (((x - start_x) * dx) + ((y - start_y) * dy)) / length_squared))
        projection_x = start_x + (t * dx)
        projection_y = start_y + (t * dy)
        return math.sqrt(((x - projection_x) ** 2) + ((y - projection_y) ** 2))

    @staticmethod
    def _offset_midpoint(start: Point, end: Point, scale: float) -> Point:
        mid_lat = (start.lat + end.lat) / 2.0
        mid_lon = (start.lon + end.lon) / 2.0
        dlat = end.lat - start.lat
        dlon = end.lon - start.lon
        norm = math.sqrt((dlat * dlat) + (dlon * dlon))
        if norm <= 1e-9:
            return Point(lat=mid_lat, lon=mid_lon)
        offset = min(0.08, max(0.003, norm * 0.18)) * scale
        return Point(
            lat=max(-90.0, min(90.0, mid_lat + ((-dlon / norm) * offset))),
            lon=max(-180.0, min(180.0, mid_lon + ((dlat / norm) * offset))),
        )

    def _cache_key(self, request: RouteRequest, start: Point, end: Point, from_index: int, to_index: int) -> str:
        departure = request.departure_at.isoformat() if request.departure_at else ""
        return (
            f"{from_index}:{to_index}|{start.lat:.6f},{start.lon:.6f}|{end.lat:.6f},{end.lon:.6f}|"
            f"{request.profile.value}|{request.vehicle_class.value}|{request.cargo.profile.value}|"
            f"{departure}|{request.priority_profile.value}"
        )

    def _cache_get(self, key: str) -> SegmentAlternativeSet | None:
        if self._cache_ttl_sec <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.time():
            self._cache.pop(key, None)
            return None
        return entry.value

    def _cache_set(self, key: str, value: SegmentAlternativeSet) -> None:
        if self._cache_ttl_sec <= 0:
            return
        self._cache[key] = _CacheEntry(expires_at=time.time() + self._cache_ttl_sec, value=value)

    @staticmethod
    def _fallback_matrix(context, enabled: bool) -> SegmentAlternativeMatrix:
        size = len(context.points)
        return SegmentAlternativeMatrix(
            segment_alternatives={},
            best_segment_score_matrix=[[0.0 for _ in range(size)] for _ in range(size)],
            best_segment_distance_matrix_km=[list(row) for row in context.distance_matrix_km],
            best_segment_duration_matrix_min=[list(row) for row in context.duration_matrix_min],
            best_segment_choice_matrix=[[None for _ in range(size)] for _ in range(size)],
            summary=SegmentAlternativesSummary(enabled=enabled),
            enabled=enabled,
        )

    @staticmethod
    def _matrix_distance(context, points: list[Point], from_index: int, to_index: int) -> float:
        if from_index < len(context.distance_matrix_km) and to_index < len(context.distance_matrix_km[from_index]):
            value = float(context.distance_matrix_km[from_index][to_index])
            if value > 0:
                return value
        return haversine_km(points[from_index], points[to_index])

    @staticmethod
    def _matrix_duration(context, from_index: int, to_index: int, distance: float) -> float:
        if from_index < len(context.duration_matrix_min) and to_index < len(context.duration_matrix_min[from_index]):
            value = float(context.duration_matrix_min[from_index][to_index])
            if value > 0:
                return value
        return SegmentAlternativeService._fallback_duration_minutes(distance)

    @staticmethod
    def _fallback_duration_minutes(distance_km: float) -> float:
        return (distance_km / 42.0) * 60.0 if distance_km > 0 else 0.0

    @staticmethod
    def _explanation(variant_type: str) -> str:
        return {
            "fastest": "baseline_provider_route",
            "shortest_proxy": "distance_weighted_segment_candidate",
            "safe_detour": "risk_aware_detour_candidate",
            "avoid_events": "dynamic_event_avoidance_candidate",
            "eco_proxy": "fuel_risk_detour_balance_candidate",
        }.get(variant_type, "segment_candidate")

    @staticmethod
    def _improvement_pct(baseline_value: float, selected_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0
        return max(0.0, ((baseline_value - selected_value) / baseline_value) * 100.0)

    @staticmethod
    def _min_max(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        return min(values), max(values)

    @staticmethod
    def _normalize(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        if high - low <= 1e-12:
            return 0.0
        return (value - low) / (high - low)

    @staticmethod
    def _matrix_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, min(1.0, float(matrix[i][j])))
        return 0.0

    @staticmethod
    def _quality_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, min(1.0, float(matrix[i][j])))
        return 1.0

    @staticmethod
    def _bool_matrix_value(matrix: list[list[bool]], i: int, j: int) -> bool:
        if i < len(matrix) and j < len(matrix[i]):
            return bool(matrix[i][j])
        return True

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
