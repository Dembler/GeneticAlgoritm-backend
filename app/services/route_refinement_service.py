from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math

from app.domain.models import (
    CriteriaWeights,
    Point,
    RouteMetrics,
    RouteRefinementInfo,
    RouteRefinementSegmentChoice,
    RouteRequest,
)
from app.repositories.routing_repository import RouteProviderResult, RoutingRepository
from app.services.context_service import OptimizationContext
from app.services.distance import haversine_km
from app.services.fuel_cost import FuelCostService


@dataclass
class RefinementResult:
    route_result: RouteProviderResult
    metrics: RouteMetrics
    info: RouteRefinementInfo


@dataclass
class SegmentCandidate:
    variant: str
    points: list[Point]
    route_result: RouteProviderResult
    metrics: SegmentCandidateMetrics
    score: float = 0.0


@dataclass
class SegmentCandidateMetrics:
    distance_km: float
    duration_min: float
    fuel_liters: float
    risk_exposure: float
    detour_ratio: float
    restriction_penalty: float


class RouteRefinementService:
    """Second-stage optimization for route geometry between fixed visit points."""

    _MIN_RELATIVE_IMPROVEMENT = 0.001

    def __init__(
        self,
        routing_repository: RoutingRepository,
        fuel_cost_service: FuelCostService,
        max_segments: int = 12,
    ) -> None:
        self._routing_repository = routing_repository
        self._fuel_cost_service = fuel_cost_service
        self._max_segments = max_segments

    async def refine(
        self,
        *,
        request: RouteRequest,
        ordered_points: list[Point],
        order_indices: list[int],
        baseline_order_indices: list[int],
        selected_route_result: RouteProviderResult,
        selected_metrics: RouteMetrics,
        context: OptimizationContext,
        weights: CriteriaWeights,
    ) -> RefinementResult:
        if len(ordered_points) < 2:
            return self._unchanged(
                route_result=selected_route_result,
                metrics=selected_metrics,
                reason="not_enough_points",
            )
        if order_indices != baseline_order_indices:
            return self._unchanged(
                route_result=selected_route_result,
                metrics=selected_metrics,
                reason="order_improved_refinement_skipped",
            )
        if len(ordered_points) - 1 > self._max_segments:
            return self._unchanged(
                route_result=selected_route_result,
                metrics=selected_metrics,
                reason="too_many_segments_for_refinement",
            )

        segment_results = await asyncio.gather(
            *(
                self._refine_segment(
                    request=request,
                    context=context,
                    weights=weights,
                    start_index=idx,
                    end_index=idx + 1,
                    start_point=ordered_points[idx],
                    end_point=ordered_points[idx + 1],
                )
                for idx in range(len(ordered_points) - 1)
            )
        )

        selected_segments = [item[0] for item in segment_results]
        baseline_segments = [item[1] for item in segment_results]
        choices = [item[2] for item in segment_results if item[2] is not None]
        candidate_count = sum(
            len(self._candidate_specs(ordered_points[idx], ordered_points[idx + 1]))
            for idx in range(len(ordered_points) - 1)
        )
        max_detour_ratio = max((item[0].metrics.detour_ratio for item in segment_results), default=0.0)
        if not choices:
            return self._unchanged(
                route_result=selected_route_result,
                metrics=selected_metrics,
                reason="same_order_no_better_segment_geometry",
                candidate_count=candidate_count,
                max_detour_ratio=max_detour_ratio,
            )

        baseline_score = sum(item.score for item in baseline_segments)
        selected_score = sum(item.score for item in selected_segments)
        improvement_pct = self._improvement_pct(baseline_score, selected_score)
        if improvement_pct < self._MIN_RELATIVE_IMPROVEMENT * 100.0:
            return self._unchanged(
                route_result=selected_route_result,
                metrics=selected_metrics,
                reason="same_order_refinement_gain_below_threshold",
                candidate_count=candidate_count,
                max_detour_ratio=max_detour_ratio,
            )

        refined_geometry = self._merge_geometries([item.route_result.geometry for item in selected_segments])
        refined_distance = sum(item.route_result.distance_km for item in selected_segments)
        refined_duration = self._sum_optional_durations([item.route_result.duration_min for item in selected_segments])
        baseline_distance = sum(item.route_result.distance_km for item in baseline_segments)
        baseline_duration = self._sum_optional_durations([item.route_result.duration_min for item in baseline_segments])
        refined_metrics = self._adjust_metrics(
            metrics=selected_metrics,
            baseline_distance=baseline_distance,
            refined_distance=refined_distance,
            baseline_duration=baseline_duration,
            refined_duration=refined_duration,
            selected_score=selected_score,
            baseline_score=baseline_score,
        )
        provider = "+".join(sorted({item.route_result.provider for item in selected_segments}))
        route_result = RouteProviderResult(
            geometry=refined_geometry or selected_route_result.geometry,
            distance_km=refined_distance,
            duration_min=refined_duration,
            provider=f"{provider}:refined",
        )
        return RefinementResult(
            route_result=route_result,
            metrics=refined_metrics,
            info=RouteRefinementInfo(
                applied=True,
                reason="same_order_refined_by_segment_geometry",
                improvement_pct=improvement_pct,
                changed_segments=len(choices),
                candidate_count=candidate_count,
                max_detour_ratio=max_detour_ratio,
                segment_choices=choices,
            ),
        )

    async def _refine_segment(
        self,
        *,
        request: RouteRequest,
        context: OptimizationContext,
        weights: CriteriaWeights,
        start_index: int,
        end_index: int,
        start_point: Point,
        end_point: Point,
    ) -> tuple[SegmentCandidate, SegmentCandidate, RouteRefinementSegmentChoice | None]:
        specs = self._candidate_specs(start_point, end_point)
        routes = await asyncio.gather(
            *(self._routing_repository.route(points, request.profile) for _variant, points in specs)
        )
        baseline_distance = routes[0].distance_km if routes else 0.0
        candidates = [
            SegmentCandidate(
                variant=variant,
                points=points,
                route_result=route_result,
                metrics=self._candidate_metrics(
                    request=request,
                    context=context,
                    route_result=route_result,
                    start_index=start_index,
                    end_index=end_index,
                    baseline_distance_km=baseline_distance,
                ),
            )
            for (variant, points), route_result in zip(specs, routes, strict=False)
        ]
        self._assign_candidate_scores(
            candidates=candidates,
            weights=weights,
        )
        baseline = candidates[0]
        selected = min(candidates, key=lambda item: item.score)
        improvement_pct = self._improvement_pct(baseline.score, selected.score)
        if selected.variant == "baseline" or improvement_pct < self._MIN_RELATIVE_IMPROVEMENT * 100.0:
            return baseline, baseline, None
        return (
            selected,
            baseline,
            RouteRefinementSegmentChoice(
                start_index=start_index,
                end_index=end_index,
                from_label=start_point.label or f"Point {start_index + 1}",
                to_label=end_point.label or f"Point {end_index + 1}",
                selected_variant=selected.variant,
                improvement_reason=self._improvement_reason(baseline.route_result, selected.route_result),
                baseline_score=baseline.score,
                selected_score=selected.score,
                improvement_pct=improvement_pct,
                distance_delta_km=selected.route_result.distance_km - baseline.route_result.distance_km,
                duration_delta_min=(
                    0.0
                    if baseline.route_result.duration_min is None or selected.route_result.duration_min is None
                    else selected.route_result.duration_min - baseline.route_result.duration_min
                ),
            ),
        )

    def _candidate_specs(self, start: Point, end: Point) -> list[tuple[str, list[Point]]]:
        return [
            ("baseline", [start, end]),
            ("shortest", [start, self._offset_midpoint(start, end, 0.40), end]),
            ("eco", [start, self._offset_midpoint(start, end, -0.35), end]),
            ("safe", [start, self._offset_midpoint(start, end, 0.70), end]),
            ("avoid_events", [start, self._offset_midpoint(start, end, -0.70), end]),
            ("avoid_bad_surface", [start, self._offset_midpoint(start, end, 1.00), end]),
            ("avoid_restricted", [start, self._offset_midpoint(start, end, -1.00), end]),
        ]

    def _candidate_metrics(
        self,
        *,
        request: RouteRequest,
        context: OptimizationContext,
        route_result: RouteProviderResult,
        start_index: int,
        end_index: int,
        baseline_distance_km: float,
    ) -> SegmentCandidateMetrics:
        distance = max(0.0, route_result.distance_km)
        duration = route_result.duration_min
        if duration is None:
            duration = self._fallback_duration_minutes(distance)
        consumption = self._fuel_cost_service.resolve_consumption_l_per_100km(request)
        fuel_liters = distance * consumption / 100.0
        baseline_distance = baseline_distance_km or self._matrix_distance(context, start_index, end_index)
        detour_ratio = self._detour_ratio(distance, baseline_distance)
        corridor_relief = self._corridor_relief(
            geometry=route_result.geometry,
            context=context,
            start_index=start_index,
            end_index=end_index,
            baseline_distance_km=baseline_distance,
            detour_ratio=detour_ratio,
        )
        risk = self._edge_risk(context, start_index, end_index)
        risk_exposure = risk * (1.0 - corridor_relief) * (1.0 + (0.25 * detour_ratio))
        restriction_penalty = 0.0
        if not self._bool_matrix_value(context.infrastructure_access_matrix, start_index, end_index):
            restriction_penalty += 100_000.0 * (1.0 - corridor_relief)
        if not self._bool_matrix_value(context.temporal_access_matrix, start_index, end_index):
            restriction_penalty += 100_000.0 * (1.0 - corridor_relief)
        return SegmentCandidateMetrics(
            distance_km=distance,
            duration_min=duration,
            fuel_liters=fuel_liters,
            risk_exposure=max(0.0, risk_exposure),
            detour_ratio=detour_ratio,
            restriction_penalty=restriction_penalty,
        )

    def _assign_candidate_scores(
        self,
        *,
        candidates: list[SegmentCandidate],
        weights: CriteriaWeights,
    ) -> None:
        if not candidates:
            return
        distance_bounds = self._min_max([item.metrics.distance_km for item in candidates])
        duration_bounds = self._min_max([item.metrics.duration_min for item in candidates])
        detour_bounds = self._min_max([item.metrics.detour_ratio for item in candidates])
        penalty_bounds = self._min_max([item.metrics.restriction_penalty for item in candidates])
        objective_total = weights.distance + weights.duration + weights.operational_cost
        if objective_total <= 0:
            distance_weight = duration_weight = operational_weight = 1.0 / 3.0
        else:
            distance_weight = weights.distance / objective_total
            duration_weight = weights.duration / objective_total
            operational_weight = weights.operational_cost / objective_total
        for candidate in candidates:
            metrics = candidate.metrics
            distance = self._normalize(metrics.distance_km, distance_bounds)
            duration = self._normalize(metrics.duration_min, duration_bounds)
            detour = self._normalize(metrics.detour_ratio, detour_bounds)
            penalty = self._normalize(metrics.restriction_penalty, penalty_bounds)
            operational_proxy = (distance + duration) / 2.0
            candidate.score = (
                distance_weight * distance
                + duration_weight * duration
                + operational_weight * operational_proxy
                + 0.08 * detour
                + penalty
            )

    def _adjust_metrics(
        self,
        *,
        metrics: RouteMetrics,
        baseline_distance: float,
        refined_distance: float,
        baseline_duration: float | None,
        refined_duration: float | None,
        selected_score: float,
        baseline_score: float,
    ) -> RouteMetrics:
        distance_ratio = refined_distance / baseline_distance if baseline_distance > 1e-9 else 1.0
        if baseline_duration is not None and refined_duration is not None and baseline_duration > 1e-9:
            duration_ratio = refined_duration / baseline_duration
            new_duration = refined_duration
        else:
            duration_ratio = distance_ratio
            new_duration = metrics.duration_min * duration_ratio
        fuel_ratio = max(0.0, (distance_ratio + duration_ratio) / 2.0)
        driver_cost = metrics.driver_cost * duration_ratio
        maintenance_cost = metrics.maintenance_cost * distance_ratio
        fuel_cost = metrics.fuel_cost * fuel_ratio
        operational_cost = fuel_cost + driver_cost + maintenance_cost
        score_ratio = selected_score / baseline_score if baseline_score > 1e-9 else fuel_ratio
        return metrics.model_copy(
            update={
                "distance_km": refined_distance,
                "duration_min": new_duration,
                "fuel_liters": metrics.fuel_liters * fuel_ratio,
                "fuel_cost": fuel_cost,
                "driver_cost": driver_cost,
                "maintenance_cost": maintenance_cost,
                "operational_cost": operational_cost,
                "co2_kg": metrics.co2_kg * fuel_ratio,
                "objective_score": metrics.objective_score * score_ratio,
            }
        )

    def _edge_risk(self, context: OptimizationContext, start_index: int, end_index: int) -> float:
        surface_risk = 1.0 - self._quality_value(context.surface_quality_matrix, start_index, end_index)
        incident_risk = self._matrix_value(context.incident_risk_matrix, start_index, end_index)
        roadwork_risk = self._matrix_value(context.roadwork_risk_matrix, start_index, end_index)
        dynamic_event_risk = max(incident_risk, roadwork_risk)
        return max(
            0.0,
            min(
                1.0,
                0.22 * self._matrix_value(context.traffic_matrix, start_index, end_index)
                + 0.18 * context.weather.severity
                + 0.20 * surface_risk
                + 0.18 * incident_risk
                + 0.17 * roadwork_risk
                + 0.05 * dynamic_event_risk,
            ),
        )

    def _corridor_relief(
        self,
        *,
        geometry: list[list[float]],
        context: OptimizationContext,
        start_index: int,
        end_index: int,
        baseline_distance_km: float,
        detour_ratio: float,
    ) -> float:
        if detour_ratio <= 0.01 or len(geometry) < 3:
            return 0.0
        if start_index >= len(context.points) or end_index >= len(context.points):
            return 0.0
        direct_distance = baseline_distance_km or self._matrix_distance(context, start_index, end_index)
        if direct_distance <= 1e-9:
            return 0.0
        start = context.points[start_index]
        end = context.points[end_index]
        clearances = [
            self._point_to_segment_distance_km(float(point[0]), float(point[1]), start, end)
            for point in geometry[1:-1]
            if len(point) >= 2
        ]
        if not clearances:
            return 0.0
        clearance_km = max(clearances)
        normalized_clearance = clearance_km / max(direct_distance, 0.01)
        return min(0.75, normalized_clearance * 2.2)

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

    def _matrix_distance(self, context: OptimizationContext, start_index: int, end_index: int) -> float:
        matrix_distance = self._positive_matrix_value(context.distance_matrix_km, start_index, end_index)
        if matrix_distance > 0:
            return matrix_distance
        if start_index < len(context.points) and end_index < len(context.points):
            return haversine_km(context.points[start_index], context.points[end_index])
        return 0.0

    @staticmethod
    def _detour_ratio(distance_km: float, baseline_distance_km: float) -> float:
        if baseline_distance_km <= 1e-9:
            return 0.0
        return max(0.0, (distance_km / baseline_distance_km) - 1.0)

    @classmethod
    def _unchanged(
        cls,
        *,
        route_result: RouteProviderResult,
        metrics: RouteMetrics,
        reason: str,
        candidate_count: int = 0,
        max_detour_ratio: float = 0.0,
    ) -> RefinementResult:
        return RefinementResult(
            route_result=route_result,
            metrics=metrics,
            info=RouteRefinementInfo(
                applied=False,
                reason=reason,
                candidate_count=candidate_count,
                max_detour_ratio=max_detour_ratio,
            ),
        )

    @staticmethod
    def _merge_geometries(geometries: list[list[list[float]]]) -> list[list[float]]:
        merged: list[list[float]] = []
        for geometry in geometries:
            if not geometry:
                continue
            if not merged:
                merged.extend(geometry)
                continue
            if RouteRefinementService._same_coordinate(merged[-1], geometry[0]):
                merged.extend(geometry[1:])
            else:
                merged.extend(geometry)
        return merged

    @staticmethod
    def _same_coordinate(a: list[float], b: list[float]) -> bool:
        return len(a) >= 2 and len(b) >= 2 and abs(float(a[0]) - float(b[0])) < 1e-7 and abs(float(a[1]) - float(b[1])) < 1e-7

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
        perp_lat = -dlon / norm
        perp_lon = dlat / norm
        return Point(
            lat=max(-90.0, min(90.0, mid_lat + (perp_lat * offset))),
            lon=max(-180.0, min(180.0, mid_lon + (perp_lon * offset))),
        )

    @staticmethod
    def _improvement_reason(baseline: RouteProviderResult, selected: RouteProviderResult) -> str:
        distance_delta = selected.distance_km - baseline.distance_km
        if baseline.duration_min is not None and selected.duration_min is not None:
            duration_delta = selected.duration_min - baseline.duration_min
            if duration_delta < distance_delta:
                return "shorter_travel_time"
        if distance_delta < 0:
            return "shorter_distance"
        return "lower_weighted_segment_score"

    @staticmethod
    def _improvement_pct(baseline_value: float, selected_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0
        return max(0.0, ((baseline_value - selected_value) / baseline_value) * 100.0)

    @staticmethod
    def _sum_optional_durations(values: list[float | None]) -> float | None:
        if any(value is None for value in values):
            return None
        return sum(float(value or 0.0) for value in values)

    @staticmethod
    def _fallback_duration_minutes(distance_km: float) -> float:
        return (distance_km / 42.0) * 60.0 if distance_km > 0 else 0.0

    @staticmethod
    def _matrix_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, min(1.0, float(matrix[i][j])))
        return 0.0

    @staticmethod
    def _positive_matrix_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, float(matrix[i][j]))
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
