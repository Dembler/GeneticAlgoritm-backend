from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.domain.models import (
    CargoProfile,
    CriteriaWeights,
    RouteFitnessComponents,
    RouteMetrics,
    RouteRequest,
    RouteSegmentFactor,
    VehicleClass,
)
from app.repositories.weather_repository import WeatherProfile, WeatherSnapshot
from app.services.context_service import OptimizationContext
from app.services.fuel_cost import FuelCostService, FuelPriceSnapshot


@dataclass
class CandidateEvaluation:
    order_indices: list[int]
    metrics: RouteMetrics
    segment_factors: list[RouteSegmentFactor]
    uphill_pct: float
    downhill_pct: float
    mean_elevation_m: float | None
    mean_temperature_c: float | None = None


class CriteriaService:
    _HARD_INFRASTRUCTURE_PENALTY = 100_000.0
    _HARD_TEMPORAL_RESTRICTION_PENALTY = 100_000.0
    _DEFAULT_DRIVER_COST_PER_HOUR = {
        VehicleClass.passenger: 300.0,
        VehicleClass.light_truck: 650.0,
        VehicleClass.heavy_truck: 950.0,
    }
    _DEFAULT_MAINTENANCE_COST_PER_KM = {
        VehicleClass.passenger: 6.0,
        VehicleClass.light_truck: 18.0,
        VehicleClass.heavy_truck: 42.0,
    }
    _DEFAULT_CARGO_VALUE_RUB = {
        CargoProfile.standard: 0.0,
        CargoProfile.perishable: 120_000.0,
        CargoProfile.fragile: 180_000.0,
        CargoProfile.hazardous: 250_000.0,
        CargoProfile.heavy: 220_000.0,
        CargoProfile.high_value: 500_000.0,
    }
    _CARGO_LOSS_SCALE = {
        CargoProfile.standard: 0.004,
        CargoProfile.perishable: 0.025,
        CargoProfile.fragile: 0.022,
        CargoProfile.hazardous: 0.040,
        CargoProfile.heavy: 0.014,
        CargoProfile.high_value: 0.018,
    }

    def __init__(self, fuel_cost_service: FuelCostService) -> None:
        self._fuel_cost_service = fuel_cost_service

    def evaluate(
        self,
        order_indices: list[int],
        request: RouteRequest,
        context: OptimizationContext,
        fuel_prices: FuelPriceSnapshot,
    ) -> CandidateEvaluation:
        night_factor = self._night_factor(context.departure_at)
        weather = context.weather.severity
        distances = context.distance_matrix_km
        durations = context.duration_matrix_min
        traffic = context.traffic_matrix
        tolls = context.toll_matrix
        road_quality = context.surface_quality_matrix
        incident_risks = context.incident_risk_matrix
        roadwork_risks = context.roadwork_risk_matrix
        temporal_access = context.temporal_access_matrix
        elevations = context.elevation.elevations_m
        elevation_gain_matrix = context.elevation_gain_matrix_m
        elevation_loss_matrix = context.elevation_loss_matrix_m
        mean_elevation_matrix = context.mean_elevation_matrix_m
        weather_profiles = context.weather_profiles

        total_distance = 0.0
        total_duration = 0.0
        total_congestion = 0.0
        total_weather_weight = 0.0
        total_weighted_weather = 0.0
        total_weighted_temperature = 0.0
        total_temperature_weight = 0.0
        total_reliability_risk = 0.0
        total_safety_risk = 0.0
        total_toll = 0.0
        total_infrastructure_penalty = 0.0
        total_temporal_restriction_penalty = 0.0
        total_weighted_road_quality_risk = 0.0
        total_road_quality_weight = 0.0
        total_weighted_incident_risk = 0.0
        total_weighted_roadwork_risk = 0.0
        total_weighted_dynamic_event_risk = 0.0
        total_dynamic_event_weight = 0.0
        total_weighted_cargo_risk = 0.0
        total_cargo_risk_weight = 0.0
        total_gain = 0.0
        total_loss = 0.0
        total_segment_alternative_gain_pct = 0.0
        total_detour_ratio = 0.0
        refined_segments_count = 0
        weighted_mean_elevation_sum = 0.0
        weighted_mean_elevation_distance = 0.0
        violated_constraints: set[str] = set()
        segments: list[RouteSegmentFactor] = []

        for edge_idx in range(len(order_indices) - 1):
            i = order_indices[edge_idx]
            j = order_indices[edge_idx + 1]
            segment_candidate = self._best_segment_candidate(context, i, j)
            distance = (
                segment_candidate.distance_km
                if segment_candidate is not None
                else self._safe_matrix_value(distances, i, j)
            )
            base_duration = (
                segment_candidate.duration_min
                if segment_candidate is not None
                else self._safe_matrix_value(durations, i, j)
            )
            congestion = self._safe_matrix_value(traffic, i, j)
            surface_quality = self._safe_quality_matrix_value(road_quality, i, j)
            surface_risk = 1.0 - surface_quality
            incident_risk = min(1.0, self._safe_matrix_value(incident_risks, i, j))
            roadwork_risk = min(1.0, self._safe_matrix_value(roadwork_risks, i, j))
            dynamic_event_risk = max(incident_risk, roadwork_risk)
            temporal_accessible = self._safe_access_value(temporal_access, i, j)
            segment_variant_type = None
            segment_variant_score = None
            segment_detour_ratio = 0.0
            segment_refinement_applied = False
            if segment_candidate is not None:
                surface_risk = self._clamp01(segment_candidate.road_quality_risk)
                surface_quality = 1.0 - surface_risk
                dynamic_event_risk = self._clamp01(segment_candidate.dynamic_event_risk)
                incident_risk = dynamic_event_risk
                roadwork_risk = dynamic_event_risk
                segment_variant_type = segment_candidate.variant_type
                segment_variant_score = segment_candidate.objective_score
                segment_detour_ratio = max(0.0, segment_candidate.detour_ratio)
                segment_refinement_applied = segment_candidate.variant_type not in {"baseline", "fastest"}
                if segment_refinement_applied:
                    refined_segments_count += 1
                total_detour_ratio += segment_detour_ratio
                baseline_candidate = context.segment_alternatives.get((i, j)).baseline_candidate if (i, j) in context.segment_alternatives else None
                if baseline_candidate is not None:
                    total_segment_alternative_gain_pct += self._improvement_pct(
                        baseline_candidate.objective_score,
                        segment_candidate.objective_score,
                    )
            provisional_duration = base_duration if base_duration > 0 else (distance / 38.0 * 60.0 if distance > 0 else 0.0)
            edge_midpoint_time = context.departure_at + timedelta(minutes=total_duration + (provisional_duration / 2.0))
            edge_weather = self._edge_weather_snapshot(weather_profiles, i, j, edge_midpoint_time, context.weather)
            weather = edge_weather.severity
            if segment_candidate is not None:
                weather = self._clamp01(segment_candidate.weather_risk)
            gain = self._safe_matrix_value(elevation_gain_matrix, i, j)
            loss = self._safe_matrix_value(elevation_loss_matrix, i, j)
            distance_m = max(distance * 1000.0, 1.0)
            grade_risk = min(1.0, (gain / distance_m) * 20.0)
            cargo_risk = self._cargo_risk_for_edge(
                request=request,
                surface_risk=surface_risk,
                weather_risk=weather,
                roadwork_risk=roadwork_risk,
                incident_risk=incident_risk,
                grade_risk=grade_risk,
                congestion_index=congestion,
                dynamic_event_risk=dynamic_event_risk,
                temporal_accessible=temporal_accessible,
            )
            weather_surface_interaction = weather * surface_risk
            weather_slope_interaction = weather * grade_risk
            weather_delay_factor = weather * 0.35
            surface_delay_factor = surface_risk * 0.18
            incident_delay_factor = incident_risk * 0.22
            roadwork_delay_factor = roadwork_risk * 0.28
            interaction_delay_factor = (weather_surface_interaction * 0.10) + (weather_slope_interaction * 0.08)
            duration = base_duration * (
                1.0
                + congestion
                + weather_delay_factor
                + surface_delay_factor
                + incident_delay_factor
                + roadwork_delay_factor
                + interaction_delay_factor
            )
            if duration <= 0 and distance > 0:
                duration = distance / 38.0 * 60.0

            segment_mean_elevation = self._safe_matrix_value(mean_elevation_matrix, i, j)
            total_gain += gain
            total_loss += loss
            if distance > 0 and segment_mean_elevation > 0:
                weighted_mean_elevation_sum += segment_mean_elevation * distance
                weighted_mean_elevation_distance += distance
            if distance > 0:
                total_weather_weight += distance
                total_weighted_weather += weather * distance
                total_road_quality_weight += distance
                total_weighted_road_quality_risk += surface_risk * distance
                total_dynamic_event_weight += distance
                total_weighted_incident_risk += incident_risk * distance
                total_weighted_roadwork_risk += roadwork_risk * distance
                total_weighted_dynamic_event_risk += dynamic_event_risk * distance
                total_cargo_risk_weight += distance
                total_weighted_cargo_risk += cargo_risk * distance
                if edge_weather.temperature_c is not None:
                    total_weighted_temperature += edge_weather.temperature_c * distance
                    total_temperature_weight += distance

            incident_proxy = max(
                0.0,
                min(
                    1.0,
                    0.08
                    + 0.22 * congestion
                    + 0.18 * weather
                    + 0.15 * surface_risk
                    + 0.28 * incident_risk
                    + 0.18 * roadwork_risk
                    + 0.12 * weather_slope_interaction,
                ),
            )
            reliability_risk = max(
                0.0,
                min(
                    1.0,
                    0.35 * congestion
                    + 0.24 * weather
                    + 0.14 * surface_risk
                    + 0.20 * incident_risk
                    + 0.18 * roadwork_risk
                    + 0.14 * weather_slope_interaction
                    + 0.18 * incident_proxy,
                ),
            )
            safety_risk = max(
                0.0,
                min(
                    1.0,
                    0.36 * weather
                    + 0.22 * congestion
                    + 0.17 * surface_risk
                    + 0.16 * incident_risk
                    + 0.13 * roadwork_risk
                    + 0.16 * weather_slope_interaction
                    + 0.16 * night_factor,
                ),
            )
            toll_cost = self._safe_matrix_value(tolls, i, j)
            infrastructure_penalty, segment_violations = self._infrastructure_penalty_for_edge(
                request=request,
                context=context,
                i=i,
                j=j,
            )
            temporal_penalty = 0.0
            if not temporal_accessible:
                temporal_penalty = self._HARD_TEMPORAL_RESTRICTION_PENALTY
                segment_violations.append("temporal_access")
            if segment_candidate is not None and segment_candidate.restriction_penalty > 0:
                temporal_penalty += segment_candidate.restriction_penalty
                segment_violations.append("segment_restriction")
            if segment_violations:
                violated_constraints.update(segment_violations)

            avg_speed = 0.0 if duration <= 0 else distance / (duration / 60.0)
            segments.append(
                RouteSegmentFactor(
                    start_index=edge_idx,
                    end_index=edge_idx + 1,
                    distance_km=distance,
                    duration_min=duration,
                    avg_speed_kph=avg_speed,
                    elevation_gain_m=gain,
                    elevation_loss_m=loss,
                    congestion_index=congestion,
                    weather_severity=weather,
                    reliability_risk=reliability_risk,
                    safety_risk=safety_risk,
                    toll_cost=toll_cost,
                    road_quality=surface_quality,
                    road_quality_risk=surface_risk,
                    incident_risk=incident_risk,
                    roadwork_risk=roadwork_risk,
                    dynamic_event_risk=dynamic_event_risk,
                    cargo_risk=cargo_risk,
                    temporal_accessible=temporal_accessible,
                    infrastructure_penalty=infrastructure_penalty,
                    violated_constraints=segment_violations,
                    height_clearance_m=self._safe_optional_matrix_value(
                        context.height_clearance_matrix_m,
                        i,
                        j,
                    ),
                    weight_limit_t=self._safe_optional_matrix_value(context.weight_limit_matrix_t, i, j),
                    width_limit_m=self._safe_optional_matrix_value(context.width_limit_matrix_m, i, j),
                    length_limit_m=self._safe_optional_matrix_value(context.length_limit_matrix_m, i, j),
                    infrastructure_accessible=self._safe_access_value(
                        context.infrastructure_access_matrix,
                        i,
                        j,
                    ),
                    segment_variant_type=segment_variant_type,
                    segment_variant_score=segment_variant_score,
                    detour_ratio=segment_detour_ratio,
                    refinement_applied_on_segment=segment_refinement_applied,
                )
            )

            total_distance += distance
            total_duration += duration
            total_congestion += congestion
            total_reliability_risk += reliability_risk
            total_safety_risk += safety_risk
            total_toll += toll_cost
            total_infrastructure_penalty += infrastructure_penalty
            total_temporal_restriction_penalty += temporal_penalty

        edge_count = max(1, len(order_indices) - 1)
        congestion_index = total_congestion / edge_count
        weather_risk_avg = (
            total_weighted_weather / total_weather_weight if total_weather_weight > 0 else context.weather.severity
        )
        road_quality_risk_avg = (
            total_weighted_road_quality_risk / total_road_quality_weight if total_road_quality_weight > 0 else 0.0
        )
        incident_risk_avg = (
            total_weighted_incident_risk / total_dynamic_event_weight if total_dynamic_event_weight > 0 else 0.0
        )
        roadwork_risk_avg = (
            total_weighted_roadwork_risk / total_dynamic_event_weight if total_dynamic_event_weight > 0 else 0.0
        )
        dynamic_event_risk_avg = (
            total_weighted_dynamic_event_risk / total_dynamic_event_weight if total_dynamic_event_weight > 0 else 0.0
        )
        cargo_risk_avg = total_weighted_cargo_risk / total_cargo_risk_weight if total_cargo_risk_weight > 0 else 0.0
        mean_temperature_c = (
            total_weighted_temperature / total_temperature_weight
            if total_temperature_weight > 0
            else context.weather.temperature_c
        )
        reliability_risk_avg = total_reliability_risk / edge_count
        safety_risk_avg = total_safety_risk / edge_count

        uphill_pct, downhill_pct = self._terrain_share_from_elevation(total_distance, total_gain, total_loss)
        mean_elevation_m = (
            weighted_mean_elevation_sum / weighted_mean_elevation_distance
            if weighted_mean_elevation_distance > 0
            else self._mean_elevation(elevations)
        )

        terrain_multiplier = self._fuel_cost_service.terrain_multiplier(
            request.vehicle_class, uphill_pct, downhill_pct
        )
        mountain_multiplier = self._fuel_cost_service.mountain_multiplier(
            request.vehicle_class,
            uphill_pct,
            mean_elevation_m,
        )
        temperature_multiplier = self._fuel_cost_service.temperature_multiplier(
            mean_temperature_c,
            total_distance,
        )
        consumption = self._fuel_cost_service.resolve_consumption_l_per_100km(request)
        liters = self._fuel_cost_service.compute_liters(total_distance, consumption, terrain_multiplier)
        liters *= mountain_multiplier
        liters *= temperature_multiplier
        liters *= 1.0 + (0.2 * congestion_index)
        liters *= 1.0 + (0.12 * road_quality_risk_avg)
        liters *= 1.0 + (0.08 * dynamic_event_risk_avg)
        liters *= self._fuel_cost_service.load_multiplier(request)
        price_per_liter = self._fuel_cost_service.price_per_liter(fuel_prices, request.fuel_type)
        fuel_cost = liters * price_per_liter + total_toll
        terrain_complexity = self._clamp01((uphill_pct + downhill_pct) / 20.0)
        driver_cost = self._driver_cost(request, total_duration)
        maintenance_cost = self._maintenance_cost(
            request=request,
            distance_km=total_distance,
            road_quality_risk=road_quality_risk_avg,
            dynamic_event_risk=dynamic_event_risk_avg,
            terrain_complexity=terrain_complexity,
        )
        cargo_expected_loss = self._cargo_expected_loss(request, cargo_risk_avg)
        operational_cost = fuel_cost + driver_cost + maintenance_cost
        co2_kg = self._fuel_cost_service.estimate_co2_kg(liters, request.fuel_type)

        deadline_penalty = self._deadline_penalty(
            request=request,
            departure_at=context.departure_at,
            duration_min=total_duration,
        )
        (
            capacity_penalty,
            vehicle_routes_used,
            max_route_load_t,
            capacity_utilization,
            capacity_feasible,
        ) = self._capacity_summary(
            order_indices=order_indices,
            request=request,
            points_count=len(context.points),
        )
        if not capacity_feasible:
            violated_constraints.add("vehicle_capacity")
        constraint_penalty = self._constraint_penalty(
            request=request,
            distance_km=total_distance,
            duration_min=total_duration,
            fuel_cost=fuel_cost,
            operational_cost=operational_cost,
            co2_kg=co2_kg,
            safety_risk=safety_risk_avg,
            cargo_risk=cargo_risk_avg,
        )
        penalty = (
            constraint_penalty
            + total_infrastructure_penalty
            + total_temporal_restriction_penalty
            + deadline_penalty
            + capacity_penalty
        )
        route_reliability_score = self._route_reliability_score(
            traffic_factor=congestion_index,
            weather_factor=weather_risk_avg,
            road_event_factor=dynamic_event_risk_avg,
            road_quality_factor=road_quality_risk_avg,
            constraint_penalty=penalty,
        )
        reliability_score = route_reliability_score / 100.0
        feasible = penalty <= 1e-9
        fitness_components = self._fitness_components(
            distance_km=total_distance,
            duration_min=total_duration,
            fuel_liters=liters,
            cost=operational_cost,
            weather_risk=weather_risk_avg,
            dynamic_event_risk=dynamic_event_risk_avg,
            road_quality_risk=road_quality_risk_avg,
            congestion_index=congestion_index,
            uphill_pct=uphill_pct,
            downhill_pct=downhill_pct,
            restriction_penalty=(
                total_infrastructure_penalty
                + total_temporal_restriction_penalty
                + deadline_penalty
                + capacity_penalty
            ),
            penalty=penalty,
        )

        metrics = RouteMetrics(
            distance_km=total_distance,
            duration_min=total_duration,
            fuel_liters=liters,
            fuel_cost=fuel_cost,
            operational_cost=operational_cost,
            driver_cost=driver_cost,
            maintenance_cost=maintenance_cost,
            co2_kg=co2_kg,
            congestion_index=congestion_index,
            weather_risk=weather_risk_avg,
            reliability_score=reliability_score,
            route_reliability_score=route_reliability_score,
            safety_risk=safety_risk_avg,
            toll_cost=total_toll,
            road_quality_risk=road_quality_risk_avg,
            incident_risk=incident_risk_avg,
            roadwork_risk=roadwork_risk_avg,
            dynamic_event_risk=dynamic_event_risk_avg,
            traffic_factor=congestion_index,
            weather_factor=weather_risk_avg,
            road_event_factor=dynamic_event_risk_avg,
            road_quality_factor=road_quality_risk_avg,
            cargo_risk=cargo_risk_avg,
            cargo_expected_loss=cargo_expected_loss,
            objective_score=penalty,
            constraint_penalty=penalty,
            feasible=feasible,
            infrastructure_penalty=total_infrastructure_penalty,
            temporal_restriction_penalty=total_temporal_restriction_penalty,
            deadline_penalty=deadline_penalty,
            capacity_penalty=capacity_penalty,
            vehicle_routes_used=vehicle_routes_used,
            max_route_load_t=max_route_load_t,
            capacity_utilization=capacity_utilization,
            capacity_feasible=capacity_feasible,
            refined_segments_count=refined_segments_count,
            average_detour_ratio=total_detour_ratio / edge_count,
            segment_alternative_gain_pct=total_segment_alternative_gain_pct / edge_count,
            violated_constraints=sorted(violated_constraints),
            fitness_components=fitness_components,
        )

        return CandidateEvaluation(
            order_indices=order_indices,
            metrics=metrics,
            segment_factors=segments,
            uphill_pct=uphill_pct,
            downhill_pct=downhill_pct,
            mean_elevation_m=mean_elevation_m,
            mean_temperature_c=mean_temperature_c,
        )

    @staticmethod
    def _edge_weather_snapshot(
        profiles: list[WeatherProfile],
        i: int,
        j: int,
        at: datetime,
        fallback: WeatherSnapshot,
    ) -> WeatherSnapshot:
        if not profiles:
            return fallback
        snapshots: list[WeatherSnapshot] = []
        for idx in {i, j}:
            if 0 <= idx < len(profiles):
                snapshots.append(profiles[idx].snapshot_at(at))
        if not snapshots:
            return fallback
        temperatures = [item.temperature_c for item in snapshots if item.temperature_c is not None]
        precipitations = [item.precipitation_mm for item in snapshots if item.precipitation_mm is not None]
        winds = [item.wind_speed_kph for item in snapshots if item.wind_speed_kph is not None]
        return WeatherSnapshot(
            severity=sum(item.severity for item in snapshots) / len(snapshots),
            temperature_c=(sum(temperatures) / len(temperatures)) if temperatures else None,
            precipitation_mm=(sum(precipitations) / len(precipitations)) if precipitations else None,
            wind_speed_kph=(sum(winds) / len(winds)) if winds else None,
            source=snapshots[0].source,
            source_url=snapshots[0].source_url,
            observed_at=at,
        )

    @staticmethod
    def dominance_vector(metrics: RouteMetrics) -> tuple[float, ...]:
        return (
            metrics.distance_km + metrics.constraint_penalty,
            metrics.duration_min + metrics.constraint_penalty,
            metrics.operational_cost + metrics.constraint_penalty,
            metrics.constraint_penalty,
        )

    @staticmethod
    def assign_weighted_scores(candidates: Iterable[CandidateEvaluation], weights: CriteriaWeights) -> None:
        candidates_list = list(candidates)
        if not candidates_list:
            return
        normalized_weights = CriteriaService._objective_weights(weights)
        if len(candidates_list) == 1:
            metrics = candidates_list[0].metrics
            metrics.objective_score = CriteriaService._absolute_weighted_score(metrics, normalized_weights)
            return
        mins_maxs = {
            "distance_km": CriteriaService._min_max([c.metrics.distance_km for c in candidates_list]),
            "duration_min": CriteriaService._min_max([c.metrics.duration_min for c in candidates_list]),
            "operational_cost": CriteriaService._min_max([c.metrics.operational_cost for c in candidates_list]),
            "constraint_penalty": CriteriaService._min_max([c.metrics.constraint_penalty for c in candidates_list]),
        }
        for candidate in candidates_list:
            metrics = candidate.metrics
            distance = CriteriaService._normalize(metrics.distance_km, mins_maxs["distance_km"])
            duration = CriteriaService._normalize(metrics.duration_min, mins_maxs["duration_min"])
            operational_cost = CriteriaService._normalize(metrics.operational_cost, mins_maxs["operational_cost"])
            penalty = CriteriaService._normalize(metrics.constraint_penalty, mins_maxs["constraint_penalty"])
            score = (
                normalized_weights.distance * distance
                + normalized_weights.duration * duration
                + normalized_weights.operational_cost * operational_cost
                + penalty
            )
            metrics.objective_score = score

    @staticmethod
    def _objective_weights(weights: CriteriaWeights) -> CriteriaWeights:
        objective_total = weights.distance + weights.duration + weights.operational_cost
        if objective_total <= 0:
            objective_total = 1.0
            distance = duration = operational_cost = 1.0 / 3.0
        else:
            distance = weights.distance / objective_total
            duration = weights.duration / objective_total
            operational_cost = weights.operational_cost / objective_total
        return CriteriaWeights(
            distance=distance,
            duration=duration,
            operational_cost=operational_cost,
            fuel_cost=0.0,
            emissions=0.0,
            congestion=0.0,
            weather_risk=0.0,
            reliability=0.0,
            safety=0.0,
            tolls=0.0,
            road_quality=0.0,
            dynamic_events=0.0,
            cargo_risk=0.0,
        )

    @staticmethod
    def _absolute_weighted_score(metrics: RouteMetrics, normalized_weights: CriteriaWeights) -> float:
        return (
            normalized_weights.distance * metrics.distance_km
            + normalized_weights.duration * metrics.duration_min
            + normalized_weights.operational_cost * metrics.operational_cost
            + metrics.constraint_penalty
        )

    @staticmethod
    def _fitness_components(
        *,
        distance_km: float,
        duration_min: float,
        fuel_liters: float,
        cost: float,
        weather_risk: float,
        dynamic_event_risk: float,
        road_quality_risk: float,
        congestion_index: float,
        uphill_pct: float,
        downhill_pct: float,
        restriction_penalty: float,
        penalty: float,
    ) -> RouteFitnessComponents:
        terrain_complexity = CriteriaService._clamp01((uphill_pct + downhill_pct) / 20.0)
        infrastructure_complexity = CriteriaService._clamp01(restriction_penalty / 100_000.0)
        operational_complexity = CriteriaService._clamp01(
            0.30 * road_quality_risk
            + 0.20 * congestion_index
            + 0.25 * terrain_complexity
            + 0.25 * infrastructure_complexity
        )
        return RouteFitnessComponents(
            distance=distance_km,
            time=duration_min,
            cost=cost,
            fuel=fuel_liters,
            traffic_factor=CriteriaService._clamp01(congestion_index),
            weather_factor=CriteriaService._clamp01(weather_risk),
            road_event_factor=CriteriaService._clamp01(dynamic_event_risk),
            road_quality_factor=CriteriaService._clamp01(road_quality_risk),
            restriction_penalty=restriction_penalty,
            operational_complexity=operational_complexity,
            penalty=penalty,
        )

    @staticmethod
    def _route_reliability_score(
        *,
        traffic_factor: float,
        weather_factor: float,
        road_event_factor: float,
        road_quality_factor: float,
        constraint_penalty: float,
    ) -> float:
        score = (
            100.0
            - CriteriaService._clamp01(traffic_factor) * 25.0
            - CriteriaService._clamp01(weather_factor) * 20.0
            - CriteriaService._clamp01(road_event_factor) * 25.0
            - CriteriaService._clamp01(road_quality_factor) * 15.0
            - min(35.0, max(0.0, constraint_penalty) / 10_000.0)
        )
        return max(0.0, min(100.0, score))

    @staticmethod
    def _cargo_risk_for_edge(
        *,
        request: RouteRequest,
        surface_risk: float,
        weather_risk: float,
        roadwork_risk: float,
        incident_risk: float,
        grade_risk: float,
        congestion_index: float,
        dynamic_event_risk: float,
        temporal_accessible: bool,
    ) -> float:
        profile = request.cargo.profile
        base = (
            0.24 * surface_risk
            + 0.16 * weather_risk
            + 0.13 * roadwork_risk
            + 0.13 * incident_risk
            + 0.12 * grade_risk
            + 0.10 * congestion_index
            + 0.12 * dynamic_event_risk
        )
        if not temporal_accessible:
            base += 0.45

        profile_multiplier = {
            CargoProfile.standard: 0.35,
            CargoProfile.perishable: 0.78,
            CargoProfile.fragile: 1.08,
            CargoProfile.hazardous: 1.02,
            CargoProfile.heavy: 0.92,
            CargoProfile.high_value: 0.88,
        }[profile]

        if profile == CargoProfile.perishable:
            base += 0.12 * weather_risk + 0.08 * congestion_index
        elif profile == CargoProfile.fragile:
            base += 0.18 * surface_risk + 0.10 * grade_risk
        elif profile == CargoProfile.hazardous:
            base += 0.16 * incident_risk + 0.12 * dynamic_event_risk
        elif profile == CargoProfile.heavy:
            base += 0.12 * grade_risk + 0.10 * surface_risk
        elif profile == CargoProfile.high_value:
            base += 0.14 * incident_risk + 0.08 * roadwork_risk

        return CriteriaService._clamp01(base * profile_multiplier)

    @classmethod
    def _driver_cost(cls, request: RouteRequest, duration_min: float) -> float:
        hourly_rate = request.operating_costs.driver_cost_per_hour
        if hourly_rate is None:
            hourly_rate = cls._DEFAULT_DRIVER_COST_PER_HOUR[request.vehicle_class]
        return max(0.0, duration_min / 60.0 * hourly_rate)

    @classmethod
    def _maintenance_cost(
        cls,
        *,
        request: RouteRequest,
        distance_km: float,
        road_quality_risk: float,
        dynamic_event_risk: float,
        terrain_complexity: float,
    ) -> float:
        per_km = request.operating_costs.maintenance_cost_per_km
        if per_km is None:
            per_km = cls._DEFAULT_MAINTENANCE_COST_PER_KM[request.vehicle_class]
        multiplier = (
            1.0
            + 0.45 * cls._clamp01(road_quality_risk)
            + 0.20 * cls._clamp01(dynamic_event_risk)
            + 0.25 * cls._clamp01(terrain_complexity)
        )
        return max(0.0, distance_km * per_km * multiplier)

    @classmethod
    def _cargo_expected_loss(cls, request: RouteRequest, cargo_risk: float) -> float:
        profile = request.cargo.profile
        declared_value = request.cargo.declared_value_rub
        if declared_value is None:
            declared_value = cls._DEFAULT_CARGO_VALUE_RUB[profile]
        loss_scale = cls._CARGO_LOSS_SCALE[profile]
        return max(0.0, declared_value * cls._clamp01(cargo_risk) * loss_scale)

    @staticmethod
    def _deadline_penalty(request: RouteRequest, departure_at: datetime, duration_min: float) -> float:
        deadline = request.cargo.deadline_at
        if deadline is None:
            return 0.0
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        arrival_at = departure_at + timedelta(minutes=duration_min)
        if arrival_at <= deadline:
            return 0.0
        lateness_min = (arrival_at - deadline).total_seconds() / 60.0
        profile_multiplier = {
            CargoProfile.standard: 0.75,
            CargoProfile.perishable: 1.75,
            CargoProfile.fragile: 1.05,
            CargoProfile.hazardous: 1.25,
            CargoProfile.heavy: 0.95,
            CargoProfile.high_value: 1.30,
        }[request.cargo.profile]
        return 1300.0 * profile_multiplier * (lateness_min / max(duration_min, 1.0))

    def _capacity_summary(
        self,
        *,
        order_indices: list[int],
        request: RouteRequest,
        points_count: int,
    ) -> tuple[float, int, float, float, bool]:
        demands = self._normalized_demands(request, points_count)
        total_demand = sum(demands)
        capacity_t = self._fuel_cost_service.vehicle_capacity_t(request)
        if total_demand <= 0 or capacity_t <= 0:
            return 0.0, 1, 0.0, 0.0, True

        depot_index = request.cvrp.depot_index
        if depot_index < 0 or depot_index >= points_count:
            return self._HARD_INFRASTRUCTURE_PENALTY, 1, 0.0, 0.0, False

        route_loads: list[float] = []
        current_load = 0.0
        for point_index in order_indices:
            if point_index == depot_index:
                continue
            demand = demands[point_index] if point_index < len(demands) else 0.0
            if demand <= 0:
                continue
            if current_load > 0 and current_load + demand > capacity_t:
                route_loads.append(current_load)
                current_load = 0.0
            current_load += demand
        if current_load > 0 or not route_loads:
            route_loads.append(current_load)

        routes_used = max(1, len(route_loads))
        max_route_load = max(route_loads) if route_loads else 0.0
        vehicle_overflow = max(0, routes_used - request.cvrp.vehicle_count)
        load_overflow_t = sum(max(0.0, load - capacity_t) for load in route_loads)
        penalty = 0.0
        if vehicle_overflow > 0:
            penalty += self._HARD_INFRASTRUCTURE_PENALTY * vehicle_overflow
        if load_overflow_t > 0:
            penalty += self._HARD_INFRASTRUCTURE_PENALTY * (load_overflow_t / max(capacity_t, 0.01))

        utilization = max_route_load / max(capacity_t, 0.01)
        feasible = penalty <= 1e-9
        return penalty, routes_used, max_route_load, utilization, feasible

    @staticmethod
    def _normalized_demands(request: RouteRequest, points_count: int) -> list[float]:
        demands = list(request.cvrp.point_demands_t)
        if len(demands) < points_count:
            demands.extend([0.0 for _ in range(points_count - len(demands))])
        return [max(0.0, float(value)) for value in demands[:points_count]]

    @staticmethod
    def _safe_matrix_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, float(matrix[i][j]))
        return 0.0

    @staticmethod
    def _best_segment_candidate(context: OptimizationContext, i: int, j: int):
        if not context.segment_alternatives_enabled:
            return None
        matrix = context.best_segment_choice_matrix
        if i < len(matrix) and j < len(matrix[i]):
            return matrix[i][j]
        return None

    @staticmethod
    def _safe_quality_matrix_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, min(1.0, float(matrix[i][j])))
        return 1.0

    @staticmethod
    def _safe_optional_matrix_value(matrix: list[list[float | None]], i: int, j: int) -> float | None:
        if i < len(matrix) and j < len(matrix[i]) and matrix[i][j] is not None:
            value = float(matrix[i][j])
            return value if value > 0 else None
        return None

    @staticmethod
    def _safe_access_value(matrix: list[list[bool]], i: int, j: int) -> bool:
        if i < len(matrix) and j < len(matrix[i]):
            return bool(matrix[i][j])
        return True

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _infrastructure_penalty_for_edge(
        cls,
        request: RouteRequest,
        context: OptimizationContext,
        i: int,
        j: int,
    ) -> tuple[float, list[str]]:
        dimensions = request.vehicle_dimensions
        penalty = 0.0
        violations: list[str] = []

        if not cls._safe_access_value(context.infrastructure_access_matrix, i, j):
            penalty += cls._HARD_INFRASTRUCTURE_PENALTY
            violations.append("infrastructure_access")

        height_limit = cls._safe_optional_matrix_value(context.height_clearance_matrix_m, i, j)
        if dimensions.height_m is not None and height_limit is not None and dimensions.height_m > height_limit:
            penalty += cls._scaled_hard_penalty(dimensions.height_m, height_limit)
            violations.append("height_clearance")

        weight_limit = cls._safe_optional_matrix_value(context.weight_limit_matrix_t, i, j)
        if dimensions.weight_t is not None and weight_limit is not None and dimensions.weight_t > weight_limit:
            penalty += cls._scaled_hard_penalty(dimensions.weight_t, weight_limit)
            violations.append("weight_limit")

        width_limit = cls._safe_optional_matrix_value(context.width_limit_matrix_m, i, j)
        if dimensions.width_m is not None and width_limit is not None and dimensions.width_m > width_limit:
            penalty += cls._scaled_hard_penalty(dimensions.width_m, width_limit)
            violations.append("width_limit")

        length_limit = cls._safe_optional_matrix_value(context.length_limit_matrix_m, i, j)
        if dimensions.length_m is not None and length_limit is not None and dimensions.length_m > length_limit:
            penalty += cls._scaled_hard_penalty(dimensions.length_m, length_limit)
            violations.append("length_limit")

        return penalty, violations

    @classmethod
    def _scaled_hard_penalty(cls, required_value: float, limit_value: float) -> float:
        relative_violation = max(0.0, (required_value - limit_value) / max(limit_value, 0.01))
        return cls._HARD_INFRASTRUCTURE_PENALTY * (1.0 + relative_violation)

    @staticmethod
    def _mean_elevation(elevations: list[float]) -> float | None:
        if not elevations:
            return None
        return sum(float(v) for v in elevations) / len(elevations)

    @staticmethod
    def _terrain_share_from_elevation(distance_km: float, gain_m: float, loss_m: float) -> tuple[float, float]:
        if distance_km <= 0:
            return 0.0, 0.0
        horizontal_m = distance_km * 1000.0
        uphill_pct = min(100.0, (gain_m / horizontal_m) * 100.0)
        downhill_pct = min(100.0, (loss_m / horizontal_m) * 100.0)
        if uphill_pct + downhill_pct > 100.0:
            scale = 100.0 / (uphill_pct + downhill_pct)
            uphill_pct *= scale
            downhill_pct *= scale
        return uphill_pct, downhill_pct

    @staticmethod
    def _night_factor(departure_at: datetime | None) -> float:
        dt = departure_at or datetime.now(timezone.utc)
        hour = dt.hour
        if 22 <= hour or hour <= 5:
            return 1.0
        if 6 <= hour <= 7 or 20 <= hour <= 21:
            return 0.5
        return 0.1

    @staticmethod
    def _constraint_penalty(
        request: RouteRequest,
        distance_km: float,
        duration_min: float,
        fuel_cost: float,
        operational_cost: float,
        co2_kg: float,
        safety_risk: float,
        cargo_risk: float,
    ) -> float:
        constraints = request.constraints
        penalty = 0.0
        if constraints.max_distance_km is not None and distance_km > constraints.max_distance_km:
            penalty += 1200.0 * ((distance_km - constraints.max_distance_km) / constraints.max_distance_km)
        if constraints.max_duration_min is not None and duration_min > constraints.max_duration_min:
            penalty += 1200.0 * ((duration_min - constraints.max_duration_min) / constraints.max_duration_min)
        if constraints.max_fuel_cost is not None and fuel_cost > constraints.max_fuel_cost:
            penalty += 1400.0 * ((fuel_cost - constraints.max_fuel_cost) / constraints.max_fuel_cost)
        if constraints.max_operational_cost is not None and operational_cost > constraints.max_operational_cost:
            penalty += 1450.0 * (
                (operational_cost - constraints.max_operational_cost) / constraints.max_operational_cost
            )
        if constraints.max_co2_kg is not None and co2_kg > constraints.max_co2_kg:
            penalty += 1100.0 * ((co2_kg - constraints.max_co2_kg) / constraints.max_co2_kg)
        return max(0.0, penalty)

    @staticmethod
    def _min_max(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        return min(values), max(values)

    @staticmethod
    def _normalize(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        if high - low <= 1e-9:
            return 0.0
        return (value - low) / (high - low)

    @staticmethod
    def _improvement_pct(baseline_value: float, selected_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0
        return max(0.0, ((baseline_value - selected_value) / baseline_value) * 100.0)
