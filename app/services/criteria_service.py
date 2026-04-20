from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.domain.models import CriteriaWeights, RouteMetrics, RouteRequest, RouteSegmentFactor
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
        total_gain = 0.0
        total_loss = 0.0
        weighted_mean_elevation_sum = 0.0
        weighted_mean_elevation_distance = 0.0
        segments: list[RouteSegmentFactor] = []

        for edge_idx in range(len(order_indices) - 1):
            i = order_indices[edge_idx]
            j = order_indices[edge_idx + 1]
            distance = self._safe_matrix_value(distances, i, j)
            base_duration = self._safe_matrix_value(durations, i, j)
            congestion = self._safe_matrix_value(traffic, i, j)
            provisional_duration = base_duration if base_duration > 0 else (distance / 38.0 * 60.0 if distance > 0 else 0.0)
            edge_midpoint_time = context.departure_at + timedelta(minutes=total_duration + (provisional_duration / 2.0))
            edge_weather = self._edge_weather_snapshot(weather_profiles, i, j, edge_midpoint_time, context.weather)
            weather = edge_weather.severity
            weather_delay_factor = weather * 0.35
            duration = base_duration * (1.0 + congestion + weather_delay_factor)
            if duration <= 0 and distance > 0:
                duration = distance / 38.0 * 60.0

            gain = self._safe_matrix_value(elevation_gain_matrix, i, j)
            loss = self._safe_matrix_value(elevation_loss_matrix, i, j)
            segment_mean_elevation = self._safe_matrix_value(mean_elevation_matrix, i, j)
            total_gain += gain
            total_loss += loss
            if distance > 0 and segment_mean_elevation > 0:
                weighted_mean_elevation_sum += segment_mean_elevation * distance
                weighted_mean_elevation_distance += distance
            if distance > 0:
                total_weather_weight += distance
                total_weighted_weather += weather * distance
                if edge_weather.temperature_c is not None:
                    total_weighted_temperature += edge_weather.temperature_c * distance
                    total_temperature_weight += distance

            incident_proxy = max(0.0, min(1.0, 0.1 + 0.25 * congestion + 0.2 * weather))
            reliability_risk = max(0.0, min(1.0, 0.45 * congestion + 0.35 * weather + 0.2 * incident_proxy))
            safety_risk = max(0.0, min(1.0, 0.5 * weather + 0.3 * congestion + 0.2 * night_factor))
            toll_cost = self._safe_matrix_value(tolls, i, j)

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
                )
            )

            total_distance += distance
            total_duration += duration
            total_congestion += congestion
            total_reliability_risk += reliability_risk
            total_safety_risk += safety_risk
            total_toll += toll_cost

        edge_count = max(1, len(order_indices) - 1)
        congestion_index = total_congestion / edge_count
        weather_risk_avg = (
            total_weighted_weather / total_weather_weight if total_weather_weight > 0 else context.weather.severity
        )
        mean_temperature_c = (
            total_weighted_temperature / total_temperature_weight
            if total_temperature_weight > 0
            else context.weather.temperature_c
        )
        reliability_risk_avg = total_reliability_risk / edge_count
        safety_risk_avg = total_safety_risk / edge_count
        reliability_score = max(0.0, min(1.0, 1.0 - reliability_risk_avg))

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
        price_per_liter = self._fuel_cost_service.price_per_liter(fuel_prices, request.fuel_type)
        fuel_cost = liters * price_per_liter + total_toll
        co2_kg = self._fuel_cost_service.estimate_co2_kg(liters, request.fuel_type)

        penalty = self._constraint_penalty(
            request=request,
            distance_km=total_distance,
            duration_min=total_duration,
            fuel_cost=fuel_cost,
            co2_kg=co2_kg,
            safety_risk=safety_risk_avg,
        )
        feasible = penalty <= 1e-9

        metrics = RouteMetrics(
            distance_km=total_distance,
            duration_min=total_duration,
            fuel_liters=liters,
            fuel_cost=fuel_cost,
            co2_kg=co2_kg,
            congestion_index=congestion_index,
            weather_risk=weather_risk_avg,
            reliability_score=reliability_score,
            safety_risk=safety_risk_avg,
            toll_cost=total_toll,
            objective_score=penalty,
            constraint_penalty=penalty,
            feasible=feasible,
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
            metrics.fuel_cost + metrics.constraint_penalty,
            metrics.co2_kg + metrics.constraint_penalty,
            metrics.congestion_index + metrics.constraint_penalty,
            metrics.weather_risk + metrics.constraint_penalty,
            (1.0 - metrics.reliability_score) + metrics.constraint_penalty,
            metrics.safety_risk + metrics.constraint_penalty,
            metrics.toll_cost + metrics.constraint_penalty,
        )

    @staticmethod
    def assign_weighted_scores(candidates: Iterable[CandidateEvaluation], weights: CriteriaWeights) -> None:
        candidates_list = list(candidates)
        if not candidates_list:
            return
        normalized_weights = weights.normalized()
        if len(candidates_list) == 1:
            metrics = candidates_list[0].metrics
            metrics.objective_score = CriteriaService._absolute_weighted_score(metrics, normalized_weights)
            return
        mins_maxs = {
            "distance_km": CriteriaService._min_max([c.metrics.distance_km for c in candidates_list]),
            "duration_min": CriteriaService._min_max([c.metrics.duration_min for c in candidates_list]),
            "fuel_cost": CriteriaService._min_max([c.metrics.fuel_cost for c in candidates_list]),
            "co2_kg": CriteriaService._min_max([c.metrics.co2_kg for c in candidates_list]),
            "congestion_index": CriteriaService._min_max([c.metrics.congestion_index for c in candidates_list]),
            "weather_risk": CriteriaService._min_max([c.metrics.weather_risk for c in candidates_list]),
            "reliability_risk": CriteriaService._min_max([1.0 - c.metrics.reliability_score for c in candidates_list]),
            "safety_risk": CriteriaService._min_max([c.metrics.safety_risk for c in candidates_list]),
            "toll_cost": CriteriaService._min_max([c.metrics.toll_cost for c in candidates_list]),
            "constraint_penalty": CriteriaService._min_max([c.metrics.constraint_penalty for c in candidates_list]),
        }
        for candidate in candidates_list:
            metrics = candidate.metrics
            distance = CriteriaService._normalize(metrics.distance_km, mins_maxs["distance_km"])
            duration = CriteriaService._normalize(metrics.duration_min, mins_maxs["duration_min"])
            fuel_cost = CriteriaService._normalize(metrics.fuel_cost, mins_maxs["fuel_cost"])
            co2 = CriteriaService._normalize(metrics.co2_kg, mins_maxs["co2_kg"])
            congestion = CriteriaService._normalize(metrics.congestion_index, mins_maxs["congestion_index"])
            weather = CriteriaService._normalize(metrics.weather_risk, mins_maxs["weather_risk"])
            reliability = CriteriaService._normalize(1.0 - metrics.reliability_score, mins_maxs["reliability_risk"])
            safety = CriteriaService._normalize(metrics.safety_risk, mins_maxs["safety_risk"])
            toll = CriteriaService._normalize(metrics.toll_cost, mins_maxs["toll_cost"])
            penalty = CriteriaService._normalize(metrics.constraint_penalty, mins_maxs["constraint_penalty"])
            score = (
                normalized_weights.distance * distance
                + normalized_weights.duration * duration
                + normalized_weights.fuel_cost * fuel_cost
                + normalized_weights.emissions * co2
                + normalized_weights.congestion * congestion
                + normalized_weights.weather_risk * weather
                + normalized_weights.reliability * reliability
                + normalized_weights.safety * safety
                + normalized_weights.tolls * toll
                + penalty
            )
            metrics.objective_score = score

    @staticmethod
    def _absolute_weighted_score(metrics: RouteMetrics, normalized_weights: CriteriaWeights) -> float:
        return (
            normalized_weights.distance * metrics.distance_km
            + normalized_weights.duration * metrics.duration_min
            + normalized_weights.fuel_cost * metrics.fuel_cost
            + normalized_weights.emissions * metrics.co2_kg
            + normalized_weights.congestion * metrics.congestion_index
            + normalized_weights.weather_risk * metrics.weather_risk
            + normalized_weights.reliability * (1.0 - metrics.reliability_score)
            + normalized_weights.safety * metrics.safety_risk
            + normalized_weights.tolls * metrics.toll_cost
            + metrics.constraint_penalty
        )

    @staticmethod
    def _safe_matrix_value(matrix: list[list[float]], i: int, j: int) -> float:
        if i < len(matrix) and j < len(matrix[i]):
            return max(0.0, float(matrix[i][j]))
        return 0.0

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
        co2_kg: float,
        safety_risk: float,
    ) -> float:
        constraints = request.constraints
        penalty = 0.0
        if constraints.max_distance_km is not None and distance_km > constraints.max_distance_km:
            penalty += 1200.0 * ((distance_km - constraints.max_distance_km) / constraints.max_distance_km)
        if constraints.max_duration_min is not None and duration_min > constraints.max_duration_min:
            penalty += 1200.0 * ((duration_min - constraints.max_duration_min) / constraints.max_duration_min)
        if constraints.max_fuel_cost is not None and fuel_cost > constraints.max_fuel_cost:
            penalty += 1400.0 * ((fuel_cost - constraints.max_fuel_cost) / constraints.max_fuel_cost)
        if constraints.max_co2_kg is not None and co2_kg > constraints.max_co2_kg:
            penalty += 1100.0 * ((co2_kg - constraints.max_co2_kg) / constraints.max_co2_kg)
        if constraints.max_safety_risk is not None and safety_risk > constraints.max_safety_risk:
            denom = max(constraints.max_safety_risk, 0.01)
            penalty += 1600.0 * ((safety_risk - constraints.max_safety_risk) / denom)
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
