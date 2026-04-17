from __future__ import annotations

import math
import json
from typing import Callable

from app.domain.models import (
    CriteriaWeights,
    RouteComparisonDelta,
    RouteComparisonInfo,
    RouteMetrics,
    ScoreComponent,
    ScoreExplanation,
    ScoreMode,
    OptimizationMode,
    RouteAlternative,
    RouteRequest,
    RouteResponse,
    RouteRunDetails,
    RouteRunListItem,
)
from app.repositories.cache_repository import RouteCacheRepository
from app.repositories.routing_repository import RoutingRepository
from app.repositories.run_repository import RouteRunRepository
from app.services.context_service import ContextService
from app.services.dynamic_weights_service import DynamicWeightsService
from app.services.fuel_cost import FuelCostService
from app.services.route_optimizer import RouteOptimizer


class RouteService:
    def __init__(
        self,
        optimizer: RouteOptimizer,
        routing_repository: RoutingRepository,
        cache_repository: RouteCacheRepository,
        fuel_cost_service: FuelCostService,
        context_service: ContextService,
        dynamic_weights_service: DynamicWeightsService,
        run_repository: RouteRunRepository,
        default_population: int = 96,
        default_generations: int = 120,
        pareto_enabled: bool = True,
    ) -> None:
        self._optimizer = optimizer
        self._routing_repository = routing_repository
        self._cache_repository = cache_repository
        self._fuel_cost_service = fuel_cost_service
        self._context_service = context_service
        self._dynamic_weights_service = dynamic_weights_service
        self._run_repository = run_repository
        self._default_population = default_population
        self._default_generations = default_generations
        self._pareto_enabled = pareto_enabled

    async def compute_route(self, request: RouteRequest) -> RouteResponse:
        request = self._apply_runtime_defaults(request)
        cache_key = self._make_cache_key(request)
        cached = self._cache_repository.get(cache_key)
        if cached is not None:
            return cached

        context = await self._context_service.build(request)
        fuel_prices = await self._fuel_cost_service.get_price_snapshot()
        price_per_liter = self._fuel_cost_service.price_per_liter(fuel_prices, request.fuel_type)
        dynamic_weights = self._dynamic_weights_service.compute(request, context, price_per_liter)

        optimization = self._optimizer.optimize(
            request=request,
            context=context,
            weights=dynamic_weights.adjusted,
            fuel_prices=fuel_prices,
        )
        baseline_evaluation = self._optimizer.evaluate_order(
            request=request,
            context=context,
            order_indices=list(range(len(context.points))),
            weights=dynamic_weights.adjusted,
            fuel_prices=fuel_prices,
        )
        comparison = self._build_comparison_info(
            context=context,
            baseline_metrics=baseline_evaluation.metrics,
            baseline_order_indices=baseline_evaluation.order_indices,
            optimized_metrics=optimization.best.metrics,
            weights=dynamic_weights.adjusted,
        )

        best_order_points = [context.points[idx] for idx in optimization.best.order_indices]
        provider_result = await self._routing_repository.route(best_order_points, request.profile)

        fuel_cost = await self._fuel_cost_service.compute(
            request,
            optimization.best.metrics.distance_km,
            uphill_pct=optimization.best.uphill_pct,
            downhill_pct=optimization.best.downhill_pct,
            temperature_c=context.weather.temperature_c,
            congestion_index=optimization.best.metrics.congestion_index,
            mean_elevation_m=self._mean_elevation(context.elevation.elevations_m),
        )

        alternatives = [
            RouteAlternative(
                ordered_points=[context.points[idx] for idx in item.evaluation.order_indices],
                metrics=item.evaluation.metrics,
                rank=item.rank + 1,
                crowding_distance=self._json_safe_float(item.crowding),
            )
            for item in optimization.pareto
        ]
        alternatives = self._deduplicate_alternatives(alternatives)

        data_sources = context.data_sources.model_copy(
            update={
                "matrix": optimization.matrix_provider,
                "fuel_prices": fuel_cost.price_source,
            }
        )

        response = RouteResponse(
            ordered_points=best_order_points,
            total_distance_km=optimization.best.metrics.distance_km,
            total_duration_min=optimization.best.metrics.duration_min,
            geometry=provider_result.geometry,
            geojson=self._as_geojson(provider_result.geometry, provider_result.provider),
            provider=provider_result.provider,
            fuel_cost=fuel_cost,
            metrics=optimization.best.metrics,
            alternatives=alternatives,
            segment_factors=optimization.best.segment_factors,
            diagnostics=optimization.diagnostics,
            dynamic_weights=dynamic_weights,
            comparison=comparison,
            data_sources=data_sources,
        )

        run_id = self._run_repository.save(request=request, response=response)
        response = response.model_copy(update={"run_id": run_id})

        self._cache_repository.set(cache_key, response)
        return response

    def list_runs(self, limit: int = 20) -> list[RouteRunListItem]:
        return self._run_repository.list_runs(limit=limit)

    def get_run(self, run_id: str) -> RouteRunDetails | None:
        return self._run_repository.get_run(run_id)

    def export_run_csv(self, run_id: str) -> str | None:
        return self._run_repository.export_csv(run_id)

    def _make_cache_key(self, request: RouteRequest) -> str:
        payload = request.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _as_geojson(self, geometry: list[list[float]], provider: str) -> dict:
        coordinates = [[lon, lat] for lat, lon in geometry]
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {"provider": provider},
        }

    def _apply_runtime_defaults(self, request: RouteRequest) -> RouteRequest:
        updates: dict = {}
        pop_default = RouteRequest.model_fields["population_size"].default
        gen_default = RouteRequest.model_fields["generations"].default
        if request.population_size == pop_default and self._default_population != pop_default:
            updates["population_size"] = self._default_population
        if request.generations == gen_default and self._default_generations != gen_default:
            updates["generations"] = self._default_generations
        if not self._pareto_enabled and request.optimize_mode == OptimizationMode.pareto:
            updates["optimize_mode"] = OptimizationMode.weighted
        if not updates:
            return request
        return request.model_copy(update=updates)

    @staticmethod
    def _json_safe_float(value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _mean_elevation(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(float(v) for v in values) / len(values)

    @staticmethod
    def _deduplicate_alternatives(alternatives: list[RouteAlternative]) -> list[RouteAlternative]:
        unique: list[RouteAlternative] = []
        seen: set[str] = set()
        for alt in alternatives:
            key = RouteService._alternative_key(alt)
            if key in seen:
                continue
            seen.add(key)
            unique.append(alt)
        ranked: list[RouteAlternative] = []
        for index, alt in enumerate(unique, start=1):
            ranked.append(alt.model_copy(update={"rank": index}))
        return ranked

    @staticmethod
    def _alternative_key(alt: RouteAlternative) -> str:
        order = "|".join([f"{point.lat:.6f},{point.lon:.6f}" for point in alt.ordered_points])
        metrics = alt.metrics
        metrics_key = (
            f"{metrics.distance_km:.3f}|{metrics.duration_min:.3f}|{metrics.fuel_cost:.3f}|"
            f"{metrics.co2_kg:.3f}|{metrics.objective_score:.5f}|{int(metrics.feasible)}"
        )
        return f"{order}::{metrics_key}"

    def _build_comparison_info(
        self,
        context,
        baseline_metrics: RouteMetrics,
        baseline_order_indices: list[int],
        optimized_metrics: RouteMetrics,
        weights: CriteriaWeights,
    ) -> RouteComparisonInfo:
        baseline_score, optimized_score = self._build_pair_score_explanations(
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            weights=weights,
        )
        delta = RouteComparisonDelta(
            distance_km=optimized_metrics.distance_km - baseline_metrics.distance_km,
            duration_min=optimized_metrics.duration_min - baseline_metrics.duration_min,
            fuel_cost=optimized_metrics.fuel_cost - baseline_metrics.fuel_cost,
            co2_kg=optimized_metrics.co2_kg - baseline_metrics.co2_kg,
            objective_score=optimized_metrics.objective_score - baseline_metrics.objective_score,
        )
        improvement_pct = RouteComparisonDelta(
            distance_km=self._improvement_pct(baseline_metrics.distance_km, optimized_metrics.distance_km),
            duration_min=self._improvement_pct(baseline_metrics.duration_min, optimized_metrics.duration_min),
            fuel_cost=self._improvement_pct(baseline_metrics.fuel_cost, optimized_metrics.fuel_cost),
            co2_kg=self._improvement_pct(baseline_metrics.co2_kg, optimized_metrics.co2_kg),
            objective_score=self._improvement_pct(
                baseline_metrics.objective_score,
                optimized_metrics.objective_score,
            ),
        )
        return RouteComparisonInfo(
            baseline_ordered_points=[context.points[idx] for idx in baseline_order_indices],
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            delta=delta,
            improvement_pct=improvement_pct,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
        )

    def _build_pair_score_explanations(
        self,
        baseline_metrics: RouteMetrics,
        optimized_metrics: RouteMetrics,
        weights: CriteriaWeights,
    ) -> tuple[ScoreExplanation, ScoreExplanation]:
        normalized_weights = weights.normalized()

        component_specs: list[tuple[str, str, Callable[[RouteMetrics], float], float]] = [
            ("distance", "Distance", lambda m: m.distance_km, normalized_weights.distance),
            ("duration", "Duration", lambda m: m.duration_min, normalized_weights.duration),
            ("fuel_cost", "Fuel cost", lambda m: m.fuel_cost, normalized_weights.fuel_cost),
            ("emissions", "Emissions", lambda m: m.co2_kg, normalized_weights.emissions),
            ("congestion", "Congestion", lambda m: m.congestion_index, normalized_weights.congestion),
            ("weather_risk", "Weather risk", lambda m: m.weather_risk, normalized_weights.weather_risk),
            ("reliability", "Reliability risk", lambda m: 1.0 - m.reliability_score, normalized_weights.reliability),
            ("safety", "Safety risk", lambda m: m.safety_risk, normalized_weights.safety),
            ("tolls", "Tolls", lambda m: m.toll_cost, normalized_weights.tolls),
            ("constraint_penalty", "Constraint penalty", lambda m: m.constraint_penalty, 1.0),
        ]

        bounds: dict[str, tuple[float, float]] = {}
        for key, _label, get_value, _weight in component_specs:
            baseline_value = get_value(baseline_metrics)
            optimized_value = get_value(optimized_metrics)
            bounds[key] = (
                min(baseline_value, optimized_value),
                max(baseline_value, optimized_value),
            )

        baseline_components = self._build_components(
            metrics=baseline_metrics,
            component_specs=component_specs,
            bounds=bounds,
        )
        optimized_components = self._build_components(
            metrics=optimized_metrics,
            component_specs=component_specs,
            bounds=bounds,
        )
        baseline_total = sum(item.contribution for item in baseline_components)
        optimized_total = sum(item.contribution for item in optimized_components)

        return (
            ScoreExplanation(
                score_mode=ScoreMode.population_normalized,
                total_score=baseline_total,
                components=baseline_components,
            ),
            ScoreExplanation(
                score_mode=ScoreMode.population_normalized,
                total_score=optimized_total,
                components=optimized_components,
            ),
        )

    @staticmethod
    def _build_components(
        metrics: RouteMetrics,
        component_specs: list[tuple[str, str, Callable[[RouteMetrics], float], float]],
        bounds: dict[str, tuple[float, float]],
    ) -> list[ScoreComponent]:
        items: list[ScoreComponent] = []
        for key, label, get_value, weight in component_specs:
            raw_value = float(get_value(metrics))
            normalized_value = RouteService._normalize(raw_value, bounds[key])
            contribution = weight * normalized_value
            items.append(
                ScoreComponent(
                    key=key,
                    label=label,
                    weight=weight,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    contribution=contribution,
                )
            )
        return items

    @staticmethod
    def _normalize(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        if high - low <= 1e-12:
            return 0.0
        return (value - low) / (high - low)

    @staticmethod
    def _improvement_pct(baseline_value: float, optimized_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0
        return ((baseline_value - optimized_value) / baseline_value) * 100.0
