from __future__ import annotations

import asyncio
import math
import json
import logging
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
from app.services.route_analysis_service import RouteAnalysisService
from app.services.route_optimizer import RouteOptimizer
from app.services.terrain_profile_service import TerrainProfileService

logger = logging.getLogger(__name__)


class RouteService:
    _ADAPTIVE_GA_BUDGETS: tuple[tuple[int, int, int, int], ...] = (
        (21, 48, 56, 4),
        (15, 60, 72, 5),
        (11, 72, 88, 6),
        (7, 84, 104, 8),
    )

    def __init__(
        self,
        optimizer: RouteOptimizer,
        routing_repository: RoutingRepository,
        cache_repository: RouteCacheRepository,
        fuel_cost_service: FuelCostService,
        context_service: ContextService,
        dynamic_weights_service: DynamicWeightsService,
        route_analysis_service: RouteAnalysisService,
        terrain_profile_service: TerrainProfileService,
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
        self._route_analysis_service = route_analysis_service
        self._terrain_profile_service = terrain_profile_service
        self._run_repository = run_repository
        self._default_population = default_population
        self._default_generations = default_generations
        self._pareto_enabled = pareto_enabled

    async def compute_route(self, request: RouteRequest) -> RouteResponse:
        request = self._apply_runtime_defaults(request)
        logger.warning(
            "Route request debug: points=%d labels=%s optimize=%s fix_ends=%s profile=%s vehicle_class=%s fuel_type=%s fuel_consumption=%s optimize_mode=%s priority_profile=%s use_dynamic_weights=%s departure_at=%s population_size=%d generations=%d crossover_rate=%.3f mutation_rate=%.3f max_alternatives=%d random_seed=%s constraints=%s criteria_weights=%s",
            len(request.points),
            [point.label for point in request.points],
            request.optimize,
            request.fix_ends,
            request.profile.value,
            request.vehicle_class.value,
            request.fuel_type.value,
            request.fuel_consumption_l_per_100km,
            request.optimize_mode.value,
            request.priority_profile.value,
            request.use_dynamic_weights,
            request.departure_at.isoformat() if request.departure_at else None,
            request.population_size,
            request.generations,
            request.crossover_rate,
            request.mutation_rate,
            request.max_alternatives,
            request.random_seed,
            request.constraints.model_dump(mode="json"),
            request.criteria_weights.model_dump(mode="json"),
        )
        cache_key = self._make_cache_key(request)
        cached = self._cache_repository.get(cache_key)
        if cached is not None:
            logger.warning("Route cache hit: points=%d key_hash=%d", len(request.points), hash(cache_key))
            return cached

        context = await self._context_service.build(request)
        fuel_prices = await self._fuel_cost_service.get_price_snapshot()
        price_per_liter = self._fuel_cost_service.price_per_liter(fuel_prices, request.fuel_type)
        logger.warning(
            "Route fuel price debug: source=%s currency=%s petrol=%.3f diesel=%.3f selected_price=%.3f price_date=%s retrieved_at=%s",
            getattr(fuel_prices, "source", type(fuel_prices).__name__),
            getattr(fuel_prices, "currency", None),
            float(getattr(fuel_prices, "petrol_rub_per_liter", 0.0)),
            float(getattr(fuel_prices, "diesel_rub_per_liter", 0.0)),
            price_per_liter,
            getattr(fuel_prices, "price_date", None),
            getattr(getattr(fuel_prices, "retrieved_at", None), "isoformat", lambda: None)(),
        )
        dynamic_weights = self._dynamic_weights_service.compute(request, context, price_per_liter)
        logger.warning(
            "Dynamic weights debug: triggers=%s base=%s adjusted=%s",
            dynamic_weights.triggers,
            dynamic_weights.base.model_dump(mode="json"),
            dynamic_weights.adjusted.model_dump(mode="json"),
        )

        optimization, baseline_evaluation = await asyncio.gather(
            asyncio.to_thread(
                self._optimizer.optimize,
                request,
                context,
                dynamic_weights.adjusted,
                fuel_prices,
            ),
            asyncio.to_thread(
                self._optimizer.evaluate_order,
                request,
                context,
                list(range(len(context.points))),
                dynamic_weights.adjusted,
                fuel_prices,
            ),
        )
        logger.warning(
            "Optimization debug: best_order=%s best_distance_km=%.3f best_duration_min=%.3f best_fuel_cost=%.3f best_weather_risk=%.3f best_congestion=%.3f best_safety=%.3f best_penalty=%.3f baseline_order=%s baseline_distance_km=%.3f baseline_duration_min=%.3f baseline_fuel_cost=%.3f diagnostics=%s pareto_candidates=%d",
            optimization.best.order_indices,
            optimization.best.metrics.distance_km,
            optimization.best.metrics.duration_min,
            optimization.best.metrics.fuel_cost,
            optimization.best.metrics.weather_risk,
            optimization.best.metrics.congestion_index,
            optimization.best.metrics.safety_risk,
            optimization.best.metrics.constraint_penalty,
            baseline_evaluation.order_indices,
            baseline_evaluation.metrics.distance_km,
            baseline_evaluation.metrics.duration_min,
            baseline_evaluation.metrics.fuel_cost,
            optimization.diagnostics.model_dump(mode="json"),
            len(optimization.pareto),
        )

        best_order_points = [context.points[idx] for idx in optimization.best.order_indices]
        baseline_order_points = [context.points[idx] for idx in baseline_evaluation.order_indices]

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

        provider_result, baseline_provider_result, alternatives = await asyncio.gather(
            self._routing_repository.route(best_order_points, request.profile),
            self._routing_repository.route(baseline_order_points, request.profile),
            self._hydrate_alternative_routes(alternatives, request.profile),
        )
        logger.warning(
            "Route geometry debug: optimized_provider=%s optimized_geometry_points=%d optimized_distance_km=%.3f optimized_duration_min=%s baseline_provider=%s baseline_geometry_points=%d baseline_distance_km=%.3f baseline_duration_min=%s alternatives=%d alt_geometry_sizes=%s",
            provider_result.provider,
            len(provider_result.geometry),
            provider_result.distance_km,
            None if provider_result.duration_min is None else round(provider_result.duration_min, 3),
            baseline_provider_result.provider,
            len(baseline_provider_result.geometry),
            baseline_provider_result.distance_km,
            None if baseline_provider_result.duration_min is None else round(baseline_provider_result.duration_min, 3),
            len(alternatives),
            [len(alt.geometry) for alt in alternatives[: min(6, len(alternatives))]],
        )

        terrain_profile, baseline_terrain_profile, alternatives = await asyncio.gather(
            self._terrain_profile_service.build_route_profile(provider_result.geometry),
            self._terrain_profile_service.build_route_profile(baseline_provider_result.geometry),
            self._hydrate_alternative_terrain(alternatives),
        )

        comparison = self._build_comparison_info(
            context=context,
            baseline_metrics=baseline_evaluation.metrics,
            baseline_order_indices=baseline_evaluation.order_indices,
            baseline_geometry=baseline_provider_result.geometry,
            baseline_terrain_profile=baseline_terrain_profile,
            optimized_metrics=optimization.best.metrics,
            weights=dynamic_weights.adjusted,
        )

        fuel_cost, segment_insights, stress_test = await asyncio.gather(
            self._fuel_cost_service.compute(
                request,
                optimization.best.metrics.distance_km,
                uphill_pct=optimization.best.uphill_pct,
                downhill_pct=optimization.best.downhill_pct,
                temperature_c=optimization.best.mean_temperature_c,
                congestion_index=optimization.best.metrics.congestion_index,
                mean_elevation_m=optimization.best.mean_elevation_m,
            ),
            asyncio.to_thread(
                self._route_analysis_service.build_segment_insights,
                best_order_points,
                optimization.best.segment_factors,
            ),
            asyncio.to_thread(
                self._route_analysis_service.build_stress_test,
                request,
                optimization.best.metrics,
                optimization.best.segment_factors,
            ),
        )
        logger.warning(
            "Analysis debug: segment_factors=%d segment_insights=%d stress_simulations=%s resilience_index=%s on_time_probability=%s within_budget_probability=%s within_safety_probability=%s",
            len(optimization.best.segment_factors),
            len(segment_insights),
            None if stress_test is None else stress_test.simulations,
            None if stress_test is None else round(stress_test.resilience_index, 4),
            None if stress_test is None else round(stress_test.on_time_probability, 4),
            None if stress_test is None else round(stress_test.within_budget_probability, 4),
            None if stress_test is None else round(stress_test.within_safety_probability, 4),
        )

        data_sources = context.data_sources.model_copy(
            update={
                "routing": provider_result.provider,
                "matrix": optimization.matrix_provider,
                "elevation": terrain_profile.source if terrain_profile else context.data_sources.elevation,
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
            segment_insights=segment_insights,
            terrain_profile=terrain_profile,
            stress_test=stress_test,
            diagnostics=optimization.diagnostics,
            dynamic_weights=dynamic_weights,
            comparison=comparison,
            data_sources=data_sources,
        )
        logger.warning(
            "Route response debug: run_provider=%s ordered_points=%d metrics=%s fuel_cost=%s data_sources=%s terrain_segments=%d gain=%.2f loss=%.2f max_up=%.4f max_down=%.4f baseline_segments=%d baseline_gain=%.2f baseline_loss=%.2f",
            response.provider,
            len(response.ordered_points),
            response.metrics.model_dump(mode="json") if response.metrics else None,
            response.fuel_cost.model_dump(mode="json") if response.fuel_cost else None,
            response.data_sources.model_dump(mode="json") if response.data_sources else None,
            len(response.terrain_profile.segments) if response.terrain_profile else 0,
            response.terrain_profile.total_gain_m if response.terrain_profile else 0.0,
            response.terrain_profile.total_loss_m if response.terrain_profile else 0.0,
            response.terrain_profile.max_uphill_grade_pct if response.terrain_profile else 0.0,
            response.terrain_profile.max_downhill_grade_pct if response.terrain_profile else 0.0,
            len(response.comparison.baseline_terrain_profile.segments)
            if response.comparison and response.comparison.baseline_terrain_profile
            else 0,
            response.comparison.baseline_terrain_profile.total_gain_m
            if response.comparison and response.comparison.baseline_terrain_profile
            else 0.0,
            response.comparison.baseline_terrain_profile.total_loss_m
            if response.comparison and response.comparison.baseline_terrain_profile
            else 0.0,
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
        alternatives_default = RouteRequest.model_fields["max_alternatives"].default
        if request.population_size == pop_default and self._default_population != pop_default:
            updates["population_size"] = self._default_population
        if request.generations == gen_default and self._default_generations != gen_default:
            updates["generations"] = self._default_generations
        if not self._pareto_enabled and request.optimize_mode == OptimizationMode.pareto:
            updates["optimize_mode"] = OptimizationMode.weighted
        adaptive_updates = self._adaptive_ga_updates(
            request.model_copy(
                update={
                    "population_size": updates.get("population_size", request.population_size),
                    "generations": updates.get("generations", request.generations),
                    "max_alternatives": updates.get("max_alternatives", request.max_alternatives),
                }
            ),
            pop_default=pop_default,
            gen_default=gen_default,
            alternatives_default=alternatives_default,
        )
        updates.update(adaptive_updates)
        if not updates:
            return request
        return request.model_copy(update=updates)

    def _adaptive_ga_updates(
        self,
        request: RouteRequest,
        *,
        pop_default: int,
        gen_default: int,
        alternatives_default: int,
    ) -> dict:
        if not request.optimize:
            return {}

        movable_points = max(0, len(request.points) - 2) if request.fix_ends else len(request.points)
        if movable_points <= 6:
            return {}

        updates: dict = {}
        for minimum_points, population_cap, generations_cap, alternatives_cap in self._ADAPTIVE_GA_BUDGETS:
            if movable_points >= minimum_points:
                if request.population_size == pop_default:
                    updates["population_size"] = min(request.population_size, population_cap)
                if request.generations == gen_default:
                    updates["generations"] = min(request.generations, generations_cap)
                if request.max_alternatives == alternatives_default:
                    updates["max_alternatives"] = min(request.max_alternatives, alternatives_cap)
                break
        return updates

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
        baseline_geometry: list[list[float]],
        baseline_terrain_profile,
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
            baseline_geometry=baseline_geometry,
            baseline_metrics=baseline_metrics,
            baseline_terrain_profile=baseline_terrain_profile,
            optimized_metrics=optimized_metrics,
            delta=delta,
            improvement_pct=improvement_pct,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
        )

    async def _hydrate_alternative_routes(
        self,
        alternatives: list[RouteAlternative],
        profile,
    ) -> list[RouteAlternative]:
        if not alternatives:
            return alternatives

        route_results = await asyncio.gather(
            *(self._routing_repository.route(alt.ordered_points, profile) for alt in alternatives)
        )
        return [
            alt.model_copy(
                update={
                    "geometry": route_result.geometry,
                    "provider": route_result.provider,
                }
            )
            for alt, route_result in zip(alternatives, route_results, strict=False)
        ]

    async def _hydrate_alternative_terrain(self, alternatives: list[RouteAlternative]) -> list[RouteAlternative]:
        if not alternatives:
            return alternatives

        terrain_profiles = await asyncio.gather(
            *(self._terrain_profile_service.build_route_profile(alt.geometry) for alt in alternatives)
        )
        return [
            alt.model_copy(update={"terrain_profile": terrain_profile})
            for alt, terrain_profile in zip(alternatives, terrain_profiles, strict=False)
        ]

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
