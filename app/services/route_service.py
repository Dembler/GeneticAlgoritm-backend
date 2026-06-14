from __future__ import annotations

import asyncio
from dataclasses import replace
import math
import json
import logging
from time import perf_counter
from typing import Callable

from app.domain.models import (
    ConstraintHealthItem,
    ConstraintHealthReport,
    CriteriaWeights,
    CvrpPlanInfo,
    DataConfidenceScore,
    OperationalCostBreakdown,
    PerformanceTimings,
    RouteComparisonDelta,
    RouteComparisonInfo,
    RouteComparisonRouteView,
    RouteComparisonSummary,
    RouteMetrics,
    RouteQualityComponent,
    RouteQualityIndex,
    ScoreComponent,
    ScoreExplanation,
    ScoreMode,
    OptimizationMode,
    RouteAlternative,
    RouteAnalysisMatrices,
    RouteRequest,
    RouteResponse,
    RouteRunDetails,
    RouteRunListItem,
    RouteVehicleRoute,
)
from app.repositories.cache_repository import RouteCacheRepository
from app.repositories.routing_repository import RoutingRepository
from app.repositories.routing_repository import RouteProviderResult
from app.repositories.run_repository import RouteRunRepository
from app.services.context_service import ContextService
from app.services.criteria_service import CriteriaService
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.dynamic_weights_service import DynamicWeightsService
from app.services.fuel_cost import FuelCostService
from app.services.route_analysis_service import RouteAnalysisService
from app.services.route_optimizer import RouteOptimizer
from app.services.route_refinement_service import RouteRefinementService
from app.services.segment_alternative_service import SegmentAlternativeService
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
        decision_explanation_service: DecisionExplanationService,
        terrain_profile_service: TerrainProfileService,
        run_repository: RouteRunRepository,
        route_refinement_service: RouteRefinementService | None = None,
        segment_alternative_service: SegmentAlternativeService | None = None,
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
        self._decision_explanation_service = decision_explanation_service
        self._terrain_profile_service = terrain_profile_service
        self._run_repository = run_repository
        self._route_refinement_service = route_refinement_service
        self._segment_alternative_service = segment_alternative_service
        self._default_population = default_population
        self._default_generations = default_generations
        self._pareto_enabled = pareto_enabled

    async def compute_route(self, request: RouteRequest) -> RouteResponse:
        total_start = perf_counter()
        context_ms = 0.0
        fuel_price_ms = 0.0
        segment_alternatives_ms = 0.0
        optimization_ms = 0.0
        geometry_ms = 0.0
        refinement_ms = 0.0
        analysis_ms = 0.0
        cvrp_ms = 0.0
        request = self._apply_runtime_defaults(request)
        request = self._attach_warm_start_orders(request)
        logger.warning(
            "Route request debug: points=%d labels=%s optimize=%s fix_ends=%s profile=%s vehicle_class=%s fuel_type=%s fuel_consumption=%s optimize_mode=%s priority_profile=%s use_dynamic_weights=%s departure_at=%s population_size=%d generations=%d crossover_rate=%.3f mutation_rate=%.3f max_alternatives=%d random_seed=%s adapt_from_run_id=%s warm_start_seeds=%d constraints=%s criteria_weights=%s",
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
            request.adapt_from_run_id,
            len(request.warm_start_orders),
            request.constraints.model_dump(mode="json"),
            request.criteria_weights.model_dump(mode="json"),
        )
        cache_key = self._make_cache_key(request)
        cached = self._cache_repository.get(cache_key)
        if cached is not None:
            logger.warning("Route cache hit: points=%d key_hash=%d", len(request.points), hash(cache_key))
            return cached

        context_start = perf_counter()
        context = await self._context_service.build(request)
        context_ms = self._elapsed_ms(context_start)
        logger.warning("Route timing: context_ms=%.3f", context_ms)
        fuel_price_start = perf_counter()
        fuel_prices = await self._fuel_cost_service.get_price_snapshot()
        fuel_price_ms = self._elapsed_ms(fuel_price_start)
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
        if self._segment_alternative_service is not None:
            segment_alternatives_start = perf_counter()
            try:
                segment_matrix = await self._segment_alternative_service.build_matrix(
                    request=request,
                    points=context.points,
                    context=context,
                    weights=dynamic_weights.adjusted,
                    fuel_prices=fuel_prices,
                )
                context = replace(
                    context,
                    segment_alternatives=segment_matrix.segment_alternatives,
                    best_segment_score_matrix=segment_matrix.best_segment_score_matrix,
                    best_segment_distance_matrix_km=segment_matrix.best_segment_distance_matrix_km,
                    best_segment_duration_matrix_min=segment_matrix.best_segment_duration_matrix_min,
                    best_segment_choice_matrix=segment_matrix.best_segment_choice_matrix,
                    segment_alternatives_enabled=segment_matrix.enabled,
                    segment_alternatives_summary=segment_matrix.summary,
                )
                logger.warning(
                    "Segment alternatives debug: enabled=%s pairs=%d candidates=%d used=%d avg_candidates=%.2f gain=%.4f",
                    segment_matrix.summary.enabled,
                    segment_matrix.summary.total_pairs,
                    segment_matrix.summary.total_candidates,
                    segment_matrix.summary.used_candidates,
                    segment_matrix.summary.average_candidates_per_pair,
                    segment_matrix.summary.estimated_gain_pct,
                )
            except Exception:
                logger.exception("Segment alternatives build failed; falling back to base matrices")
            finally:
                segment_alternatives_ms = self._elapsed_ms(segment_alternatives_start)
                logger.warning("Route timing: segment_alternatives_ms=%.3f", segment_alternatives_ms)

        optimization_start = perf_counter()
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
        optimization_ms = self._elapsed_ms(optimization_start)
        logger.warning("Route timing: optimization_ms=%.3f", optimization_ms)
        self._attach_segment_alternative_diagnostics(optimization.diagnostics, context)
        self._enforce_non_regressing_selection(
            request=request,
            optimization=optimization,
            baseline_evaluation=baseline_evaluation,
            reason="selected_route_regressed_key_metrics",
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

        geometry_start = perf_counter()
        provider_result, baseline_provider_result, alternatives = await self._hydrate_route_geometries(
            best_order_points=best_order_points,
            baseline_order_points=baseline_order_points,
            alternatives=alternatives,
            profile=request.profile,
        )
        geometry_ms = self._elapsed_ms(geometry_start)
        logger.warning("Route timing: geometry_ms=%.3f", geometry_ms)
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

        refinement = None
        if self._route_refinement_service is not None:
            refinement_start = perf_counter()
            refinement_result = await self._route_refinement_service.refine(
                request=request,
                ordered_points=best_order_points,
                order_indices=optimization.best.order_indices,
                baseline_order_indices=baseline_evaluation.order_indices,
                selected_route_result=provider_result,
                selected_metrics=optimization.best.metrics,
                context=context,
                weights=dynamic_weights.adjusted,
            )
            provider_result = refinement_result.route_result
            optimization.best.metrics = refinement_result.metrics
            refinement = refinement_result.info
            refinement_ms = self._elapsed_ms(refinement_start)
            if optimization.diagnostics is not None:
                optimization.diagnostics.route_refinement_applied = refinement.applied
                optimization.diagnostics.route_refinement_reason = refinement.reason
            if self._enforce_non_regressing_selection(
                request=request,
                optimization=optimization,
                baseline_evaluation=baseline_evaluation,
                reason="refinement_regressed_key_metrics",
            ):
                best_order_points = baseline_order_points
                provider_result = baseline_provider_result
                refinement = refinement.model_copy(
                    update={
                        "applied": False,
                        "reason": "refinement_regressed_key_metrics",
                        "improvement_pct": 0.0,
                        "changed_segments": 0,
                        "segment_choices": [],
                    }
                )
                if optimization.diagnostics is not None:
                    optimization.diagnostics.route_refinement_applied = False
                    optimization.diagnostics.route_refinement_reason = refinement.reason
            logger.warning(
                "Route refinement debug: applied=%s reason=%s improvement_pct=%.4f changed_segments=%d provider=%s geometry_points=%d distance_km=%.3f duration_min=%s",
                refinement.applied,
                refinement.reason,
                refinement.improvement_pct,
                refinement.changed_segments,
                provider_result.provider,
                len(provider_result.geometry),
                provider_result.distance_km,
                None if provider_result.duration_min is None else round(provider_result.duration_min, 3),
            )

        analysis_start = perf_counter()
        terrain_profile, baseline_terrain_profile, alternatives = await self._hydrate_terrain_profiles(
            selected_geometry=provider_result.geometry,
            baseline_geometry=baseline_provider_result.geometry,
            alternatives=alternatives,
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
        comparison_summary = self._build_comparison_summary(
            comparison=comparison,
            selected_ordered_points=best_order_points,
            selected_geometry=provider_result.geometry,
            selected_terrain_profile=terrain_profile,
            baseline_provider=baseline_provider_result.provider,
            selected_provider=provider_result.provider,
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
                road_quality_risk=optimization.best.metrics.road_quality_risk,
                dynamic_event_risk=optimization.best.metrics.dynamic_event_risk,
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
        cvrp_start = perf_counter()
        cvrp_plan = await self._build_cvrp_plan(
            request=request,
            context=context,
            best_order_indices=optimization.best.order_indices,
            weights=dynamic_weights.adjusted,
            fuel_prices=fuel_prices,
        )
        cvrp_ms = self._elapsed_ms(cvrp_start)
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
        decision_explanation = self._decision_explanation_service.build(
            request=request,
            comparison=comparison,
            alternatives=alternatives,
            diagnostics=optimization.diagnostics,
        )
        route_quality_index = self._build_route_quality_index(
            metrics=optimization.best.metrics,
            stress_resilience=None if stress_test is None else stress_test.resilience_index,
        )
        data_confidence = self._build_data_confidence(data_sources)
        constraint_health = self._build_constraint_health(request, optimization.best.metrics)
        analysis_ms = self._elapsed_ms(analysis_start)
        total_ms = self._elapsed_ms(total_start)
        if optimization.diagnostics is not None:
            optimization.diagnostics.performance_timings = PerformanceTimings(
                context_ms=context_ms,
                optimization_ms=optimization_ms,
                refinement_ms=refinement_ms,
                analysis_ms=analysis_ms,
                total_ms=total_ms,
            )
        logger.warning(
            "Route timing summary: context_ms=%.3f fuel_price_ms=%.3f segment_alternatives_ms=%.3f optimization_ms=%.3f geometry_ms=%.3f refinement_ms=%.3f analysis_ms=%.3f cvrp_ms=%.3f total_ms=%.3f",
            context_ms,
            fuel_price_ms,
            segment_alternatives_ms,
            optimization_ms,
            geometry_ms,
            refinement_ms,
            analysis_ms,
            cvrp_ms,
            total_ms,
        )

        response = RouteResponse(
            ordered_points=best_order_points,
            total_distance_km=optimization.best.metrics.distance_km,
            total_duration_min=optimization.best.metrics.duration_min,
            geometry=provider_result.geometry,
            geojson=self._as_geojson(provider_result.geometry, provider_result.provider),
            provider=provider_result.provider,
            fuel_cost=fuel_cost,
            operational_cost=self._build_operational_cost_breakdown(
                metrics=optimization.best.metrics,
                fuel_cost_total=fuel_cost.total_cost,
                currency=fuel_cost.currency,
            ),
            metrics=optimization.best.metrics,
            alternatives=alternatives,
            segment_factors=optimization.best.segment_factors,
            segment_insights=segment_insights,
            terrain_profile=terrain_profile,
            stress_test=stress_test,
            diagnostics=optimization.diagnostics,
            decision_explanation=decision_explanation,
            route_quality_index=route_quality_index,
            data_confidence=data_confidence,
            constraint_health=constraint_health,
            dynamic_weights=dynamic_weights,
            refinement=refinement,
            segment_alternatives_summary=context.segment_alternatives_summary,
            comparison=comparison,
            comparison_summary=comparison_summary,
            analysis_matrices=self._build_analysis_matrices(context),
            data_sources=data_sources,
            population_memory_orders=optimization.population_memory_orders,
            cvrp_plan=cvrp_plan,
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

    def export_run_pdf(self, run_id: str) -> bytes | None:
        return self._run_repository.export_pdf(run_id)

    def _make_cache_key(self, request: RouteRequest) -> str:
        payload = request.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((perf_counter() - start) * 1000.0, 3)

    @staticmethod
    def _attach_segment_alternative_diagnostics(diagnostics, context) -> None:
        if diagnostics is None:
            return
        summary = context.segment_alternatives_summary
        diagnostics.segment_alternatives_enabled = summary.enabled
        diagnostics.segment_alternatives_total_candidates = summary.total_candidates
        diagnostics.segment_alternatives_used = summary.used_candidates
        diagnostics.segment_alternative_gain_pct = summary.estimated_gain_pct

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

    def _attach_warm_start_orders(self, request: RouteRequest) -> RouteRequest:
        if not request.adapt_from_run_id:
            return request
        details = self._run_repository.get_run(request.adapt_from_run_id)
        if details is None:
            logger.warning("Warm start skipped: run_id=%s not found", request.adapt_from_run_id)
            return request
        seed_orders = self._seed_orders_from_response(details.response, request.points)
        if not seed_orders:
            logger.warning("Warm start skipped: run_id=%s has no compatible route orders", request.adapt_from_run_id)
            return request
        logger.warning("Warm start attached: run_id=%s compatible_orders=%d", request.adapt_from_run_id, len(seed_orders))
        return request.model_copy(update={"warm_start_orders": seed_orders})

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
    def _seed_orders_from_response(response: dict, points: list) -> list[list[int]]:
        raw_orders: list[list] = []
        best_order = response.get("ordered_points") if isinstance(response, dict) else None
        if isinstance(best_order, list):
            raw_orders.append(best_order)
        for population_order in response.get("population_memory_orders") or []:
            if isinstance(population_order, list):
                raw_orders.append(population_order)
        for alt in response.get("alternatives") or []:
            if isinstance(alt, dict) and isinstance(alt.get("ordered_points"), list):
                raw_orders.append(alt["ordered_points"])

        seed_orders: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for raw_order in raw_orders:
            if all(isinstance(item, int) for item in raw_order):
                order = RouteService._validate_index_order(raw_order, len(points))
            else:
                order = RouteService._resolve_order_indices(raw_order, points)
            if order is None:
                continue
            key = tuple(order)
            if key in seen:
                continue
            seen.add(key)
            seed_orders.append(order)
        return seed_orders

    @staticmethod
    def _validate_index_order(raw_order: list[int], points_count: int) -> list[int] | None:
        if len(raw_order) != points_count:
            return None
        if sorted(raw_order) != list(range(points_count)):
            return None
        return [int(idx) for idx in raw_order]

    @staticmethod
    def _resolve_order_indices(raw_order: list[dict], points: list) -> list[int] | None:
        if len(raw_order) != len(points):
            return None
        coord_map = {
            RouteService._coord_key(point.lat, point.lon): idx
            for idx, point in enumerate(points)
        }
        label_map = {point.label: idx for idx, point in enumerate(points) if point.label}
        order: list[int] = []
        used: set[int] = set()
        for raw_point in raw_order:
            if not isinstance(raw_point, dict):
                return None
            idx = None
            if "lat" in raw_point and "lon" in raw_point:
                idx = coord_map.get(RouteService._coord_key(raw_point["lat"], raw_point["lon"]))
            if idx is None and raw_point.get("label"):
                idx = label_map.get(raw_point["label"])
            if idx is None or idx in used:
                return None
            used.add(idx)
            order.append(idx)
        return order if len(used) == len(points) else None

    @staticmethod
    def _coord_key(lat: object, lon: object) -> str:
        return f"{float(lat):.6f},{float(lon):.6f}"

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
            f"{metrics.operational_cost:.3f}|{metrics.cargo_risk:.5f}|"
            f"{metrics.co2_kg:.3f}|{metrics.road_quality_risk:.5f}|"
            f"{metrics.dynamic_event_risk:.5f}|"
            f"{metrics.objective_score:.5f}|{int(metrics.feasible)}"
        )
        return f"{order}::{metrics_key}"

    @staticmethod
    def _build_route_quality_index(
        *,
        metrics: RouteMetrics,
        stress_resilience: float | None,
    ) -> RouteQualityIndex:
        speed_kph = metrics.distance_km / max(metrics.duration_min / 60.0, 1e-6)
        cost_per_km = metrics.operational_cost / max(metrics.distance_km, 1e-6)
        co2_per_km = metrics.co2_kg / max(metrics.distance_km, 1e-6)
        components = [
            RouteQualityComponent(
                key="safety",
                label="Safety",
                score=RouteService._clamp_score((1.0 - metrics.safety_risk) * 100.0),
                weight=0.22,
            ),
            RouteQualityComponent(
                key="reliability",
                label="Reliability",
                score=RouteService._clamp_score(metrics.reliability_score * 100.0),
                weight=0.20,
            ),
            RouteQualityComponent(
                key="time",
                label="Time efficiency",
                score=RouteService._clamp_score((speed_kph / 90.0) * 100.0),
                weight=0.16,
            ),
            RouteQualityComponent(
                key="cost",
                label="Cost efficiency",
                score=RouteService._clamp_score(100.0 - min(100.0, cost_per_km / 2.0)),
                weight=0.16,
            ),
            RouteQualityComponent(
                key="constraints",
                label="Constraints",
                score=RouteService._clamp_score(100.0 if metrics.feasible else 100.0 - metrics.constraint_penalty / 1000.0),
                weight=0.14,
            ),
            RouteQualityComponent(
                key="emissions",
                label="Ecology",
                score=RouteService._clamp_score(100.0 - min(100.0, co2_per_km * 12.0)),
                weight=0.08,
            ),
            RouteQualityComponent(
                key="resilience",
                label="Stress resilience",
                score=RouteService._clamp_score((stress_resilience if stress_resilience is not None else 0.75) * 100.0),
                weight=0.04,
            ),
        ]
        total = sum(item.score * item.weight for item in components) / max(sum(item.weight for item in components), 1e-9)
        score = round(RouteService._clamp_score(total), 1)
        return RouteQualityIndex(score=score, label=RouteService._quality_label(score), components=components)

    @staticmethod
    def _build_data_confidence(data_sources) -> DataConfidenceScore:
        source_map = data_sources.model_dump(mode="json") if data_sources is not None else {}
        source_scores: dict[str, float] = {}
        fallback_sources: list[str] = []
        notes: list[str] = []
        for key, value in source_map.items():
            text = str(value or "unknown").lower()
            if "haversine" in text:
                score = 30.0
            elif "fallback" in text or "heuristic" in text:
                score = 55.0
            elif "unknown" in text or "mock" in text:
                score = 45.0
            else:
                score = 95.0
            source_scores[key] = score
            if score < 70.0:
                fallback_sources.append(key)
        if fallback_sources:
            notes.append("Some inputs were estimated or produced by fallback providers.")
        else:
            notes.append("All primary data groups came from configured providers.")
        score = round(sum(source_scores.values()) / max(len(source_scores), 1), 1)
        return DataConfidenceScore(
            score=score,
            label=RouteService._confidence_label(score),
            source_scores=source_scores,
            fallback_sources=fallback_sources,
            notes=notes,
        )

    @staticmethod
    def _build_constraint_health(request: RouteRequest, metrics: RouteMetrics) -> ConstraintHealthReport:
        specs = [
            ("max_distance_km", "Distance limit", metrics.distance_km),
            ("max_duration_min", "Duration limit", metrics.duration_min),
            ("max_fuel_cost", "Fuel cost limit", metrics.fuel_cost),
            ("max_operational_cost", "Operational cost limit", metrics.operational_cost),
            ("max_co2_kg", "CO2 limit", metrics.co2_kg),
            ("max_safety_risk", "Safety risk limit", metrics.safety_risk),
            ("max_cargo_risk", "Cargo risk limit", metrics.cargo_risk),
        ]
        items: list[ConstraintHealthItem] = []
        for key, label, value in specs:
            limit = getattr(request.constraints, key)
            if limit is None:
                continue
            margin_pct = ((float(limit) - float(value)) / max(float(limit), 1e-9)) * 100.0
            violated = float(value) > float(limit)
            if violated:
                status = "violated"
            elif margin_pct <= 10.0:
                status = "near_violation"
            else:
                status = "satisfied"
            items.append(
                ConstraintHealthItem(
                    key=key,
                    label=label,
                    status=status,
                    value=float(value),
                    limit=float(limit),
                    margin_pct=round(margin_pct, 2),
                    violated=violated,
                )
            )
        for violated_key in metrics.violated_constraints:
            items.append(
                ConstraintHealthItem(
                    key=violated_key,
                    label=violated_key,
                    status="violated",
                    violated=True,
                )
            )
        if not items:
            return ConstraintHealthReport(overall_status="not_configured", items=[])
        if any(item.status == "violated" for item in items):
            overall = "violated"
        elif any(item.status == "near_violation" for item in items):
            overall = "near_violation"
        else:
            overall = "satisfied"
        return ConstraintHealthReport(overall_status=overall, items=items)

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
            operational_cost=optimized_metrics.operational_cost - baseline_metrics.operational_cost,
            cargo_risk=optimized_metrics.cargo_risk - baseline_metrics.cargo_risk,
            co2_kg=optimized_metrics.co2_kg - baseline_metrics.co2_kg,
            objective_score=optimized_metrics.objective_score - baseline_metrics.objective_score,
        )
        improvement_pct = RouteComparisonDelta(
            distance_km=self._improvement_pct(baseline_metrics.distance_km, optimized_metrics.distance_km),
            duration_min=self._improvement_pct(baseline_metrics.duration_min, optimized_metrics.duration_min),
            fuel_cost=self._improvement_pct(baseline_metrics.fuel_cost, optimized_metrics.fuel_cost),
            operational_cost=self._improvement_pct(
                baseline_metrics.operational_cost,
                optimized_metrics.operational_cost,
            ),
            cargo_risk=self._improvement_pct(baseline_metrics.cargo_risk, optimized_metrics.cargo_risk),
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

    @staticmethod
    def _build_comparison_summary(
        *,
        comparison: RouteComparisonInfo,
        selected_ordered_points: list,
        selected_geometry: list[list[float]],
        selected_terrain_profile,
        baseline_provider: str,
        selected_provider: str,
    ) -> RouteComparisonSummary:
        improved_metrics = [
            key
            for key in (
                "distance_km",
                "duration_min",
                "fuel_cost",
                "operational_cost",
                "cargo_risk",
                "co2_kg",
                "objective_score",
            )
            if getattr(comparison.improvement_pct, key) > 0.0
        ]
        return RouteComparisonSummary(
            baseline=RouteComparisonRouteView(
                label="baseline",
                ordered_points=comparison.baseline_ordered_points,
                geometry=comparison.baseline_geometry,
                metrics=comparison.baseline_metrics,
                terrain_profile=comparison.baseline_terrain_profile,
                score=comparison.baseline_score,
                provider=baseline_provider,
            ),
            selected=RouteComparisonRouteView(
                label="selected",
                ordered_points=selected_ordered_points,
                geometry=selected_geometry,
                metrics=comparison.optimized_metrics,
                terrain_profile=selected_terrain_profile,
                score=comparison.optimized_score,
                provider=selected_provider,
            ),
            delta=comparison.delta,
            improvement_pct=comparison.improvement_pct,
            improved_metrics=improved_metrics,
        )

    @staticmethod
    def _build_analysis_matrices(context) -> RouteAnalysisMatrices:
        return RouteAnalysisMatrices(
            point_labels=[
                point.label or f"Точка {index + 1}"
                for index, point in enumerate(context.points)
            ],
            distance_km=context.distance_matrix_km,
            duration_min=context.duration_matrix_min,
            traffic_index=context.traffic_matrix,
            toll_cost=context.toll_matrix,
            road_quality=context.surface_quality_matrix,
            incident_risk=context.incident_risk_matrix,
            roadwork_risk=context.roadwork_risk_matrix,
            elevation_gain_m=context.elevation_gain_matrix_m,
            elevation_loss_m=context.elevation_loss_matrix_m,
            mean_elevation_m=context.mean_elevation_matrix_m,
            height_clearance_m=context.height_clearance_matrix_m,
            weight_limit_t=context.weight_limit_matrix_t,
            width_limit_m=context.width_limit_matrix_m,
            length_limit_m=context.length_limit_matrix_m,
            infrastructure_access=context.infrastructure_access_matrix,
            temporal_access=context.temporal_access_matrix,
        )

    @staticmethod
    def _enforce_non_regressing_selection(
        *,
        request: RouteRequest,
        optimization,
        baseline_evaluation,
        reason: str,
    ) -> bool:
        regressions = RouteOptimizer._key_metric_regressions(
            baseline_evaluation.metrics,
            optimization.best.metrics,
        )
        if not regressions:
            return False
        if RouteOptimizer._is_candidate_acceptable_for_strategy(
            request,
            baseline_evaluation.metrics,
            optimization.best.metrics,
        ):
            if optimization.diagnostics is not None:
                optimization.diagnostics.accepted_tradeoff = True
                optimization.diagnostics.rejected_regression_metrics = regressions
            logger.warning(
                "Baseline guard accepted controlled tradeoff: strategy=%s regressions=%s",
                request.optimization_strategy.value,
                regressions,
            )
            return False
        optimization.best = baseline_evaluation
        if optimization.diagnostics is not None:
            optimization.diagnostics.baseline_guard_applied = True
            optimization.diagnostics.final_selected_from = "baseline"
            optimization.diagnostics.baseline_guard_reason = reason
            optimization.diagnostics.final_selection_reason = reason
            optimization.diagnostics.baseline_score = baseline_evaluation.metrics.objective_score
            optimization.diagnostics.rejected_regression_metrics = regressions
            optimization.diagnostics.rejected_alternative_reasons.append(
                f"{reason}: {', '.join(regressions)}",
            )
        logger.warning(
            "Baseline guard selected entered order: reason=%s regressions=%s",
            reason,
            regressions,
        )
        return True

    async def _hydrate_route_geometries(
        self,
        *,
        best_order_points: list,
        baseline_order_points: list,
        alternatives: list[RouteAlternative],
        profile,
    ) -> tuple[RouteProviderResult, RouteProviderResult, list[RouteAlternative]]:
        route_inputs: dict[str, list] = {}
        selected_key = self._points_key(best_order_points)
        baseline_key = self._points_key(baseline_order_points)
        route_inputs[selected_key] = best_order_points
        route_inputs.setdefault(baseline_key, baseline_order_points)
        for alt in alternatives:
            route_inputs.setdefault(self._points_key(alt.ordered_points), alt.ordered_points)

        keys = list(route_inputs)
        route_results = await asyncio.gather(
            *(self._routing_repository.route(route_inputs[key], profile) for key in keys)
        )
        result_by_key = dict(zip(keys, route_results, strict=False))

        hydrated_alternatives = [
            alt.model_copy(
                update={
                    "geometry": result_by_key[self._points_key(alt.ordered_points)].geometry,
                    "provider": result_by_key[self._points_key(alt.ordered_points)].provider,
                }
            )
            for alt in alternatives
        ]
        return result_by_key[selected_key], result_by_key[baseline_key], hydrated_alternatives

    async def _hydrate_terrain_profiles(
        self,
        *,
        selected_geometry: list[list[float]],
        baseline_geometry: list[list[float]],
        alternatives: list[RouteAlternative],
    ):
        terrain_inputs: dict[str, list[list[float]]] = {}
        selected_key = self._geometry_key(selected_geometry)
        baseline_key = self._geometry_key(baseline_geometry)
        terrain_inputs[selected_key] = selected_geometry
        terrain_inputs.setdefault(baseline_key, baseline_geometry)
        for alt in alternatives:
            if len(alt.geometry) <= 250:
                terrain_inputs.setdefault(self._geometry_key(alt.geometry), alt.geometry)

        keys = list(terrain_inputs)
        terrain_profiles = await asyncio.gather(
            *(self._terrain_profile_service.build_route_profile(terrain_inputs[key]) for key in keys)
        )
        profile_by_key = dict(zip(keys, terrain_profiles, strict=False))
        hydrated_alternatives = [
            alt.model_copy(update={"terrain_profile": profile_by_key[self._geometry_key(alt.geometry)]})
            if len(alt.geometry) <= 250 and self._geometry_key(alt.geometry) in profile_by_key
            else alt
            for alt in alternatives
        ]
        return profile_by_key[selected_key], profile_by_key[baseline_key], hydrated_alternatives

    @staticmethod
    def _points_key(points: list) -> str:
        return "|".join(f"{point.lat:.6f},{point.lon:.6f}" for point in points)

    @staticmethod
    def _geometry_key(geometry: list[list[float]]) -> str:
        return "|".join(f"{float(lat):.6f},{float(lon):.6f}" for lat, lon in geometry)

    async def _build_cvrp_plan(
        self,
        *,
        request: RouteRequest,
        context,
        best_order_indices: list[int],
        weights: CriteriaWeights,
        fuel_prices,
    ) -> CvrpPlanInfo | None:
        demands = self._normalized_demands(request, len(context.points))
        total_demand = sum(demands)
        enabled = total_demand > 0 or request.cvrp.vehicle_count > 1 or bool(request.cvrp.point_demands_t)
        if not enabled:
            return None

        capacity_t = self._fuel_cost_service.vehicle_capacity_t(request)
        depot_index = request.cvrp.depot_index
        route_orders = self._split_vehicle_route_orders(
            best_order_indices=best_order_indices,
            demands=demands,
            capacity_t=capacity_t,
            depot_index=depot_index,
            return_to_depot=request.cvrp.return_to_depot,
        )

        evaluations = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._optimizer.evaluate_order,
                    request,
                    context,
                    order,
                    weights,
                    fuel_prices,
                )
                for order in route_orders
            )
        )
        route_results = await asyncio.gather(
            *(self._routing_repository.route([context.points[idx] for idx in order], request.profile) for order in route_orders)
        )
        terrain_profiles = await asyncio.gather(
            *(self._terrain_profile_service.build_route_profile(result.geometry) for result in route_results)
        )

        vehicle_routes: list[RouteVehicleRoute] = []
        for idx, (order, evaluation, route_result, terrain_profile) in enumerate(
            zip(route_orders, evaluations, route_results, terrain_profiles, strict=False),
            start=1,
        ):
            demand_t = sum(demands[point_idx] for point_idx in order if point_idx != depot_index)
            vehicle_routes.append(
                RouteVehicleRoute(
                    vehicle_index=idx,
                    order_indices=order,
                    ordered_points=[context.points[point_idx] for point_idx in order],
                    demand_t=demand_t,
                    capacity_t=capacity_t,
                    load_ratio=demand_t / max(capacity_t, 0.01),
                    feasible=demand_t <= capacity_t and idx <= request.cvrp.vehicle_count,
                    metrics=evaluation.metrics,
                    geometry=route_result.geometry,
                    provider=route_result.provider,
                    terrain_profile=terrain_profile,
                )
            )

        total_distance = sum(route.metrics.distance_km for route in vehicle_routes if route.metrics is not None)
        total_duration = sum(route.metrics.duration_min for route in vehicle_routes if route.metrics is not None)
        makespan = max((route.metrics.duration_min for route in vehicle_routes if route.metrics is not None), default=0.0)
        max_route_load = max((route.demand_t for route in vehicle_routes), default=0.0)
        max_load_ratio = max((route.load_ratio for route in vehicle_routes), default=0.0)
        capacity_penalty = sum(
            100_000.0 * max(0.0, route.demand_t - capacity_t) / max(capacity_t, 0.01)
            for route in vehicle_routes
        )
        if len(vehicle_routes) > request.cvrp.vehicle_count:
            capacity_penalty += 100_000.0 * (len(vehicle_routes) - request.cvrp.vehicle_count)
        feasible = all(route.feasible for route in vehicle_routes) and capacity_penalty <= 1e-9

        return CvrpPlanInfo(
            enabled=True,
            depot_index=depot_index,
            vehicle_count=request.cvrp.vehicle_count,
            routes_used=len(vehicle_routes),
            capacity_t=capacity_t,
            total_demand_t=total_demand,
            max_route_load_t=max_route_load,
            max_load_ratio=max_load_ratio,
            feasible=feasible,
            capacity_penalty=capacity_penalty,
            total_distance_km=total_distance,
            total_duration_min=total_duration,
            makespan_min=makespan,
            routes=vehicle_routes,
        )

    @staticmethod
    def _split_vehicle_route_orders(
        *,
        best_order_indices: list[int],
        demands: list[float],
        capacity_t: float,
        depot_index: int,
        return_to_depot: bool,
    ) -> list[list[int]]:
        customers = [idx for idx in best_order_indices if idx != depot_index]
        routes: list[list[int]] = []
        current: list[int] = []
        current_load = 0.0
        for idx in customers:
            demand = demands[idx] if idx < len(demands) else 0.0
            if current and current_load + demand > capacity_t:
                routes.append(current)
                current = []
                current_load = 0.0
            current.append(idx)
            current_load += demand
        if current or not routes:
            routes.append(current)

        route_orders: list[list[int]] = []
        for route in routes:
            order = [depot_index] + route
            if return_to_depot:
                order.append(depot_index)
            route_orders.append(order)
        return route_orders

    @staticmethod
    def _normalized_demands(request: RouteRequest, points_count: int) -> list[float]:
        demands = list(request.cvrp.point_demands_t)
        if len(demands) < points_count:
            demands.extend([0.0 for _ in range(points_count - len(demands))])
        return [max(0.0, float(value)) for value in demands[:points_count]]

    def _build_pair_score_explanations(
        self,
        baseline_metrics: RouteMetrics,
        optimized_metrics: RouteMetrics,
        weights: CriteriaWeights,
    ) -> tuple[ScoreExplanation, ScoreExplanation]:
        normalized_weights = CriteriaService._objective_weights(weights)

        component_specs: list[tuple[str, str, Callable[[RouteMetrics], float], float]] = [
            ("distance", "Distance", lambda m: m.distance_km, normalized_weights.distance),
            ("duration", "Duration", lambda m: m.duration_min, normalized_weights.duration),
            (
                "operational_cost",
                "Operational cost",
                lambda m: m.operational_cost,
                normalized_weights.operational_cost,
            ),
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
    def _build_operational_cost_breakdown(
        *,
        metrics: RouteMetrics,
        fuel_cost_total: float,
        currency: str,
    ) -> OperationalCostBreakdown:
        return OperationalCostBreakdown(
            fuel_and_tolls=metrics.fuel_cost,
            fuel_only=fuel_cost_total,
            toll_cost=metrics.toll_cost,
            driver_cost=metrics.driver_cost,
            maintenance_cost=metrics.maintenance_cost,
            cargo_expected_loss=metrics.cargo_expected_loss,
            total_cost=metrics.operational_cost,
            cargo_risk=metrics.cargo_risk,
            currency=currency,
        )

    @staticmethod
    def _normalize(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        if high - low <= 1e-12:
            return 0.0
        return (value - low) / (high - low)

    @staticmethod
    def _relative_delta_pct(baseline_value: float, delta: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0 if delta <= 0 else 100.0
        return max(0.0, (delta / abs(baseline_value)) * 100.0)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @staticmethod
    def _quality_label(score: float) -> str:
        if score >= 85.0:
            return "Excellent route"
        if score >= 70.0:
            return "Good route"
        if score >= 50.0:
            return "Acceptable route"
        return "High-risk route"

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 85.0:
            return "High confidence"
        if score >= 60.0:
            return "Medium confidence"
        return "Low confidence"

    @staticmethod
    def _improvement_pct(baseline_value: float, optimized_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0
        return max(0.0, ((baseline_value - optimized_value) / baseline_value) * 100.0)
