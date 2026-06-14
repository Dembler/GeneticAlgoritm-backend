from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from app.domain.models import (
    CriteriaWeights,
    DataSourceInfo,
    DynamicWeightsInfo,
    FuelCostBreakdown,
    OptimizationDiagnostics,
    Point,
    RouteMetrics,
    RouteRequest,
    RouteSegmentFactor,
    ScoreMode,
)
from app.repositories.elevation_repository import ElevationProfile
from app.repositories.routing_repository import RouteProviderResult
from app.repositories.weather_repository import WeatherSnapshot
from app.services.context_service import OptimizationContext
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.route_analysis_service import RouteAnalysisService
from app.services.route_optimizer import OptimizationResult, ParetoItem
from app.services.route_refinement_service import RouteRefinementService
from app.services.route_service import RouteService


class DummyCacheRepository:
    def __init__(self) -> None:
        self.value = None

    def get(self, _key: str):
        return self.value

    def set(self, _key: str, value) -> None:
        self.value = value


class DummyContextService:
    def __init__(self, context: OptimizationContext) -> None:
        self.context = context

    async def build(self, request: RouteRequest) -> OptimizationContext:
        return self.context


class DummyRoutingRepository:
    def __init__(self) -> None:
        self.route_calls: list[tuple[tuple[float, float], ...]] = []

    async def route(self, points: list[Point], profile) -> RouteProviderResult:
        self.route_calls.append(tuple((point.lat, point.lon) for point in points))
        return RouteProviderResult(
            geometry=[[point.lat, point.lon] for point in points],
            distance_km=18.0,
            duration_min=40.0,
            provider="fallback",
        )


class DummyFuelCostService:
    async def get_price_snapshot(self):
        return object()

    def price_per_liter(self, _fuel_prices, _fuel_type) -> float:
        return 63.0

    def vehicle_capacity_t(self, request: RouteRequest) -> float:
        return float(request.vehicle_capacity_t or 1.0)

    def resolve_consumption_l_per_100km(self, request: RouteRequest) -> float:
        return float(request.fuel_consumption_l_per_100km or 8.5)

    async def compute(
        self,
        request,
        distance_km,
        uphill_pct,
        downhill_pct,
        temperature_c,
        congestion_index,
        mean_elevation_m,
        road_quality_risk=0.0,
        dynamic_event_risk=0.0,
    ):
        return FuelCostBreakdown(
            fuel_type=request.fuel_type,
            vehicle_class=request.vehicle_class,
            consumption_l_per_100km=float(request.fuel_consumption_l_per_100km or 8.5),
            distance_km=distance_km,
            uphill_share_pct=uphill_pct,
            downhill_share_pct=downhill_pct,
            terrain_multiplier=1.02,
            mountain_multiplier=1.01,
            temperature_multiplier=1.03,
            congestion_multiplier=1.04,
            surface_multiplier=1.0 + (0.12 * road_quality_risk),
            dynamic_events_multiplier=1.0 + (0.08 * dynamic_event_risk),
            liters_total=2.9,
            price_per_liter=63.0,
            total_cost=182.7,
            currency="RUB",
            price_source="test",
            price_source_url=None,
            price_date=None,
            price_retrieved_at=datetime.now(timezone.utc).isoformat(),
        )


class DummyDynamicWeightsService:
    def compute(self, request: RouteRequest, context: OptimizationContext, fuel_price_per_liter: float) -> DynamicWeightsInfo:
        return DynamicWeightsInfo(
            base=request.criteria_weights,
            adjusted=request.criteria_weights,
            triggers=["test"],
        )


class DummyRunRepository:
    def save(self, request: RouteRequest, response) -> str:
        return "run-test"

    def list_runs(self, limit: int = 20):
        return []

    def get_run(self, run_id: str):
        return None

    def export_csv(self, run_id: str):
        return None

    def export_pdf(self, run_id: str):
        return None


class DummyTerrainProfileService:
    def __init__(self) -> None:
        self.profile_calls: list[tuple[tuple[float, float], ...]] = []

    async def build_route_profile(self, geometry: list[list[float]]):
        from app.domain.models import RouteTerrainProfile, RouteTerrainSegment, TerrainTrend

        self.profile_calls.append(tuple((float(lat), float(lon)) for lat, lon in geometry))
        if len(geometry) < 2:
            return RouteTerrainProfile(source="test")
        return RouteTerrainProfile(
            sampled_points=len(geometry),
            total_gain_m=12.0,
            total_loss_m=4.0,
            max_uphill_grade_pct=3.2,
            max_downhill_grade_pct=1.8,
            source="test",
            segments=[
                RouteTerrainSegment(
                    trend=TerrainTrend.uphill,
                    geometry=geometry,
                    distance_km=18.0,
                    elevation_delta_m=8.0,
                    elevation_gain_m=12.0,
                    elevation_loss_m=4.0,
                    grade_pct=1.1,
                )
            ],
        )


class RefinementRoutingRepository:
    def __init__(self) -> None:
        self.route_calls: list[tuple[tuple[float, float], ...]] = []

    async def route(self, points: list[Point], profile) -> RouteProviderResult:
        self.route_calls.append(tuple((point.lat, point.lon) for point in points))
        if len(points) == 3 and points[0].lat == 51.67 and points[-1].lat == 51.68:
            return RouteProviderResult(
                geometry=[[points[0].lat, points[0].lon], [points[1].lat, points[1].lon], [points[-1].lat, points[-1].lon]],
                distance_km=5.0,
                duration_min=10.0,
                provider="test-refined",
            )
        if len(points) == 2:
            return RouteProviderResult(
                geometry=[[point.lat, point.lon] for point in points],
                distance_km=7.0 if points[0].lat == 51.67 and points[1].lat == 51.68 else 11.0,
                duration_min=15.0 if points[0].lat == 51.67 and points[1].lat == 51.68 else 26.0,
                provider="test-baseline",
            )
        return RouteProviderResult(
            geometry=[[point.lat, point.lon] for point in points],
            distance_km=20.0,
            duration_min=50.0,
            provider="test-detour",
        )


class RiskAwareRefinementRoutingRepository:
    def __init__(self, *, excessive_detour: bool = False) -> None:
        self.excessive_detour = excessive_detour

    async def route(self, points: list[Point], profile) -> RouteProviderResult:
        geometry = [[point.lat, point.lon] for point in points]
        if len(points) == 2:
            return RouteProviderResult(
                geometry=geometry,
                distance_km=10.0,
                duration_min=12.0,
                provider="risk-baseline",
            )
        if len(points) == 3 and points[1].lat > ((points[0].lat + points[-1].lat) / 2.0):
            return RouteProviderResult(
                geometry=geometry,
                distance_km=30.0 if self.excessive_detour else 10.8,
                duration_min=36.0 if self.excessive_detour else 13.0,
                provider="risk-detour",
            )
        return RouteProviderResult(
            geometry=geometry,
            distance_km=16.0,
            duration_min=24.0,
            provider="risk-other",
        )


class DummyOptimizer:
    def __init__(self, evaluation) -> None:
        self.evaluation = evaluation

    def optimize(self, request: RouteRequest, context: OptimizationContext, weights: CriteriaWeights, fuel_prices) -> OptimizationResult:
        return OptimizationResult(
            best=self.evaluation,
            pareto=[ParetoItem(evaluation=self.evaluation, rank=0, crowding=1.0)],
            diagnostics=OptimizationDiagnostics(
                mode=request.optimize_mode,
                optimization_active=True,
                optimization_reason=None,
                score_mode=ScoreMode.population_normalized,
                generations=24,
                population_size=24,
                crossover_rate=request.crossover_rate,
                mutation_rate=request.mutation_rate,
                stagnation_generations=3,
                evaluated_solutions=24,
                pareto_solutions=1,
                population_memory_solutions=1,
            ),
            matrix_provider="test-matrix",
            population_memory_orders=[self.evaluation.order_indices],
        )

    def evaluate_order(self, request: RouteRequest, context: OptimizationContext, order_indices: list[int], weights: CriteriaWeights, fuel_prices):
        return self.evaluation


class BaselineAwareDummyOptimizer(DummyOptimizer):
    def __init__(self, best_evaluation, baseline_evaluation) -> None:
        super().__init__(best_evaluation)
        self.baseline_evaluation = baseline_evaluation

    def evaluate_order(self, request: RouteRequest, context: OptimizationContext, order_indices: list[int], weights: CriteriaWeights, fuel_prices):
        if order_indices == list(range(len(context.points))):
            return self.baseline_evaluation
        return self.evaluation


def _context() -> OptimizationContext:
    points = [
        Point(lat=51.67, lon=39.18, label="Старт"),
        Point(lat=51.68, lon=39.20, label="Промежуточная"),
        Point(lat=51.69, lon=39.22, label="Финиш"),
    ]
    return OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 7.0, 12.0], [7.0, 0.0, 11.0], [12.0, 11.0, 0.0]],
        duration_matrix_min=[[0.0, 14.0, 26.0], [14.0, 0.0, 22.0], [26.0, 22.0, 0.0]],
        traffic_matrix=[[0.0, 0.6, 0.3], [0.6, 0.0, 0.2], [0.3, 0.2, 0.0]],
        toll_matrix=[[0.0, 0.0, 20.0], [0.0, 0.0, 10.0], [20.0, 10.0, 0.0]],
        weather=WeatherSnapshot(
            severity=0.4,
            temperature_c=8.0,
            precipitation_mm=1.0,
            wind_speed_kph=11.0,
            source="test",
            source_url=None,
            observed_at=datetime.now(timezone.utc),
        ),
        elevation=ElevationProfile(
            elevations_m=[140.0, 190.0, 230.0],
            source="test",
            source_url=None,
        ),
        surface_quality_matrix=[[1.0, 0.9, 0.85], [0.9, 1.0, 0.8], [0.85, 0.8, 1.0]],
        incident_risk_matrix=[[0.0, 0.1, 0.12], [0.1, 0.0, 0.08], [0.12, 0.08, 0.0]],
        roadwork_risk_matrix=[[0.0, 0.08, 0.1], [0.08, 0.0, 0.06], [0.1, 0.06, 0.0]],
        infrastructure_access_matrix=[[True, True, True], [True, True, True], [True, True, True]],
        temporal_access_matrix=[[True, True, True], [True, True, True], [True, True, True]],
        departure_at=datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc),
        data_sources=DataSourceInfo(
            routing="test",
            matrix="test",
            weather="test",
            elevation="test",
            traffic="test",
            tolls="test",
            fuel_prices="test",
        ),
        matrix_provider="test",
    )


def _evaluation():
    from app.services.criteria_service import CandidateEvaluation

    return CandidateEvaluation(
        order_indices=[0, 1, 2],
        metrics=RouteMetrics(
            distance_km=18.0,
            duration_min=41.0,
            fuel_liters=2.9,
            fuel_cost=182.7,
            co2_kg=6.8,
            congestion_index=0.4,
            weather_risk=0.4,
            reliability_score=0.74,
            safety_risk=0.21,
            toll_cost=10.0,
            objective_score=1.12,
            constraint_penalty=0.0,
            feasible=True,
        ),
        segment_factors=[
            RouteSegmentFactor(
                start_index=0,
                end_index=1,
                distance_km=7.0,
                duration_min=15.0,
                avg_speed_kph=28.0,
                elevation_gain_m=50.0,
                elevation_loss_m=0.0,
                congestion_index=0.6,
                weather_severity=0.4,
                reliability_risk=0.35,
                safety_risk=0.22,
                toll_cost=0.0,
            ),
            RouteSegmentFactor(
                start_index=1,
                end_index=2,
                distance_km=11.0,
                duration_min=26.0,
                avg_speed_kph=25.4,
                elevation_gain_m=40.0,
                elevation_loss_m=0.0,
                congestion_index=0.2,
                weather_severity=0.4,
                reliability_risk=0.28,
                safety_risk=0.20,
                toll_cost=10.0,
            ),
        ],
        uphill_pct=3.2,
        downhill_pct=0.0,
        mean_elevation_m=186.0,
    )


@pytest.mark.asyncio
async def test_route_service_response_contains_segment_insights_and_stress_test() -> None:
    context = _context()
    evaluation = _evaluation()
    request = RouteRequest(points=context.points, random_seed=9)
    routing_repository = DummyRoutingRepository()
    terrain_profile_service = DummyTerrainProfileService()
    service = RouteService(
        optimizer=DummyOptimizer(evaluation),
        routing_repository=routing_repository,
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=terrain_profile_service,
            run_repository=DummyRunRepository(),
        )

    response = await service.compute_route(request)

    assert response.run_id == "run-test"
    assert len(response.segment_insights) == 2
    assert response.segment_insights[0].dominant_factor_label
    assert response.stress_test is not None
    assert response.stress_test.simulations == 120
    assert 0.0 <= response.stress_test.resilience_index <= 1.0
    assert response.comparison is not None
    assert response.comparison.baseline_geometry == [[point.lat, point.lon] for point in context.points]
    assert response.comparison.baseline_terrain_profile is not None
    assert response.comparison_summary is not None
    assert response.comparison_summary.baseline.label == "baseline"
    assert response.comparison_summary.selected.label == "selected"
    assert response.comparison_summary.baseline.geometry == [[point.lat, point.lon] for point in context.points]
    assert response.comparison_summary.selected.metrics == response.metrics
    assert response.analysis_matrices is not None
    assert response.analysis_matrices.point_labels == [point.label for point in context.points]
    assert response.analysis_matrices.distance_km == context.distance_matrix_km
    assert response.analysis_matrices.duration_min == context.duration_matrix_min
    assert response.analysis_matrices.traffic_index == context.traffic_matrix
    assert response.alternatives
    assert response.alternatives[0].geometry == [[point.lat, point.lon] for point in context.points]
    assert response.terrain_profile is not None
    assert response.alternatives[0].terrain_profile is not None
    assert response.population_memory_orders == [[0, 1, 2]]
    assert len(routing_repository.route_calls) == 1
    assert len(terrain_profile_service.profile_calls) == 1


@pytest.mark.asyncio
async def test_route_service_falls_back_to_baseline_when_selected_route_regresses_key_metrics() -> None:
    context = _context()
    baseline = _evaluation()
    selected = _evaluation()
    selected.order_indices = [0, 2, 1]
    selected.metrics = baseline.metrics.model_copy(
        update={
            "distance_km": baseline.metrics.distance_km + 4.0,
            "fuel_liters": baseline.metrics.fuel_liters + 0.8,
            "fuel_cost": baseline.metrics.fuel_cost + 50.0,
            "operational_cost": baseline.metrics.operational_cost + 50.0,
            "co2_kg": baseline.metrics.co2_kg + 2.0,
            "objective_score": 0.01,
        }
    )
    service = RouteService(
        optimizer=BaselineAwareDummyOptimizer(selected, baseline),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )

    response = await service.compute_route(RouteRequest(points=context.points, random_seed=9))

    assert response.ordered_points == context.points
    assert response.metrics.distance_km == baseline.metrics.distance_km
    assert response.diagnostics is not None
    assert response.diagnostics.baseline_guard_applied is True
    assert response.diagnostics.final_selected_from == "baseline"
    assert response.diagnostics.final_selection_reason == "selected_route_regressed_key_metrics"
    assert response.comparison is not None
    assert response.comparison.improvement_pct.distance_km == 0.0
    assert response.comparison.improvement_pct.fuel_cost == 0.0
    assert response.comparison.improvement_pct.operational_cost == 0.0
    assert response.comparison.improvement_pct.co2_kg == 0.0


@pytest.mark.asyncio
async def test_route_service_response_contains_decision_quality_and_confidence() -> None:
    context = _context()
    evaluation = _evaluation()
    service = RouteService(
        optimizer=DummyOptimizer(evaluation),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )

    response = await service.compute_route(RouteRequest(points=context.points, random_seed=9))

    assert response.decision_explanation is not None
    assert response.decision_explanation.main_reason
    assert response.route_quality_index is not None
    assert 0.0 <= response.route_quality_index.score <= 100.0
    assert response.data_confidence is not None
    assert 0.0 <= response.data_confidence.score <= 100.0
    assert response.constraint_health is not None
    assert response.diagnostics is not None
    assert response.diagnostics.performance_timings is not None
    assert response.diagnostics.performance_timings.total_ms >= 0.0


@pytest.mark.asyncio
async def test_route_service_balanced_strategy_accepts_controlled_safety_tradeoff() -> None:
    context = _context()
    baseline = _evaluation()
    selected_metrics = baseline.metrics.model_copy(
        update={
            "distance_km": baseline.metrics.distance_km * 1.05,
            "duration_min": baseline.metrics.duration_min * 1.04,
            "fuel_liters": baseline.metrics.fuel_liters * 1.04,
            "fuel_cost": baseline.metrics.fuel_cost * 1.04,
            "operational_cost": baseline.metrics.operational_cost * 1.04,
            "co2_kg": baseline.metrics.co2_kg * 1.04,
            "weather_risk": baseline.metrics.weather_risk * 0.45,
            "reliability_score": min(1.0, baseline.metrics.reliability_score * 1.16),
            "safety_risk": baseline.metrics.safety_risk * 0.45,
            "dynamic_event_risk": baseline.metrics.dynamic_event_risk * 0.45,
            "cargo_risk": baseline.metrics.cargo_risk * 0.45,
            "objective_score": 0.01,
        }
    )
    selected = replace(
        baseline,
        order_indices=[0, 2, 1],
        metrics=selected_metrics,
    )
    service = RouteService(
        optimizer=BaselineAwareDummyOptimizer(selected, baseline),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )

    response = await service.compute_route(
        RouteRequest(points=context.points, optimization_strategy="balanced", random_seed=9)
    )

    assert [point.label for point in response.ordered_points] == [
        context.points[0].label,
        context.points[2].label,
        context.points[1].label,
    ]
    assert response.diagnostics is not None
    assert response.diagnostics.accepted_tradeoff is True
    assert response.diagnostics.final_selected_from == "optimizer"
    assert response.decision_explanation is not None
    assert response.decision_explanation.compromise_accepted is True


@pytest.mark.asyncio
async def test_route_service_balanced_strategy_ignores_cargo_risk_tradeoff_for_selection() -> None:
    context = _context()
    baseline = _evaluation()
    selected_metrics = baseline.metrics.model_copy(
        update={
            "distance_km": baseline.metrics.distance_km * 0.55,
            "duration_min": baseline.metrics.duration_min * 0.48,
            "fuel_liters": baseline.metrics.fuel_liters * 0.60,
            "fuel_cost": baseline.metrics.fuel_cost * 0.60,
            "operational_cost": baseline.metrics.operational_cost * 0.60,
            "co2_kg": baseline.metrics.co2_kg * 0.60,
            "cargo_risk": baseline.metrics.cargo_risk + 0.05,
            "objective_score": 0.01,
        }
    )
    selected = replace(
        baseline,
        order_indices=[0, 2, 1],
        metrics=selected_metrics,
    )
    service = RouteService(
        optimizer=BaselineAwareDummyOptimizer(selected, baseline),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )

    response = await service.compute_route(RouteRequest(points=context.points, random_seed=9))

    assert response.metrics is not None
    assert response.metrics.duration_min == selected_metrics.duration_min
    assert response.diagnostics is not None
    assert response.diagnostics.final_selected_from == "optimizer"
    assert response.diagnostics.accepted_tradeoff is False
    assert response.comparison is not None
    assert response.comparison.improvement_pct.duration_min > 50.0


@pytest.mark.asyncio
async def test_route_service_builds_cvrp_vehicle_plan() -> None:
    context = _context()
    evaluation = _evaluation()
    request = RouteRequest(
        points=context.points,
        vehicle_capacity_t=1.0,
        cvrp={"point_demands_t": [0.0, 0.8, 0.8], "vehicle_count": 2},
        random_seed=9,
    )
    service = RouteService(
        optimizer=DummyOptimizer(evaluation),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )

    response = await service.compute_route(request)

    assert response.cvrp_plan is not None
    assert response.cvrp_plan.enabled is True
    assert response.cvrp_plan.routes_used == 2
    assert response.cvrp_plan.feasible is True
    assert [route.order_indices for route in response.cvrp_plan.routes] == [[0, 1, 0], [0, 2, 0]]
    assert all(route.geometry for route in response.cvrp_plan.routes)


@pytest.mark.asyncio
async def test_route_service_applies_refinement_when_same_order_segment_is_better() -> None:
    context = _context()
    evaluation = _evaluation()
    request = RouteRequest(points=context.points, random_seed=9)
    routing_repository = RefinementRoutingRepository()
    fuel_cost_service = DummyFuelCostService()
    service = RouteService(
        optimizer=DummyOptimizer(evaluation),
        routing_repository=routing_repository,
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=fuel_cost_service,
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
            route_refinement_service=RouteRefinementService(
            routing_repository=routing_repository,
            fuel_cost_service=fuel_cost_service,
        ),
    )

    response = await service.compute_route(request)

    assert response.refinement is not None
    assert response.refinement.applied is True
    assert response.refinement.reason == "same_order_refined_by_segment_geometry"
    assert response.refinement.changed_segments == 1
    assert response.refinement.segment_choices[0].selected_variant != "baseline"
    assert response.total_distance_km < context.distance_matrix_km[0][1] + context.distance_matrix_km[1][2]
    assert response.provider.endswith(":refined")
    assert len(response.geometry) > len(context.points)


@pytest.mark.asyncio
async def test_route_refinement_reports_no_improvement_when_candidates_do_not_win() -> None:
    context = _context()
    evaluation = _evaluation()
    request = RouteRequest(points=context.points, random_seed=9)
    routing_repository = DummyRoutingRepository()
    selected_route = RouteProviderResult(
        geometry=[[point.lat, point.lon] for point in context.points],
        distance_km=18.0,
        duration_min=41.0,
        provider="test",
    )
    service = RouteRefinementService(
        routing_repository=routing_repository,
        fuel_cost_service=DummyFuelCostService(),
    )

    result = await service.refine(
        request=request,
        ordered_points=context.points,
        order_indices=[0, 1, 2],
        baseline_order_indices=[0, 1, 2],
        selected_route_result=selected_route,
        selected_metrics=evaluation.metrics,
        context=context,
        weights=request.criteria_weights,
    )

    assert result.info.applied is False
    assert result.info.reason == "same_order_no_better_segment_geometry"
    assert result.route_result == selected_route
    assert result.metrics == evaluation.metrics


@pytest.mark.asyncio
async def test_route_refinement_does_not_select_detour_only_for_risk_reduction() -> None:
    context = _context()
    context.traffic_matrix[0][1] = 0.95
    context.incident_risk_matrix[0][1] = 0.95
    context.roadwork_risk_matrix[0][1] = 0.85
    context.surface_quality_matrix[0][1] = 0.2
    request = RouteRequest(
        points=context.points,
        criteria_weights=CriteriaWeights(
            distance=0.1,
            duration=0.1,
            fuel_cost=0.1,
            emissions=0.1,
            congestion=1.0,
            weather_risk=1.0,
            reliability=1.0,
            safety=2.0,
            road_quality=2.0,
            dynamic_events=2.0,
            cargo_risk=1.0,
        ),
        random_seed=9,
    )
    service = RouteRefinementService(
        routing_repository=RiskAwareRefinementRoutingRepository(),
        fuel_cost_service=DummyFuelCostService(),
    )

    selected, baseline, choice = await service._refine_segment(
        request=request,
        context=context,
        weights=request.criteria_weights,
        start_index=0,
        end_index=1,
        start_point=context.points[0],
        end_point=context.points[1],
    )

    assert choice is None
    assert selected.variant == "baseline"
    assert selected.route_result.provider == "risk-baseline"
    assert selected.score == baseline.score


@pytest.mark.asyncio
async def test_route_refinement_rejects_excessive_detour_even_when_it_reduces_risk() -> None:
    context = _context()
    context.traffic_matrix[0][1] = 0.95
    context.incident_risk_matrix[0][1] = 0.95
    context.roadwork_risk_matrix[0][1] = 0.85
    context.surface_quality_matrix[0][1] = 0.2
    request = RouteRequest(
        points=context.points,
        criteria_weights=CriteriaWeights(
            distance=1.0,
            duration=1.0,
            fuel_cost=1.0,
            emissions=1.0,
            congestion=1.0,
            weather_risk=1.0,
            reliability=1.0,
            safety=1.0,
            road_quality=1.0,
            dynamic_events=1.0,
            cargo_risk=1.0,
        ),
        random_seed=9,
    )
    service = RouteRefinementService(
        routing_repository=RiskAwareRefinementRoutingRepository(excessive_detour=True),
        fuel_cost_service=DummyFuelCostService(),
    )

    selected, baseline, choice = await service._refine_segment(
        request=request,
        context=context,
        weights=request.criteria_weights,
        start_index=0,
        end_index=1,
        start_point=context.points[0],
        end_point=context.points[1],
    )

    assert choice is None
    assert selected == baseline


def test_route_service_adapts_default_ga_budget_for_large_routes() -> None:
    context = _context()
    service = RouteService(
        optimizer=DummyOptimizer(_evaluation()),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )
    points = [Point(lat=51.0 + idx * 0.01, lon=39.0 + idx * 0.01) for idx in range(18)]
    request = RouteRequest(points=points)

    adjusted = service._apply_runtime_defaults(request)

    assert adjusted.population_size == 60
    assert adjusted.generations == 72
    assert adjusted.max_alternatives == 5


def test_route_service_preserves_custom_ga_budget_values() -> None:
    context = _context()
    service = RouteService(
        optimizer=DummyOptimizer(_evaluation()),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
            context_service=DummyContextService(context),
            dynamic_weights_service=DummyDynamicWeightsService(),
            route_analysis_service=RouteAnalysisService(),
            decision_explanation_service=DecisionExplanationService(),
            terrain_profile_service=DummyTerrainProfileService(),
            run_repository=DummyRunRepository(),
        )
    points = [Point(lat=51.0 + idx * 0.01, lon=39.0 + idx * 0.01) for idx in range(18)]
    request = RouteRequest(
        points=points,
        population_size=140,
        generations=180,
        max_alternatives=10,
    )

    adjusted = service._apply_runtime_defaults(request)

    assert adjusted.population_size == 140
    assert adjusted.generations == 180
    assert adjusted.max_alternatives == 10


def test_route_service_resolves_seed_orders_from_previous_response() -> None:
    points = [
        Point(lat=51.67, lon=39.18, label="Start"),
        Point(lat=51.68, lon=39.20, label="Depot"),
        Point(lat=51.69, lon=39.22, label="Finish"),
    ]
    response = {
        "ordered_points": [
            points[0].model_dump(mode="json"),
            points[2].model_dump(mode="json"),
            points[1].model_dump(mode="json"),
        ],
        "population_memory_orders": [[0, 2, 1], [0, 1, 2]],
        "alternatives": [
            {
                "ordered_points": [
                    points[0].model_dump(mode="json"),
                    points[1].model_dump(mode="json"),
                    points[2].model_dump(mode="json"),
                ],
            }
        ],
    }

    orders = RouteService._seed_orders_from_response(response, points)

    assert orders == [[0, 2, 1], [0, 1, 2]]
