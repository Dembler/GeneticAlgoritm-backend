from __future__ import annotations

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
from app.services.route_analysis_service import RouteAnalysisService
from app.services.route_optimizer import OptimizationResult, ParetoItem
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
    async def route(self, points: list[Point], profile) -> RouteProviderResult:
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

    async def compute(self, request, distance_km, uphill_pct, downhill_pct, temperature_c, congestion_index, mean_elevation_m):
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


class DummyTerrainProfileService:
    async def build_route_profile(self, geometry: list[list[float]]):
        from app.domain.models import RouteTerrainProfile, RouteTerrainSegment, TerrainTrend

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
            ),
            matrix_provider="test-matrix",
        )

    def evaluate_order(self, request: RouteRequest, context: OptimizationContext, order_indices: list[int], weights: CriteriaWeights, fuel_prices):
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
    service = RouteService(
        optimizer=DummyOptimizer(evaluation),
        routing_repository=DummyRoutingRepository(),
        cache_repository=DummyCacheRepository(),
        fuel_cost_service=DummyFuelCostService(),
        context_service=DummyContextService(context),
        dynamic_weights_service=DummyDynamicWeightsService(),
        route_analysis_service=RouteAnalysisService(),
        terrain_profile_service=DummyTerrainProfileService(),
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
    assert response.alternatives
    assert response.alternatives[0].geometry == [[point.lat, point.lon] for point in context.points]
    assert response.terrain_profile is not None
    assert response.alternatives[0].terrain_profile is not None


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
