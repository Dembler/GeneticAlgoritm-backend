from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models import CriteriaWeights, DataSourceInfo, Point, RouteRequest, SegmentAlternativesSummary, SegmentCandidate
from app.repositories.elevation_repository import ElevationProfile
from app.repositories.routing_repository import RouteProviderResult
from app.repositories.weather_repository import WeatherSnapshot
from app.services.context_service import OptimizationContext
from app.services.criteria_service import CriteriaService
from app.services.segment_alternative_service import SegmentAlternativeService


class DummyFuelCostService:
    def resolve_consumption_l_per_100km(self, request: RouteRequest) -> float:
        return float(request.fuel_consumption_l_per_100km or 8.5)

    def price_per_liter(self, _fuel_prices, _fuel_type) -> float:
        return 63.0

    def terrain_multiplier(self, *_args, **_kwargs) -> float:
        return 1.0

    def mountain_multiplier(self, *_args, **_kwargs) -> float:
        return 1.0

    def temperature_multiplier(self, *_args, **_kwargs) -> float:
        return 1.0

    def compute_liters(self, distance_km: float, consumption: float, terrain_multiplier: float) -> float:
        return distance_km * consumption / 100.0 * terrain_multiplier

    def load_multiplier(self, _request: RouteRequest) -> float:
        return 1.0

    def estimate_co2_kg(self, liters: float, _fuel_type) -> float:
        return liters * 2.31

    def vehicle_capacity_t(self, request: RouteRequest) -> float:
        return float(request.vehicle_capacity_t or 1.0)


class CandidateRoutingRepository:
    def __init__(self, excessive_detour: bool = False, fail_detours: bool = False) -> None:
        self.excessive_detour = excessive_detour
        self.fail_detours = fail_detours
        self.calls = 0

    async def route(self, points: list[Point], profile) -> RouteProviderResult:
        self.calls += 1
        if len(points) == 2:
            return RouteProviderResult(
                geometry=[[points[0].lat, points[0].lon], [points[1].lat, points[1].lon]],
                distance_km=10.0,
                duration_min=20.0,
                provider="baseline",
            )
        if self.fail_detours:
            raise RuntimeError("detour failed")
        distance = 16.0 if self.excessive_detour else 10.6
        return RouteProviderResult(
            geometry=[[point.lat, point.lon] for point in points],
            distance_km=distance,
            duration_min=21.0,
            provider="detour",
        )


def _context(points_count: int = 2) -> OptimizationContext:
    points = [Point(lat=51.0 + idx * 0.03, lon=39.0 + idx * 0.03) for idx in range(points_count)]
    distance = [[0.0 if i == j else 10.0 for j in range(points_count)] for i in range(points_count)]
    duration = [[0.0 if i == j else 20.0 for j in range(points_count)] for i in range(points_count)]
    high_risk = [[0.0 if i == j else 0.9 for j in range(points_count)] for i in range(points_count)]
    return OptimizationContext(
        points=points,
        distance_matrix_km=distance,
        duration_matrix_min=duration,
        traffic_matrix=high_risk,
        toll_matrix=[[0.0 for _ in range(points_count)] for _ in range(points_count)],
        weather=WeatherSnapshot(
            severity=0.8,
            temperature_c=15.0,
            precipitation_mm=0.0,
            wind_speed_kph=4.0,
            source="test",
            source_url=None,
            observed_at=datetime.now(timezone.utc),
        ),
        elevation=ElevationProfile(elevations_m=[200.0 for _ in range(points_count)], source="test", source_url=None),
        departure_at=datetime(2026, 2, 18, 12, 0, tzinfo=timezone.utc),
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
        surface_quality_matrix=[[1.0 if i == j else 0.2 for j in range(points_count)] for i in range(points_count)],
        incident_risk_matrix=high_risk,
        roadwork_risk_matrix=high_risk,
        temporal_access_matrix=[[True for _ in range(points_count)] for _ in range(points_count)],
        infrastructure_access_matrix=[[True for _ in range(points_count)] for _ in range(points_count)],
    )


@pytest.mark.asyncio
async def test_segment_alternative_service_does_not_select_detour_from_risk_weight_only() -> None:
    context = _context()
    request = RouteRequest(points=context.points, criteria_weights=CriteriaWeights(distance=0.2, safety=5.0))
    service = SegmentAlternativeService(
        routing_repository=CandidateRoutingRepository(),
        fuel_cost_service=DummyFuelCostService(),
        max_candidates_per_edge=5,
        max_detour_ratio=1.15,
    )

    matrix = await service.build_matrix(
        request=request,
        points=context.points,
        context=context,
        weights=request.criteria_weights,
        fuel_prices=object(),
    )

    alt_set = matrix.segment_alternatives[(0, 1)]
    assert matrix.enabled is True
    assert alt_set.baseline_candidate.variant_type == "fastest"
    assert len(alt_set.candidates) == 5
    assert alt_set.best_candidate.variant_type == "fastest"
    assert alt_set.best_candidate.detour_ratio <= 1.15
    assert matrix.summary.used_candidates == 0


@pytest.mark.asyncio
async def test_segment_alternative_service_rejects_excessive_detour() -> None:
    context = _context()
    request = RouteRequest(points=context.points, criteria_weights=CriteriaWeights(distance=0.2, safety=5.0))
    service = SegmentAlternativeService(
        routing_repository=CandidateRoutingRepository(excessive_detour=True),
        fuel_cost_service=DummyFuelCostService(),
        max_candidates_per_edge=5,
        max_detour_ratio=1.15,
    )

    matrix = await service.build_matrix(
        request=request,
        points=context.points,
        context=context,
        weights=request.criteria_weights,
        fuel_prices=object(),
    )

    assert matrix.segment_alternatives[(0, 1)].best_candidate.variant_type == "fastest"


@pytest.mark.asyncio
async def test_segment_alternative_service_falls_back_when_provider_errors() -> None:
    context = _context()
    request = RouteRequest(points=context.points)
    service = SegmentAlternativeService(
        routing_repository=CandidateRoutingRepository(fail_detours=True),
        fuel_cost_service=DummyFuelCostService(),
    )

    matrix = await service.build_matrix(
        request=request,
        points=context.points,
        context=context,
        weights=request.criteria_weights,
        fuel_prices=object(),
    )

    assert matrix.segment_alternatives[(0, 1)].baseline_candidate.variant_type == "fastest"
    assert any(candidate.data_source == "fallback-candidate" for candidate in matrix.segment_alternatives[(0, 1)].candidates)


def test_criteria_service_uses_best_segment_candidate_when_enabled() -> None:
    context = _context()
    request = RouteRequest(points=context.points)
    candidate = SegmentCandidate(
        variant_id="0:1:safe_detour",
        variant_type="safe_detour",
        from_index=0,
        to_index=1,
        distance_km=8.0,
        duration_min=14.0,
        fuel_liters=0.7,
        fuel_cost=44.1,
        risk_exposure=0.1,
        road_quality_risk=0.1,
        weather_risk=0.1,
        dynamic_event_risk=0.1,
        safety_risk=0.1,
        cargo_risk=0.1,
        detour_ratio=1.05,
        restriction_penalty=0.0,
        objective_score=0.2,
        geometry=None,
        data_source="test",
    )
    context.segment_alternatives_enabled = True
    context.segment_alternatives_summary = SegmentAlternativesSummary(enabled=True, total_pairs=1, total_candidates=2, used_candidates=1)
    context.best_segment_choice_matrix = [[None, candidate], [None, None]]
    context.segment_alternatives = {
        (0, 1): type(
            "AltSet",
            (),
            {
                "baseline_candidate": candidate.model_copy(update={"variant_id": "0:1:fastest", "variant_type": "fastest", "objective_score": 0.4}),
            },
        )()
    }
    service = CriteriaService(DummyFuelCostService())

    evaluation = service.evaluate([0, 1], request, context, object())

    assert evaluation.metrics.distance_km == 8.0
    assert evaluation.metrics.refined_segments_count == 1
    assert evaluation.segment_factors[0].segment_variant_type == "safe_detour"
    assert evaluation.segment_factors[0].detour_ratio == 1.05
