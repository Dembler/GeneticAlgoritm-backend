from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models import DataSourceInfo, Point, RouteRequest
from app.repositories.elevation_repository import ElevationProfile
from app.repositories.weather_repository import WeatherSnapshot
from app.services.context_service import OptimizationContext
from app.services.criteria_service import CriteriaService
from app.services.fuel_cost import FuelCostService, FuelPriceService, FuelPriceSnapshot
from app.services.route_optimizer import RouteOptimizer


def _optimizer() -> RouteOptimizer:
    fuel_cost_service = FuelCostService(
        FuelPriceService(
            repository=None,
            fallback_petrol=63.0,
            fallback_diesel=68.0,
            currency="RUB",
            source_name="test",
            source_url=None,
        )
    )
    return RouteOptimizer(criteria_service=CriteriaService(fuel_cost_service))


def _context(points_count: int) -> OptimizationContext:
    points = [Point(lat=51.60 + i * 0.01, lon=39.10 + i * 0.01) for i in range(points_count)]
    distance = [[0.0 for _ in range(points_count)] for _ in range(points_count)]
    duration = [[0.0 for _ in range(points_count)] for _ in range(points_count)]
    for i in range(points_count):
        for j in range(points_count):
            if i == j:
                continue
            d = float(abs(i - j) + 1)
            distance[i][j] = d
            duration[i][j] = d * 2.0

    return OptimizationContext(
        points=points,
        distance_matrix_km=distance,
        duration_matrix_min=duration,
        traffic_matrix=[[0.0 for _ in range(points_count)] for _ in range(points_count)],
        toll_matrix=[[0.0 for _ in range(points_count)] for _ in range(points_count)],
        weather=WeatherSnapshot(
            severity=0.1,
            temperature_c=15.0,
            precipitation_mm=0.0,
            wind_speed_kph=4.0,
            source="test",
            source_url=None,
            observed_at=datetime.now(timezone.utc),
        ),
        elevation=ElevationProfile(
            elevations_m=[200.0 + i for i in range(points_count)],
            source="test",
            source_url=None,
        ),
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
    )


def _prices() -> FuelPriceSnapshot:
    return FuelPriceSnapshot(
        petrol_rub_per_liter=63.0,
        diesel_rub_per_liter=68.0,
        currency="RUB",
        source="test",
        source_url=None,
        price_date=None,
        retrieved_at=datetime.now(timezone.utc),
    )


def test_not_enough_points_reason_exposed() -> None:
    optimizer = _optimizer()
    context = _context(points_count=2)
    request = RouteRequest(points=context.points, optimize=True, fix_ends=True)

    result = optimizer.optimize(request=request, context=context, weights=request.criteria_weights, fuel_prices=_prices())

    assert result.diagnostics.optimization_active is False
    assert result.diagnostics.optimization_reason == "not_enough_points"
    assert result.diagnostics.score_mode == "absolute_single_candidate"


def test_fixed_route_reason_exposed() -> None:
    optimizer = _optimizer()
    context = _context(points_count=3)
    request = RouteRequest(points=context.points, optimize=True, fix_ends=True)

    result = optimizer.optimize(request=request, context=context, weights=request.criteria_weights, fuel_prices=_prices())

    assert result.diagnostics.optimization_active is False
    assert result.diagnostics.optimization_reason == "fixed_route"
    assert result.diagnostics.score_mode == "absolute_single_candidate"


def test_weighted_and_pareto_use_different_generation_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = _optimizer()
    context = _context(points_count=5)
    prices = _prices()
    calls = {"pareto": 0, "weighted": 0}

    original_pareto = RouteOptimizer._select_next_generation
    original_weighted = RouteOptimizer._select_next_generation_weighted

    def wrapped_pareto(fronts, population_size):
        calls["pareto"] += 1
        return original_pareto(fronts, population_size)

    def wrapped_weighted(states, population_size):
        calls["weighted"] += 1
        return original_weighted(states, population_size)

    monkeypatch.setattr(RouteOptimizer, "_select_next_generation", staticmethod(wrapped_pareto))
    monkeypatch.setattr(RouteOptimizer, "_select_next_generation_weighted", staticmethod(wrapped_weighted))

    weighted_request = RouteRequest(
        points=context.points,
        optimize=True,
        fix_ends=True,
        optimize_mode="weighted",
        random_seed=7,
        population_size=24,
        generations=20,
    )
    optimizer.optimize(
        request=weighted_request,
        context=context,
        weights=weighted_request.criteria_weights,
        fuel_prices=prices,
    )
    weighted_calls = dict(calls)

    calls["pareto"] = 0
    calls["weighted"] = 0

    pareto_request = weighted_request.model_copy(update={"optimize_mode": "pareto"})
    optimizer.optimize(
        request=pareto_request,
        context=context,
        weights=pareto_request.criteria_weights,
        fuel_prices=prices,
    )
    pareto_calls = dict(calls)

    assert weighted_calls["weighted"] > 0
    assert weighted_calls["pareto"] == 0
    assert pareto_calls["pareto"] > 0
    assert pareto_calls["weighted"] == 0
