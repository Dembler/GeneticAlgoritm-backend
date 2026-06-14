from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from app.domain.models import DataSourceInfo, Point, RouteRequest
from app.repositories.elevation_repository import ElevationProfile
from app.repositories.weather_repository import WeatherSnapshot
from app.services.context_service import OptimizationContext
from app.services.criteria_service import CriteriaService
from app.services.fuel_cost import FuelCostService, FuelPriceService, FuelPriceSnapshot
from app.services.route_optimizer import CandidateState, RouteOptimizer


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


def test_pmx_crossover_preserves_permutation() -> None:
    rng = random.Random(11)
    parent_a = (1, 2, 3, 4, 5, 6)
    parent_b = (4, 1, 6, 2, 5, 3)

    child_a, child_b = RouteOptimizer._pmx_crossover(parent_a, parent_b, rng)

    assert sorted(child_a) == sorted(parent_a)
    assert sorted(child_b) == sorted(parent_a)
    assert len(set(child_a)) == len(parent_a)
    assert len(set(child_b)) == len(parent_a)


def test_insertion_mutation_preserves_permutation_and_moves_gene() -> None:
    class FakeRandom:
        def sample(self, _population, _count):
            return [0, 2]

    genome = (1, 2, 3, 4)

    mutated = RouteOptimizer._insertion_mutation(genome, FakeRandom())

    assert mutated == (2, 3, 1, 4)
    assert sorted(mutated) == sorted(genome)


def test_repair_genome_avoids_forbidden_edges_when_possible() -> None:
    genome = (1, 2)

    repaired = RouteOptimizer._repair_genome(
        genome,
        fix_ends=True,
        points_count=4,
        forbidden_edges={(1, 2)},
    )

    assert sorted(repaired) == sorted(genome)
    assert RouteOptimizer._bad_edge_count(
        RouteOptimizer._decode(repaired, fix_ends=True, points_count=4),
        {(1, 2)},
    ) == 0


def test_initial_population_uses_warm_start_orders() -> None:
    optimizer = _optimizer()
    context = _context(points_count=5)

    population = optimizer._initial_population(
        middle_indices=[1, 2, 3],
        population_size=24,
        rng=random.Random(3),
        context=context,
        fix_ends=True,
        points_count=5,
        warm_start_orders=[[0, 3, 1, 2, 4]],
    )

    assert (3, 1, 2) in population


def test_initial_population_repairs_warm_start_forbidden_edges() -> None:
    optimizer = _optimizer()
    context = _context(points_count=4)

    population = optimizer._initial_population(
        middle_indices=[1, 2],
        population_size=24,
        rng=random.Random(3),
        context=context,
        fix_ends=True,
        points_count=4,
        warm_start_orders=[[0, 1, 2, 3]],
        forbidden_edges={(1, 2)},
    )

    assert (2, 1) in population
    assert (1, 2) not in population


def test_optimizer_reports_warm_start_seed_count() -> None:
    optimizer = _optimizer()
    context = _context(points_count=5)
    request = RouteRequest(
        points=context.points,
        optimize=True,
        fix_ends=True,
        random_seed=7,
        population_size=24,
        generations=20,
        warm_start_orders=[[0, 3, 1, 2, 4]],
    )

    result = optimizer.optimize(
        request=request,
        context=context,
        weights=request.criteria_weights,
        fuel_prices=_prices(),
    )

    assert result.diagnostics.warm_start_solutions == 1
    assert result.diagnostics.population_memory_solutions > 0
    assert result.population_memory_orders


def test_baseline_guard_selects_baseline_when_original_best_is_worse() -> None:
    optimizer = _optimizer()
    context = _context(points_count=4)
    request = RouteRequest(points=context.points, optimize=True, fix_ends=True)
    bad_order = [0, 2, 1, 3]
    bad_evaluation = optimizer._criteria_service.evaluate(bad_order, request, context, _prices())
    bad_state = CandidateState(genome=(2, 1), evaluation=bad_evaluation)

    selection = optimizer._select_final_candidate(
        request=request,
        context=context,
        weights=request.criteria_weights,
        fuel_prices=_prices(),
        candidate_states=[bad_state],
        original_best_state=bad_state,
        middle_indices=[1, 2],
        fix_ends=True,
        points_count=4,
    )

    assert selection.state.evaluation.order_indices == [0, 1, 2, 3]
    assert selection.baseline_guard_applied is True
    assert selection.final_selected_from == "baseline"
    assert selection.final_selection_reason == "optimizer_best_regressed_key_metrics"
    assert selection.state.evaluation.metrics.objective_score <= selection.baseline_score


def test_preserve_elites_keeps_baseline_seed_in_population() -> None:
    optimizer = _optimizer()
    context = _context(points_count=4)
    request = RouteRequest(points=context.points, optimize=True, fix_ends=True)
    prices = _prices()

    def evaluate_genome(genome: tuple[int, ...]):
        order = RouteOptimizer._decode(genome, fix_ends=True, points_count=4)
        return optimizer._criteria_service.evaluate(order, request, context, prices)

    selected = [CandidateState(genome=(2, 1), evaluation=evaluate_genome((2, 1)))]
    preserved = optimizer._preserve_elites(
        selected=selected,
        elite_genomes={(1, 2)},
        evaluate_genome=evaluate_genome,
        weights=request.criteria_weights,
        population_size=2,
        use_pareto=False,
    )

    assert (1, 2) in {state.genome for state in preserved}
