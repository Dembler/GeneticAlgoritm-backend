from __future__ import annotations

from app.domain.models import CriteriaWeights, RouteMetrics
from app.services.criteria_service import CandidateEvaluation, CriteriaService
from app.services.fuel_cost import FuelCostService, FuelPriceService


def _candidate(distance_km: float, duration_min: float) -> CandidateEvaluation:
    return CandidateEvaluation(
        order_indices=[0, 1],
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_min=duration_min,
            fuel_liters=1.0,
            fuel_cost=100.0,
            co2_kg=2.0,
            congestion_index=0.2,
            weather_risk=0.15,
            reliability_score=0.85,
            safety_risk=0.1,
            toll_cost=5.0,
            objective_score=0.0,
            constraint_penalty=0.0,
            feasible=True,
        ),
        segment_factors=[],
        uphill_pct=0.0,
        downhill_pct=0.0,
    )


def _service() -> CriteriaService:
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
    return CriteriaService(fuel_cost_service)


def test_single_candidate_uses_absolute_score_and_reacts_to_weights() -> None:
    service = _service()
    candidate_1 = _candidate(distance_km=10.0, duration_min=40.0)
    candidate_2 = _candidate(distance_km=10.0, duration_min=40.0)

    service.assign_weighted_scores([candidate_1], CriteriaWeights(distance=5.0, duration=0.1))
    service.assign_weighted_scores([candidate_2], CriteriaWeights(distance=0.1, duration=5.0))

    assert candidate_1.metrics.objective_score > 0.0
    assert candidate_2.metrics.objective_score > 0.0
    assert candidate_1.metrics.objective_score != candidate_2.metrics.objective_score


def test_multi_candidate_keeps_population_normalization() -> None:
    service = _service()
    best = _candidate(distance_km=10.0, duration_min=20.0)
    worse = _candidate(distance_km=20.0, duration_min=20.0)

    service.assign_weighted_scores([best, worse], CriteriaWeights())

    assert best.metrics.objective_score < worse.metrics.objective_score
