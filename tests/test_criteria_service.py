from __future__ import annotations

from datetime import datetime, timezone

from app.domain.models import CriteriaWeights, RouteMetrics
from app.domain.models import DataSourceInfo, Point, RouteRequest
from app.repositories.elevation_repository import ElevationProfile
from app.repositories.weather_repository import WeatherProfile, WeatherSnapshot
from app.services.criteria_service import CandidateEvaluation, CriteriaService
from app.services.context_service import OptimizationContext
from app.services.fuel_cost import FuelCostService, FuelPriceService
from app.services.fuel_cost import FuelPriceSnapshot


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
        mean_elevation_m=0.0,
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


def test_evaluate_uses_accumulated_elevation_profile_not_only_endpoint_delta() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.69, lon=39.30),
    ]
    context = OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 14.0], [14.0, 0.0]],
        duration_matrix_min=[[0.0, 25.0], [25.0, 0.0]],
        traffic_matrix=[[0.0, 0.0], [0.0, 0.0]],
        toll_matrix=[[0.0, 0.0], [0.0, 0.0]],
        weather=WeatherSnapshot(
            severity=0.0,
            temperature_c=18.0,
            precipitation_mm=0.0,
            wind_speed_kph=3.0,
            source="test",
            source_url=None,
            observed_at=datetime.now(timezone.utc),
        ),
        elevation=ElevationProfile(
            elevations_m=[150.0, 150.0],
            source="test",
            source_url=None,
        ),
        departure_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
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
        elevation_gain_matrix_m=[[0.0, 240.0], [60.0, 0.0]],
        elevation_loss_matrix_m=[[0.0, 60.0], [240.0, 0.0]],
        mean_elevation_matrix_m=[[0.0, 320.0], [320.0, 0.0]],
    )
    prices = FuelPriceSnapshot(
        petrol_rub_per_liter=63.0,
        diesel_rub_per_liter=68.0,
        currency="RUB",
        source="test",
        source_url=None,
        price_date=None,
        retrieved_at=datetime.now(timezone.utc),
    )

    evaluation = service.evaluate(order_indices=[0, 1], request=RouteRequest(points=points), context=context, fuel_prices=prices)

    assert evaluation.segment_factors[0].elevation_gain_m == 240.0
    assert evaluation.segment_factors[0].elevation_loss_m == 60.0
    assert evaluation.uphill_pct > 0.0
    assert evaluation.downhill_pct > 0.0
    assert evaluation.mean_elevation_m == 320.0


def test_evaluate_uses_weather_profiles_by_edge_time() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.68, lon=39.20),
        Point(lat=51.69, lon=39.22),
    ]
    start_time = datetime(2026, 4, 20, 6, 0, tzinfo=timezone.utc)
    profiles = [
        WeatherProfile(
            times=[start_time, start_time.replace(hour=7)],
            temperatures_c=[10.0, -5.0],
            precipitations_mm=[0.0, 6.0],
            wind_speeds_kph=[4.0, 28.0],
            source="test",
            source_url=None,
        )
        for _ in points
    ]
    context = OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]],
        duration_matrix_min=[[0.0, 30.0, 0.0], [0.0, 0.0, 90.0], [0.0, 0.0, 0.0]],
        traffic_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        toll_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        weather=WeatherSnapshot(
            severity=0.0,
            temperature_c=10.0,
            precipitation_mm=0.0,
            wind_speed_kph=4.0,
            source="test",
            source_url=None,
            observed_at=start_time,
        ),
        elevation=ElevationProfile(elevations_m=[100.0, 100.0, 100.0], source="test", source_url=None),
        departure_at=start_time,
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
        weather_profiles=profiles,
        elevation_gain_matrix_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        elevation_loss_matrix_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        mean_elevation_matrix_m=[[0.0, 100.0, 0.0], [0.0, 0.0, 100.0], [0.0, 0.0, 0.0]],
    )
    prices = FuelPriceSnapshot(
        petrol_rub_per_liter=63.0,
        diesel_rub_per_liter=68.0,
        currency="RUB",
        source="test",
        source_url=None,
        price_date=None,
        retrieved_at=datetime.now(timezone.utc),
    )

    evaluation = service.evaluate(order_indices=[0, 1, 2], request=RouteRequest(points=points), context=context, fuel_prices=prices)

    assert evaluation.segment_factors[0].weather_severity < evaluation.segment_factors[1].weather_severity
    assert evaluation.mean_temperature_c is not None
    assert evaluation.mean_temperature_c < 10.0
