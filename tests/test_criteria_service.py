from __future__ import annotations

from datetime import datetime, timezone

import pytest

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


def test_evaluate_marks_infrastructure_violations_infeasible() -> None:
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
        elevation=ElevationProfile(elevations_m=[150.0, 150.0], source="test", source_url=None),
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
        height_clearance_matrix_m=[[None, 3.2], [3.2, None]],
        weight_limit_matrix_t=[[None, 12.0], [12.0, None]],
        width_limit_matrix_m=[[None, None], [None, None]],
        length_limit_matrix_m=[[None, None], [None, None]],
        infrastructure_access_matrix=[[True, True], [True, True]],
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
    request = RouteRequest(
        points=points,
        vehicle_dimensions={"height_m": 3.8, "weight_t": 14.0},
    )

    evaluation = service.evaluate(order_indices=[0, 1], request=request, context=context, fuel_prices=prices)

    assert evaluation.metrics.feasible is False
    assert evaluation.metrics.infrastructure_penalty > 0.0
    assert evaluation.metrics.constraint_penalty == evaluation.metrics.infrastructure_penalty
    assert evaluation.metrics.violated_constraints == ["height_clearance", "weight_limit"]
    assert evaluation.segment_factors[0].violated_constraints == ["height_clearance", "weight_limit"]


def test_evaluate_uses_road_quality_surface_risk() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.69, lon=39.30),
    ]
    base_context = OptimizationContext(
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
        elevation=ElevationProfile(elevations_m=[150.0, 150.0], source="test", source_url=None),
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
        surface_quality_matrix=[[1.0, 1.0], [1.0, 1.0]],
    )
    poor_surface_context = base_context.__class__(
        **{
            **base_context.__dict__,
            "surface_quality_matrix": [[1.0, 0.55], [0.55, 1.0]],
        }
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
    request = RouteRequest(points=points)

    good = service.evaluate(order_indices=[0, 1], request=request, context=base_context, fuel_prices=prices)
    poor = service.evaluate(order_indices=[0, 1], request=request, context=poor_surface_context, fuel_prices=prices)

    assert good.metrics.road_quality_risk == 0.0
    assert poor.metrics.road_quality_risk > 0.0
    assert poor.segment_factors[0].road_quality == 0.55
    assert round(poor.segment_factors[0].road_quality_risk, 2) == 0.45
    assert poor.metrics.duration_min > good.metrics.duration_min
    assert poor.metrics.fuel_liters > good.metrics.fuel_liters
    assert poor.metrics.safety_risk > good.metrics.safety_risk


def test_evaluate_uses_dynamic_road_events_and_temporal_restrictions() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.69, lon=39.30),
    ]
    base_context = OptimizationContext(
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
        elevation=ElevationProfile(elevations_m=[150.0, 150.0], source="test", source_url=None),
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
        incident_risk_matrix=[[0.0, 0.0], [0.0, 0.0]],
        roadwork_risk_matrix=[[0.0, 0.0], [0.0, 0.0]],
        temporal_access_matrix=[[True, True], [True, True]],
    )
    event_context = base_context.__class__(
        **{
            **base_context.__dict__,
            "incident_risk_matrix": [[0.0, 0.55], [0.0, 0.0]],
            "roadwork_risk_matrix": [[0.0, 0.35], [0.0, 0.0]],
            "temporal_access_matrix": [[True, False], [True, True]],
        }
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
    request = RouteRequest(points=points)

    clear = service.evaluate(order_indices=[0, 1], request=request, context=base_context, fuel_prices=prices)
    eventful = service.evaluate(order_indices=[0, 1], request=request, context=event_context, fuel_prices=prices)

    assert eventful.metrics.incident_risk == 0.55
    assert eventful.metrics.roadwork_risk == 0.35
    assert eventful.metrics.dynamic_event_risk == 0.55
    assert eventful.metrics.temporal_restriction_penalty > 0.0
    assert eventful.metrics.feasible is False
    assert eventful.metrics.duration_min > clear.metrics.duration_min
    assert eventful.metrics.fuel_liters > clear.metrics.fuel_liters
    assert eventful.metrics.reliability_score < clear.metrics.reliability_score
    assert eventful.segment_factors[0].temporal_accessible is False
    assert "temporal_access" in eventful.segment_factors[0].violated_constraints
    assert eventful.metrics.fitness_components is not None
    assert clear.metrics.fitness_components is not None
    assert eventful.metrics.fitness_components.road_event_factor > clear.metrics.fitness_components.road_event_factor
    assert eventful.metrics.fitness_components.restriction_penalty > clear.metrics.fitness_components.restriction_penalty


def test_evaluate_accounts_for_cargo_risk_and_operational_cost() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.69, lon=39.30),
    ]
    context = OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 42.0], [42.0, 0.0]],
        duration_matrix_min=[[0.0, 70.0], [70.0, 0.0]],
        traffic_matrix=[[0.0, 0.35], [0.35, 0.0]],
        toll_matrix=[[0.0, 120.0], [120.0, 0.0]],
        weather=WeatherSnapshot(
            severity=0.35,
            temperature_c=4.0,
            precipitation_mm=2.0,
            wind_speed_kph=18.0,
            source="test",
            source_url=None,
            observed_at=datetime.now(timezone.utc),
        ),
        elevation=ElevationProfile(elevations_m=[150.0, 210.0], source="test", source_url=None),
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
        surface_quality_matrix=[[1.0, 0.45], [0.45, 1.0]],
        incident_risk_matrix=[[0.0, 0.20], [0.20, 0.0]],
        roadwork_risk_matrix=[[0.0, 0.40], [0.40, 0.0]],
        temporal_access_matrix=[[True, True], [True, True]],
        elevation_gain_matrix_m=[[0.0, 320.0], [80.0, 0.0]],
        elevation_loss_matrix_m=[[0.0, 80.0], [320.0, 0.0]],
        mean_elevation_matrix_m=[[0.0, 260.0], [260.0, 0.0]],
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

    standard = service.evaluate(
        order_indices=[0, 1],
        request=RouteRequest(points=points),
        context=context,
        fuel_prices=prices,
    )
    fragile = service.evaluate(
        order_indices=[0, 1],
        request=RouteRequest(
            points=points,
            cargo={"profile": "fragile", "declared_value_rub": 250_000.0},
            operating_costs={"driver_cost_per_hour": 800.0, "maintenance_cost_per_km": 24.0},
        ),
        context=context,
        fuel_prices=prices,
    )

    assert fragile.metrics.cargo_risk > standard.metrics.cargo_risk
    assert fragile.segment_factors[0].cargo_risk > standard.segment_factors[0].cargo_risk
    assert fragile.metrics.cargo_expected_loss > 0.0
    assert fragile.metrics.driver_cost > 0.0
    assert fragile.metrics.maintenance_cost > 0.0
    assert fragile.metrics.operational_cost > fragile.metrics.fuel_cost
    assert fragile.metrics.fitness_components is not None
    assert fragile.metrics.fitness_components.cost == fragile.metrics.operational_cost


def test_evaluate_applies_cargo_load_to_fuel_liters() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.69, lon=39.30),
    ]
    context = OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 100.0], [100.0, 0.0]],
        duration_matrix_min=[[0.0, 130.0], [130.0, 0.0]],
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
        elevation=ElevationProfile(elevations_m=[150.0, 150.0], source="test", source_url=None),
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

    empty = service.evaluate(
        order_indices=[0, 1],
        request=RouteRequest(points=points, vehicle_class="heavy_truck", fuel_type="diesel"),
        context=context,
        fuel_prices=prices,
    )
    loaded = service.evaluate(
        order_indices=[0, 1],
        request=RouteRequest(
            points=points,
            vehicle_class="heavy_truck",
            fuel_type="diesel",
            vehicle_capacity_t=20.0,
            cargo={"weight_t": 10.0},
        ),
        context=context,
        fuel_prices=prices,
    )

    assert loaded.metrics.fuel_liters > empty.metrics.fuel_liters
    assert loaded.metrics.fuel_cost > empty.metrics.fuel_cost


def test_evaluate_penalizes_insufficient_cvrp_vehicle_count() -> None:
    service = _service()
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.68, lon=39.20),
        Point(lat=51.69, lon=39.30),
    ]
    context = OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 10.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 0.0]],
        duration_matrix_min=[[0.0, 20.0, 20.0], [20.0, 0.0, 20.0], [20.0, 20.0, 0.0]],
        traffic_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        toll_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        weather=WeatherSnapshot(
            severity=0.0,
            temperature_c=18.0,
            precipitation_mm=0.0,
            wind_speed_kph=3.0,
            source="test",
            source_url=None,
            observed_at=datetime.now(timezone.utc),
        ),
        elevation=ElevationProfile(elevations_m=[150.0, 150.0, 150.0], source="test", source_url=None),
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

    infeasible = service.evaluate(
        order_indices=[0, 1, 2],
        request=RouteRequest(
            points=points,
            vehicle_capacity_t=1.0,
            cvrp={"point_demands_t": [0.0, 0.8, 0.8], "vehicle_count": 1},
        ),
        context=context,
        fuel_prices=prices,
    )
    feasible = service.evaluate(
        order_indices=[0, 1, 2],
        request=RouteRequest(
            points=points,
            vehicle_capacity_t=1.0,
            cvrp={"point_demands_t": [0.0, 0.8, 0.8], "vehicle_count": 2},
        ),
        context=context,
        fuel_prices=prices,
    )

    assert infeasible.metrics.capacity_penalty > 0.0
    assert infeasible.metrics.capacity_feasible is False
    assert infeasible.metrics.vehicle_routes_used == 2
    assert "vehicle_capacity" in infeasible.metrics.violated_constraints
    assert feasible.metrics.capacity_penalty == 0.0
    assert feasible.metrics.capacity_feasible is True


@pytest.mark.asyncio
async def test_fuel_cost_breakdown_exposes_load_multiplier() -> None:
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
    request = RouteRequest(
        points=[Point(lat=51.67, lon=39.18), Point(lat=51.69, lon=39.30)],
        vehicle_class="heavy_truck",
        fuel_type="diesel",
        vehicle_capacity_t=20.0,
        cargo={"weight_t": 10.0},
    )

    breakdown = await fuel_cost_service.compute(request, distance_km=100.0)

    assert breakdown.load_ratio == 0.5
    assert breakdown.load_multiplier > 1.0
    assert breakdown.cargo_weight_t == 10.0
    assert breakdown.vehicle_capacity_t == 20.0
