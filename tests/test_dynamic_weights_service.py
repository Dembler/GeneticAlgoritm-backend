from __future__ import annotations

from datetime import datetime, timezone

from app.domain.models import DataSourceInfo, Point, RouteRequest
from app.repositories.elevation_repository import ElevationProfile
from app.repositories.weather_repository import WeatherSnapshot
from app.services.context_service import OptimizationContext
from app.services.dynamic_weights_service import DynamicWeightsService


def _context(
    departure_hour: int,
    weather_severity: float,
    traffic_value: float,
) -> OptimizationContext:
    points = [
        Point(lat=51.67, lon=39.18),
        Point(lat=51.68, lon=39.20),
        Point(lat=51.69, lon=39.22),
    ]
    traffic_matrix = [
        [0.0, traffic_value, traffic_value],
        [traffic_value, 0.0, traffic_value],
        [traffic_value, traffic_value, 0.0],
    ]
    return OptimizationContext(
        points=points,
        distance_matrix_km=[[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        duration_matrix_min=[[0.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 0.0]],
        traffic_matrix=traffic_matrix,
        toll_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        weather=WeatherSnapshot(
            severity=weather_severity,
            temperature_c=5.0,
            precipitation_mm=1.0,
            wind_speed_kph=8.0,
            source="test",
            source_url=None,
            observed_at=datetime(2026, 2, 18, departure_hour, 0, tzinfo=timezone.utc),
        ),
        elevation=ElevationProfile(elevations_m=[120.0, 130.0, 140.0], source="test", source_url=None),
        departure_at=datetime(2026, 2, 18, departure_hour, 0, tzinfo=timezone.utc),
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


def test_priority_profile_kept_when_dynamic_disabled() -> None:
    service = DynamicWeightsService()
    context = _context(departure_hour=12, weather_severity=0.1, traffic_value=0.0)
    points = context.points

    fastest = service.compute(
        RouteRequest(points=points, priority_profile="fastest", use_dynamic_weights=False),
        context,
        fuel_price_per_liter=63.0,
    )
    cheapest = service.compute(
        RouteRequest(points=points, priority_profile="cheapest", use_dynamic_weights=False),
        context,
        fuel_price_per_liter=63.0,
    )

    assert fastest.triggers == ["dynamic_disabled"]
    assert cheapest.triggers == ["dynamic_disabled"]
    assert fastest.adjusted.duration != cheapest.adjusted.duration
    assert fastest.adjusted.fuel_cost != cheapest.adjusted.fuel_cost


def test_dynamic_context_triggers_are_applied_when_enabled() -> None:
    service = DynamicWeightsService()
    context = _context(departure_hour=8, weather_severity=0.6, traffic_value=0.7)
    request = RouteRequest(points=context.points, priority_profile="fastest", use_dynamic_weights=True)

    enabled = service.compute(request, context, fuel_price_per_liter=75.0)
    disabled = service.compute(
        request.model_copy(update={"use_dynamic_weights": False}),
        context,
        fuel_price_per_liter=75.0,
    )

    assert "peak_hour" in enabled.triggers
    assert "bad_weather" in enabled.triggers
    assert "high_congestion" in enabled.triggers
    assert "high_fuel_price" in enabled.triggers
    assert enabled.adjusted.duration > disabled.adjusted.duration
