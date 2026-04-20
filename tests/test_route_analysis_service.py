from __future__ import annotations

from app.domain.models import Point, RouteMetrics, RouteRequest, RouteSegmentFactor
from app.services.route_analysis_service import RouteAnalysisService


def test_build_segment_insights_detects_dominant_factor() -> None:
    service = RouteAnalysisService()
    points = [
        Point(lat=51.67, lon=39.18, label="Старт"),
        Point(lat=51.68, lon=39.20, label="Склад"),
        Point(lat=51.69, lon=39.22, label="Финиш"),
    ]
    segments = [
        RouteSegmentFactor(
            start_index=0,
            end_index=1,
            distance_km=8.0,
            duration_min=16.0,
            avg_speed_kph=30.0,
            elevation_gain_m=25.0,
            elevation_loss_m=0.0,
            congestion_index=0.82,
            weather_severity=0.25,
            reliability_risk=0.42,
            safety_risk=0.28,
            toll_cost=0.0,
        ),
        RouteSegmentFactor(
            start_index=1,
            end_index=2,
            distance_km=6.0,
            duration_min=20.0,
            avg_speed_kph=30.0,
            elevation_gain_m=180.0,
            elevation_loss_m=10.0,
            congestion_index=0.10,
            weather_severity=0.10,
            reliability_risk=0.20,
            safety_risk=0.18,
            toll_cost=40.0,
        ),
    ]

    insights = service.build_segment_insights(points=points, segment_factors=segments)

    assert len(insights) == 2
    assert insights[0].dominant_factor_key == "congestion"
    assert insights[0].start_label == "Старт"
    assert insights[0].end_label == "Склад"
    assert insights[1].dominant_factor_key == "elevation"
    assert insights[1].severity_level in {"medium", "high"}


def test_build_stress_test_returns_probabilities_and_highlights() -> None:
    service = RouteAnalysisService()
    request = RouteRequest(
        points=[
            Point(lat=51.67, lon=39.18, label="Старт"),
            Point(lat=51.69, lon=39.22, label="Финиш"),
        ],
        random_seed=17,
    )
    metrics = RouteMetrics(
        distance_km=24.0,
        duration_min=52.0,
        fuel_liters=2.8,
        fuel_cost=210.0,
        co2_kg=6.4,
        congestion_index=0.45,
        weather_risk=0.35,
        reliability_score=0.76,
        safety_risk=0.21,
        toll_cost=30.0,
        objective_score=1.42,
        constraint_penalty=0.0,
        feasible=True,
    )
    segments = [
        RouteSegmentFactor(
            start_index=0,
            end_index=1,
            distance_km=12.0,
            duration_min=24.0,
            avg_speed_kph=30.0,
            elevation_gain_m=20.0,
            elevation_loss_m=0.0,
            congestion_index=0.65,
            weather_severity=0.40,
            reliability_risk=0.38,
            safety_risk=0.24,
            toll_cost=10.0,
        ),
        RouteSegmentFactor(
            start_index=1,
            end_index=2,
            distance_km=12.0,
            duration_min=28.0,
            avg_speed_kph=25.7,
            elevation_gain_m=85.0,
            elevation_loss_m=15.0,
            congestion_index=0.25,
            weather_severity=0.40,
            reliability_risk=0.32,
            safety_risk=0.26,
            toll_cost=20.0,
        ),
    ]

    stress = service.build_stress_test(request=request, metrics=metrics, segment_factors=segments)

    assert stress.simulations == 120
    assert 0.0 <= stress.on_time_probability <= 1.0
    assert 0.0 <= stress.failure_probability <= 1.0
    assert 0.0 <= stress.resilience_index <= 1.0
    assert stress.duration_p90_min >= stress.duration_p10_min
    assert stress.fuel_cost_p90 >= stress.fuel_cost_p10
    assert stress.expected_duration_min >= metrics.duration_min
    assert stress.highlights
    assert all(item.factor_label for item in stress.highlights)
