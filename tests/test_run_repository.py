from __future__ import annotations

from app.domain.models import Point, RouteMetrics, RouteRequest, RouteResponse
from app.repositories.run_repository import SqliteRouteRunRepository


def test_sqlite_run_repository_exports_pdf_report(tmp_path) -> None:
    repository = SqliteRouteRunRepository(str(tmp_path / "runs.db"))
    points = [
        Point(lat=55.7558, lon=37.6173, label="Moscow"),
        Point(lat=57.6261, lon=39.8845, label="Yaroslavl"),
    ]
    metrics = RouteMetrics(
        distance_km=250.0,
        duration_min=210.0,
        fuel_liters=22.0,
        fuel_cost=1500.0,
        operational_cost=5600.0,
        driver_cost=2100.0,
        maintenance_cost=2000.0,
        co2_kg=58.0,
        congestion_index=0.2,
        weather_risk=0.1,
        reliability_score=0.9,
        safety_risk=0.12,
        toll_cost=0.0,
        objective_score=0.2,
        constraint_penalty=0.0,
        feasible=True,
    )
    request = RouteRequest(points=points, random_seed=42)
    response = RouteResponse(
        ordered_points=points,
        total_distance_km=metrics.distance_km,
        total_duration_min=metrics.duration_min,
        geometry=[[point.lat, point.lon] for point in points],
        geojson={"type": "Feature", "geometry": {"type": "LineString", "coordinates": []}},
        provider="test-provider",
        metrics=metrics,
    )

    run_id = repository.save(request, response)
    report = repository.export_pdf(run_id)

    assert report is not None
    assert report.startswith(b"%PDF-1.4")
    assert b"Route optimization analytical report" in report
    assert run_id.encode("ascii") in report
