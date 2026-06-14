from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models import Point
from app.repositories.road_event_repository import JsonRoadEventRepository, SyntheticRoadEventRepository


@pytest.mark.asyncio
async def test_synthetic_road_event_repository_returns_bounded_dynamic_risks() -> None:
    points = [
        Point(lat=51.670, lon=39.180),
        Point(lat=51.682, lon=39.205),
        Point(lat=51.695, lon=39.225),
    ]
    repository = SyntheticRoadEventRepository()

    snapshot = await repository.fetch(points, datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc))

    assert snapshot.source == "synthetic-road-events"
    assert len(snapshot.incident_risk_matrix) == len(points)
    assert snapshot.temporal_access_matrix[0][1] is True
    assert snapshot.incident_risk_matrix[0][0] == 0.0
    assert 0.0 <= snapshot.incident_risk_matrix[0][1] <= 1.0
    assert 0.0 <= snapshot.roadwork_risk_matrix[0][1] <= 1.0
    assert snapshot.incident_risk_matrix[0][1] > 0.0


@pytest.mark.asyncio
async def test_json_road_event_repository_loads_active_labelled_edges(tmp_path) -> None:
    source_path = tmp_path / "road_events.json"
    source_path.write_text(
        """
        {
          "edges": [
            {
              "from_label": "Start",
              "to_label": "Depot",
              "incident_risk": 0.65,
              "roadwork_risk": 0.35,
              "temporal_accessible": false,
              "active_from": "2026-04-20T08:00:00+00:00",
              "active_to": "2026-04-20T12:00:00+00:00",
              "bidirectional": true
            },
            {
              "from": 1,
              "to": 2,
              "incident_risk": 0.9,
              "active_from": "2026-04-21T08:00:00+00:00"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    points = [
        Point(lat=51.670, lon=39.180, label="Start"),
        Point(lat=51.682, lon=39.205, label="Depot"),
        Point(lat=51.695, lon=39.225, label="Finish"),
    ]
    repository = JsonRoadEventRepository(str(source_path))

    snapshot = await repository.fetch(points, datetime(2026, 4, 20, 9, 0, tzinfo=timezone.utc))

    assert snapshot.source == "road-events-json"
    assert snapshot.source_url == str(source_path)
    assert snapshot.incident_risk_matrix[0][1] == 0.65
    assert snapshot.incident_risk_matrix[1][0] == 0.65
    assert snapshot.roadwork_risk_matrix[0][1] == 0.35
    assert snapshot.temporal_access_matrix[0][1] is False
    assert snapshot.temporal_access_matrix[1][0] is False
    assert snapshot.incident_risk_matrix[1][2] == 0.0
