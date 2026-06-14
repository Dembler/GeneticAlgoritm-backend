from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models import Point
from app.repositories.traffic_repository import JsonTrafficRepository, SyntheticTrafficRepository, TomTomTrafficRepository


@pytest.mark.asyncio
async def test_synthetic_traffic_repository_returns_peak_congestion() -> None:
    points = [
        Point(lat=51.670, lon=39.180),
        Point(lat=51.682, lon=39.205),
        Point(lat=51.695, lon=39.225),
    ]
    repository = SyntheticTrafficRepository()

    peak = await repository.fetch(points, datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc))
    off_peak = await repository.fetch(points, datetime(2026, 4, 20, 13, 0, tzinfo=timezone.utc))

    assert peak.source == "synthetic-traffic"
    assert len(peak.congestion_matrix) == len(points)
    assert peak.congestion_matrix[0][1] > 0.0
    assert peak.congestion_matrix[0][1] > off_peak.congestion_matrix[0][1]


@pytest.mark.asyncio
async def test_json_traffic_repository_loads_edge_values(tmp_path) -> None:
    source_path = tmp_path / "traffic.json"
    source_path.write_text(
        """
        {
          "edges": [
            {"from_label": "Start", "to_label": "Depot", "congestion": 0.72, "bidirectional": true}
          ]
        }
        """,
        encoding="utf-8",
    )
    points = [
        Point(lat=51.670, lon=39.180, label="Start"),
        Point(lat=51.682, lon=39.205, label="Depot"),
    ]
    repository = JsonTrafficRepository(str(source_path))

    snapshot = await repository.fetch(points, datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc))

    assert snapshot.source == "traffic-json"
    assert snapshot.source_url == str(source_path)
    assert snapshot.congestion_matrix[0][1] == 0.72
    assert snapshot.congestion_matrix[1][0] == 0.72


def test_tomtom_traffic_congestion_from_speeds() -> None:
    assert TomTomTrafficRepository._congestion_from_speeds(40, 80) == 0.5
    assert TomTomTrafficRepository._congestion_from_speeds(90, 80) == 0.0
    assert TomTomTrafficRepository._congestion_from_speeds(None, 80) == 0.0
    assert TomTomTrafficRepository._congestion_from_speeds(40, 0) == 0.0
