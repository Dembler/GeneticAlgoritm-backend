from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models import Point
from app.repositories.road_quality_repository import JsonRoadQualityRepository, SyntheticRoadQualityRepository


@pytest.mark.asyncio
async def test_synthetic_road_quality_repository_returns_bounded_surface_quality() -> None:
    points = [
        Point(lat=51.670, lon=39.180),
        Point(lat=51.682, lon=39.205),
        Point(lat=51.695, lon=39.225),
    ]
    repository = SyntheticRoadQualityRepository()

    snapshot = await repository.fetch(points, datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc))

    assert snapshot.source == "synthetic-road-quality"
    assert len(snapshot.surface_quality_matrix) == len(points)
    assert snapshot.surface_quality_matrix[0][0] == 1.0
    quality = snapshot.surface_quality_matrix[0][1]
    assert 0.35 <= quality <= 1.0
    assert quality < 1.0


@pytest.mark.asyncio
async def test_json_road_quality_repository_loads_edge_values(tmp_path) -> None:
    source_path = tmp_path / "road_quality.json"
    source_path.write_text(
        """
        {
          "edges": [
            {"from_label": "Start", "to_label": "Depot", "surface_quality": 0.42, "bidirectional": true}
          ]
        }
        """,
        encoding="utf-8",
    )
    points = [
        Point(lat=51.670, lon=39.180, label="Start"),
        Point(lat=51.682, lon=39.205, label="Depot"),
    ]
    repository = JsonRoadQualityRepository(str(source_path))

    snapshot = await repository.fetch(points, datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc))

    assert snapshot.source == "road-quality-json"
    assert snapshot.source_url == str(source_path)
    assert snapshot.surface_quality_matrix[0][1] == 0.42
    assert snapshot.surface_quality_matrix[1][0] == 0.42
