from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models import Point, TransportProfile
from app.repositories.infrastructure_repository import JsonInfrastructureRepository, SyntheticInfrastructureRepository
from app.repositories.osm_overpass_repository import OverpassRoadDataClient


@pytest.mark.asyncio
async def test_synthetic_infrastructure_repository_returns_vehicle_limits_for_driving() -> None:
    points = [
        Point(lat=51.670, lon=39.180),
        Point(lat=51.682, lon=39.205),
        Point(lat=51.695, lon=39.225),
    ]
    repository = SyntheticInfrastructureRepository()

    snapshot = await repository.fetch(
        points,
        TransportProfile.driving,
        datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert snapshot.source == "synthetic-infrastructure"
    assert len(snapshot.height_clearance_matrix_m) == len(points)
    assert snapshot.access_matrix[0][1] is True
    assert snapshot.height_clearance_matrix_m[0][0] is None
    assert snapshot.weight_limit_matrix_t[0][1] is not None
    assert snapshot.height_clearance_matrix_m[0][1] is not None
    assert 2.9 <= snapshot.height_clearance_matrix_m[0][1] <= 5.0
    assert 6.0 <= snapshot.weight_limit_matrix_t[0][1] <= 44.0


@pytest.mark.asyncio
async def test_synthetic_infrastructure_repository_leaves_non_driving_unrestricted() -> None:
    points = [
        Point(lat=51.670, lon=39.180),
        Point(lat=51.682, lon=39.205),
    ]
    repository = SyntheticInfrastructureRepository()

    snapshot = await repository.fetch(
        points,
        TransportProfile.walking,
        datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert snapshot.source == "synthetic-infrastructure"
    assert snapshot.height_clearance_matrix_m == [[None, None], [None, None]]
    assert snapshot.weight_limit_matrix_t == [[None, None], [None, None]]
    assert snapshot.access_matrix == [[True, True], [True, True]]


@pytest.mark.asyncio
async def test_json_infrastructure_repository_loads_labelled_bidirectional_edges(tmp_path) -> None:
    source_path = tmp_path / "infrastructure.json"
    source_path.write_text(
        """
        {
          "edges": [
            {
              "from_label": "Start",
              "to_label": "Depot",
              "height_clearance_m": 3.4,
              "weight_limit_t": 12.5,
              "width_limit_m": 2.6,
              "length_limit_m": 10.0,
              "bidirectional": true
            },
            {
              "from": 1,
              "to": 2,
              "accessible": false
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
    repository = JsonInfrastructureRepository(str(source_path))

    snapshot = await repository.fetch(
        points,
        TransportProfile.driving,
        datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert snapshot.source == "infrastructure-json"
    assert snapshot.source_url == str(source_path)
    assert snapshot.height_clearance_matrix_m[0][1] == 3.4
    assert snapshot.height_clearance_matrix_m[1][0] == 3.4
    assert snapshot.weight_limit_matrix_t[0][1] == 12.5
    assert snapshot.width_limit_matrix_m[0][1] == 2.6
    assert snapshot.length_limit_matrix_m[0][1] == 10.0
    assert snapshot.access_matrix[1][2] is False


def test_overpass_access_allows_point_when_any_nearby_driving_road_is_allowed() -> None:
    tags = [
        {"highway": "service", "access": "private"},
        {"highway": "residential", "surface": "asphalt"},
        {"highway": "footway", "access": "no"},
    ]

    assert OverpassRoadDataClient._access_allowed(tags) is True


def test_overpass_vehicle_limits_use_restriction_tags_not_generic_dimensions() -> None:
    tags = [
        {"highway": "residential", "width": "0.5", "height": "2.0", "length": "4.0"},
        {"highway": "primary", "maxheight": "4.2", "maxwidth": "2.7", "maxlength": "12.0"},
    ]

    assert OverpassRoadDataClient._min_limit(tags, "maxheight", "maxheight:physical") == 4.2
    assert OverpassRoadDataClient._min_limit(tags, "maxwidth") == 2.7
    assert OverpassRoadDataClient._min_limit(tags, "maxlength") == 12.0
