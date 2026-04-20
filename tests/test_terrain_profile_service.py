from __future__ import annotations

import math

import pytest

from app.domain.models import Point
from app.repositories.elevation_repository import ElevationProfile
from app.services.terrain_profile_service import PolylineSample, TerrainProfileService


class SyntheticElevationRepository:
    async def fetch(self, points: list[Point]) -> ElevationProfile:
        elevations = [100.0 * math.sin(point.lon * math.pi) for point in points]
        return ElevationProfile(
            elevations_m=elevations,
            source="synthetic-elevation",
            source_url=None,
        )


@pytest.mark.asyncio
async def test_edge_matrices_capture_intermediate_climb_even_when_endpoints_match() -> None:
    service = TerrainProfileService(elevation_repository=SyntheticElevationRepository())
    points = [
        Point(lat=51.67, lon=0.0),
        Point(lat=51.67, lon=1.0),
    ]

    gain_matrix, loss_matrix, mean_matrix = await service.build_edge_matrices(points)

    assert gain_matrix[0][1] > 0.0
    assert loss_matrix[0][1] > 0.0
    assert mean_matrix[0][1] > 0.0
    assert gain_matrix[1][0] == pytest.approx(loss_matrix[0][1], rel=1e-6)
    assert loss_matrix[1][0] == pytest.approx(gain_matrix[0][1], rel=1e-6)


@pytest.mark.asyncio
async def test_route_profile_marks_uphill_and_downhill_segments() -> None:
    service = TerrainProfileService(elevation_repository=SyntheticElevationRepository())
    geometry = [
        [51.67, 0.0],
        [51.67, 0.25],
        [51.67, 0.5],
        [51.67, 0.75],
        [51.67, 1.0],
    ]

    profile = await service.build_route_profile(geometry)

    assert profile.total_gain_m > 0.0
    assert profile.total_loss_m > 0.0
    assert {segment.trend for segment in profile.segments} >= {"uphill", "downhill"}


@pytest.mark.asyncio
async def test_route_profile_segment_geometry_keeps_route_shape() -> None:
    service = TerrainProfileService(elevation_repository=SyntheticElevationRepository())
    geometry = [
        [51.67, 0.0],
        [51.68, 0.0],
        [51.68, 0.01],
        [51.69, 0.01],
        [51.69, 0.02],
    ]

    profile = await service.build_route_profile(geometry)

    assert profile.segments
    assert any(len(segment.geometry) > 2 for segment in profile.segments)


def test_edge_sampling_budget_gets_lighter_for_large_point_sets() -> None:
    service = TerrainProfileService(elevation_repository=SyntheticElevationRepository())

    small_budget = service._edge_sampling_budget(6)
    medium_budget = service._edge_sampling_budget(18)
    large_budget = service._edge_sampling_budget(30)

    assert small_budget == (10.0, 80)
    assert medium_budget[0] > small_budget[0]
    assert medium_budget[1] < small_budget[1]
    assert large_budget[0] > medium_budget[0]
    assert large_budget[1] < medium_budget[1]


def test_smooth_elevations_recovers_broad_trend_from_step_like_profile() -> None:
    service = TerrainProfileService(elevation_repository=SyntheticElevationRepository())
    samples = [
        PolylineSample(point=Point(lat=51.67, lon=39.18 + index * 0.01), edge_index=index, edge_ratio=0.0)
        for index in range(8)
    ]
    elevations = [100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 102.0, 102.0]

    smoothed = service._smooth_elevations(samples, elevations)

    assert smoothed[-1] > smoothed[0]


def test_route_profile_keeps_long_same_trend_split_for_map_detail() -> None:
    service = TerrainProfileService(elevation_repository=SyntheticElevationRepository())
    route_points = [Point(lat=0.0, lon=index * 0.01) for index in range(9)]
    samples = service._sample_geometry(route_points, spacing_km=1.0, max_points=32)
    elevations = [float(index * 2.0) for index in range(len(samples))]

    profile = service._build_route_profile_from_samples(route_points, samples, elevations, source="test")

    assert len(profile.segments) >= 2
    assert {segment.trend for segment in profile.segments} == {"uphill"}
    assert max(segment.distance_km for segment in profile.segments) <= 1.05
