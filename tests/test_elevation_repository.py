from __future__ import annotations

import pytest

from app.domain.models import Point
from app.repositories.elevation_repository import FallbackElevationRepository


@pytest.mark.asyncio
async def test_fallback_elevation_repository_returns_varied_positive_heights() -> None:
    repository = FallbackElevationRepository()
    points = [
        Point(lat=51.667139, lon=39.174125),
        Point(lat=54.708314, lon=20.512335),
        Point(lat=43.10562, lon=131.87353),
    ]

    profile = await repository.fetch(points)

    assert len(profile.elevations_m) == len(points)
    assert all(value > 0 for value in profile.elevations_m)
    assert len(set(profile.elevations_m)) == len(points)
