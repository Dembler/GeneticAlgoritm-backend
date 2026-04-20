from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.repositories.weather_repository import FallbackWeatherRepository, WeatherProfile


def test_weather_profile_snapshot_skips_none_values() -> None:
    timestamp = datetime(2026, 4, 20, 6, 0, tzinfo=timezone.utc)
    profile = WeatherProfile(
        times=[timestamp, timestamp.replace(hour=7), timestamp.replace(hour=8)],
        temperatures_c=[None, 5.5, None],
        precipitations_mm=[None, 0.4, None],
        wind_speeds_kph=[None, 12.0, None],
        source="test",
        source_url=None,
    )

    snapshot = profile.snapshot_at(timestamp)

    assert snapshot.temperature_c == 5.5
    assert snapshot.precipitation_mm == 0.4
    assert snapshot.wind_speed_kph == 12.0
    assert snapshot.severity > 0.0


def test_weather_profile_snapshot_returns_none_if_series_is_all_missing() -> None:
    timestamp = datetime(2026, 4, 20, 6, 0, tzinfo=timezone.utc)
    profile = WeatherProfile(
        times=[timestamp, timestamp.replace(hour=7)],
        temperatures_c=[None, None],
        precipitations_mm=[None, None],
        wind_speeds_kph=[None, None],
        source="test",
        source_url=None,
    )

    snapshot = profile.snapshot_at(timestamp)

    assert snapshot.temperature_c is None
    assert snapshot.precipitation_mm is None
    assert snapshot.wind_speed_kph is None
    assert snapshot.severity == 0.0


@pytest.mark.asyncio
async def test_fallback_weather_profile_produces_usable_snapshot() -> None:
    repository = FallbackWeatherRepository()
    timestamp = datetime(2026, 4, 20, 11, 0, tzinfo=timezone.utc)

    profile = await repository.fetch_profile(lat=51.667139, lon=39.174125, at=timestamp)
    snapshot = profile.snapshot_at(timestamp)

    assert len(profile.times) == repository._PROFILE_HOURS
    assert snapshot.temperature_c is not None
    assert snapshot.precipitation_mm is not None
    assert snapshot.wind_speed_kph is not None
    assert snapshot.severity >= 0.14
