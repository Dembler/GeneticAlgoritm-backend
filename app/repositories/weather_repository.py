from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WeatherSnapshot:
    severity: float
    temperature_c: float | None
    precipitation_mm: float | None
    wind_speed_kph: float | None
    source: str
    source_url: str | None
    observed_at: datetime


class WeatherRepository(Protocol):
    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        raise NotImplementedError


class OpenMeteoWeatherRepository(WeatherRepository):
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        url = f"{self._base_url}/v1/forecast"
        observed_at = self._to_utc(at or datetime.now(timezone.utc))
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "UTC",
            "forecast_days": 16,
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()

        payload = response.json()
        hourly = payload.get("hourly", {})
        times = hourly.get("time") or []
        temperatures = hourly.get("temperature_2m") or []
        precipitations = hourly.get("precipitation") or []
        winds = hourly.get("wind_speed_10m") or []
        idx = self._nearest_hour_index(hourly.get("time"), observed_at)
        temperature = self._value_at(temperatures, idx)
        precipitation = self._value_at(precipitations, idx)
        wind_speed = self._value_at(winds, idx)
        severity = self._severity(precipitation, wind_speed)
        logger.warning(
            "Open-Meteo weather raw: lat=%.6f lon=%.6f points=%d sample_time=%s sample_temp=%s sample_precip=%s sample_wind=%s",
            lat,
            lon,
            len(times),
            times[:3],
            temperatures[:3],
            precipitations[:3],
            winds[:3],
        )
        logger.warning(
            "Open-Meteo weather selected: at=%s idx=%s temp=%s precip=%s wind=%s severity=%.3f",
            observed_at.isoformat(),
            idx,
            temperature,
            precipitation,
            wind_speed,
            severity,
        )

        return WeatherSnapshot(
            severity=severity,
            temperature_c=temperature,
            precipitation_mm=precipitation,
            wind_speed_kph=wind_speed,
            source="open-meteo",
            source_url=url,
            observed_at=observed_at,
        )

    @staticmethod
    def _value_at(values: list[float] | None, idx: int | None) -> float | None:
        if not values:
            return None
        if idx is None or idx < 0 or idx >= len(values):
            return float(values[0])
        return float(values[idx])

    @staticmethod
    def _nearest_hour_index(times: list[str] | None, target: datetime) -> int | None:
        if not times:
            return None
        nearest_idx: int | None = None
        nearest_delta = float("inf")
        for i, value in enumerate(times):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            delta = abs((dt - target).total_seconds())
            if delta < nearest_delta:
                nearest_delta = delta
                nearest_idx = i
        return nearest_idx

    @staticmethod
    def _to_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _severity(precipitation_mm: float | None, wind_speed_kph: float | None) -> float:
        p = 0.0 if precipitation_mm is None else min(1.0, precipitation_mm / 10.0)
        w = 0.0 if wind_speed_kph is None else min(1.0, wind_speed_kph / 60.0)
        return max(0.0, min(1.0, 0.6 * p + 0.4 * w))


class FallbackWeatherRepository(WeatherRepository):
    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        _ = lat, lon
        return WeatherSnapshot(
            severity=0.2,
            temperature_c=None,
            precipitation_mm=None,
            wind_speed_kph=None,
            source="fallback-weather",
            source_url=None,
            observed_at=at or datetime.now(timezone.utc),
        )


class CompositeWeatherRepository(WeatherRepository):
    def __init__(self, primary: WeatherRepository | None, fallback: WeatherRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(lat, lon, at)
            except Exception:
                return await self._fallback.fetch(lat, lon, at)
        return await self._fallback.fetch(lat, lon, at)
