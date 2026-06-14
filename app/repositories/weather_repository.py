from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
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


@dataclass
class WeatherProfile:
    times: list[datetime]
    temperatures_c: list[float | None]
    precipitations_mm: list[float | None]
    wind_speeds_kph: list[float | None]
    source: str
    source_url: str | None

    def snapshot_at(self, at: datetime | None = None) -> WeatherSnapshot:
        observed_at = OpenMeteoWeatherRepository._to_utc(at or datetime.now(timezone.utc))
        idx = OpenMeteoWeatherRepository._nearest_hour_index(
            [value.isoformat() for value in self.times],
            observed_at,
        )
        temperature = OpenMeteoWeatherRepository._value_at(self.temperatures_c, idx)
        precipitation = OpenMeteoWeatherRepository._value_at(self.precipitations_mm, idx)
        wind_speed = OpenMeteoWeatherRepository._value_at(self.wind_speeds_kph, idx)
        severity = OpenMeteoWeatherRepository._severity(precipitation, wind_speed)
        return WeatherSnapshot(
            severity=severity,
            temperature_c=temperature,
            precipitation_mm=precipitation,
            wind_speed_kph=wind_speed,
            source=self.source,
            source_url=self.source_url,
            observed_at=observed_at,
        )


class WeatherRepository(Protocol):
    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        raise NotImplementedError

    async def fetch_profile(self, lat: float, lon: float, at: datetime | None = None) -> WeatherProfile:
        raise NotImplementedError


class OpenMeteoWeatherRepository(WeatherRepository):
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        profile = await self.fetch_profile(lat, lon, at)
        snapshot = profile.snapshot_at(at)
        idx = self._nearest_hour_index(
            [value.isoformat() for value in profile.times],
            snapshot.observed_at,
        )
        logger.warning(
            "Open-Meteo weather selected: at=%s idx=%s temp=%s precip=%s wind=%s severity=%.3f",
            snapshot.observed_at.isoformat(),
            idx,
            snapshot.temperature_c,
            snapshot.precipitation_mm,
            snapshot.wind_speed_kph,
            snapshot.severity,
        )
        return snapshot

    async def fetch_profile(self, lat: float, lon: float, at: datetime | None = None) -> WeatherProfile:
        url = f"{self._base_url}/v1/forecast"
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
        normalized_times = [self._to_utc(datetime.fromisoformat(value)) for value in times]
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
        return WeatherProfile(
            times=normalized_times,
            temperatures_c=[float(value) if value is not None else None for value in temperatures],
            precipitations_mm=[float(value) if value is not None else None for value in precipitations],
            wind_speeds_kph=[float(value) if value is not None else None for value in winds],
            source="open-meteo",
            source_url=url,
        )

    @staticmethod
    def _value_at(values: list[float | None] | None, idx: int | None) -> float | None:
        if not values:
            return None
        if idx is None or idx < 0 or idx >= len(values):
            idx = 0

        direct_value = values[idx]
        if direct_value is not None:
            return float(direct_value)

        for offset in range(1, len(values)):
            right = idx + offset
            if right < len(values):
                candidate = values[right]
                if candidate is not None:
                    return float(candidate)
            left = idx - offset
            if left >= 0:
                candidate = values[left]
                if candidate is not None:
                    return float(candidate)
        return None

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


class MetNoWeatherRepository(WeatherRepository):
    def __init__(
        self,
        base_url: str = "https://api.met.no",
        timeout_seconds: float = 10.0,
        user_agent: str = "RouteOptimizationLab/1.0 (contact: local)",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._headers = {"User-Agent": user_agent}

    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        profile = await self.fetch_profile(lat, lon, at)
        snapshot = profile.snapshot_at(at)
        idx = OpenMeteoWeatherRepository._nearest_hour_index(
            [value.isoformat() for value in profile.times],
            snapshot.observed_at,
        )
        logger.warning(
            "MET Norway weather selected: at=%s idx=%s temp=%s precip=%s wind=%s severity=%.3f",
            snapshot.observed_at.isoformat(),
            idx,
            snapshot.temperature_c,
            snapshot.precipitation_mm,
            snapshot.wind_speed_kph,
            snapshot.severity,
        )
        return snapshot

    async def fetch_profile(self, lat: float, lon: float, at: datetime | None = None) -> WeatherProfile:
        url = f"{self._base_url}/weatherapi/locationforecast/2.0/compact"
        params = {"lat": lat, "lon": lon}
        async with httpx.AsyncClient(
            timeout=max(self._timeout_seconds, 20.0),
            headers=self._headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()

        payload = response.json()
        timeseries = payload.get("properties", {}).get("timeseries") or []
        times: list[datetime] = []
        temperatures: list[float | None] = []
        precipitations: list[float | None] = []
        winds: list[float | None] = []

        for item in timeseries:
            raw_time = item.get("time")
            if not raw_time:
                continue
            try:
                parsed_time = OpenMeteoWeatherRepository._to_utc(datetime.fromisoformat(raw_time.replace("Z", "+00:00")))
            except ValueError:
                continue
            data = item.get("data") or {}
            instant = data.get("instant", {}).get("details", {})
            next_1h = data.get("next_1_hours", {}).get("details", {})
            next_6h = data.get("next_6_hours", {}).get("details", {})
            next_12h = data.get("next_12_hours", {}).get("details", {})
            precipitation = (
                next_1h.get("precipitation_amount")
                if next_1h.get("precipitation_amount") is not None
                else next_6h.get("precipitation_amount")
                if next_6h.get("precipitation_amount") is not None
                else next_12h.get("precipitation_amount")
            )
            wind_speed_ms = instant.get("wind_speed")

            times.append(parsed_time)
            temperatures.append(float(instant["air_temperature"]) if instant.get("air_temperature") is not None else None)
            precipitations.append(float(precipitation) if precipitation is not None else None)
            winds.append((float(wind_speed_ms) * 3.6) if wind_speed_ms is not None else None)

        logger.warning(
            "MET Norway weather raw: lat=%.6f lon=%.6f points=%d sample_time=%s sample_temp=%s sample_precip=%s sample_wind=%s",
            lat,
            lon,
            len(times),
            [value.isoformat() for value in times[:3]],
            temperatures[:3],
            precipitations[:3],
            winds[:3],
        )
        return WeatherProfile(
            times=times,
            temperatures_c=temperatures,
            precipitations_mm=precipitations,
            wind_speeds_kph=winds,
            source="met.no",
            source_url=url,
        )


class FallbackWeatherRepository(WeatherRepository):
    _PROFILE_HOURS = 48

    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        profile = await self.fetch_profile(lat, lon, at)
        return profile.snapshot_at(at)

    async def fetch_profile(self, lat: float, lon: float, at: datetime | None = None) -> WeatherProfile:
        observed_at = OpenMeteoWeatherRepository._to_utc(at or datetime.now(timezone.utc))
        start_at = observed_at.replace(minute=0, second=0, microsecond=0) - timedelta(hours=6)
        times: list[datetime] = []
        temperatures: list[float] = []
        precipitations: list[float] = []
        winds: list[float] = []

        for hour_offset in range(self._PROFILE_HOURS):
            timestamp = start_at + timedelta(hours=hour_offset)
            temperature_c, precipitation_mm, wind_speed_kph = self._fallback_conditions(lat, lon, timestamp)
            times.append(timestamp)
            temperatures.append(temperature_c)
            precipitations.append(precipitation_mm)
            winds.append(wind_speed_kph)

        return WeatherProfile(
            times=times,
            temperatures_c=temperatures,
            precipitations_mm=precipitations,
            wind_speeds_kph=winds,
            source="fallback-weather",
            source_url=None,
        )

    @staticmethod
    def _fallback_conditions(lat: float, lon: float, at: datetime) -> tuple[float, float, float]:
        normalized_at = OpenMeteoWeatherRepository._to_utc(at)
        day_of_year = normalized_at.timetuple().tm_yday
        hour = normalized_at.hour + normalized_at.minute / 60.0
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        seasonal = math.sin(((day_of_year - 80) / 365.0) * math.tau)
        diurnal = math.sin(((hour - 14.0) / 24.0) * math.tau)
        regional = math.sin(lat_rad * 2.6) + math.cos(lon_rad * 3.1)
        frontal_phase = (day_of_year * 0.11) + (hour * 0.42)

        temperature_c = 7.5 + (10.5 * seasonal) + (5.5 * diurnal) + (1.8 * regional)
        precipitation_signal = (
            0.58
            + (0.34 * math.sin(frontal_phase + (lon_rad * 2.2)))
            + (0.18 * math.cos((day_of_year * 0.07) - (lat_rad * 1.9)))
        )
        precipitation_mm = max(0.0, min(7.5, (precipitation_signal - 0.42) * 6.6))
        wind_signal = (
            10.8
            + (6.2 * (0.5 + 0.5 * math.sin((hour * 0.33) + (lon_rad * 2.8) + 0.9)))
            + (3.4 * (0.5 + 0.5 * math.cos((day_of_year * 0.05) - (lat_rad * 2.1))))
            + (0.85 * precipitation_mm)
        )
        wind_speed_kph = max(4.0, min(38.0, wind_signal))

        return (
            round(temperature_c, 1),
            round(precipitation_mm, 1),
            round(wind_speed_kph, 1),
        )


class CompositeWeatherRepository(WeatherRepository):
    def __init__(self, primary: WeatherRepository | None, fallback: WeatherRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, lat: float, lon: float, at: datetime | None = None) -> WeatherSnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(lat, lon, at)
            except Exception as exc:
                logger.warning("Weather provider primary failed: %s", repr(exc))
                return await self._fallback.fetch(lat, lon, at)
        return await self._fallback.fetch(lat, lon, at)

    async def fetch_profile(self, lat: float, lon: float, at: datetime | None = None) -> WeatherProfile:
        if self._primary is not None:
            try:
                return await self._primary.fetch_profile(lat, lon, at)
            except Exception as exc:
                logger.warning("Weather profile provider primary failed: %s", repr(exc))
                return await self._fallback.fetch_profile(lat, lon, at)
        return await self._fallback.fetch_profile(lat, lon, at)
