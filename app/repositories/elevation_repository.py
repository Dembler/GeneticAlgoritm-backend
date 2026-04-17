from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

import httpx

from app.domain.models import Point

logger = logging.getLogger(__name__)


@dataclass
class ElevationProfile:
    elevations_m: list[float]
    source: str
    source_url: str | None


class ElevationRepository(Protocol):
    async def fetch(self, points: list[Point]) -> ElevationProfile:
        raise NotImplementedError


class OpenMeteoElevationRepository(ElevationRepository):
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def fetch(self, points: list[Point]) -> ElevationProfile:
        if not points:
            return ElevationProfile(elevations_m=[], source="open-meteo-elevation", source_url=None)
        latitudes = ",".join([f"{p.lat:.7f}" for p in points])
        longitudes = ",".join([f"{p.lon:.7f}" for p in points])
        url = f"{self._base_url}/v1/elevation"
        params = {"latitude": latitudes, "longitude": longitudes}
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        elevations = payload.get("elevation") or payload.get("elevations") or []
        normalized = [float(v) for v in elevations]
        logger.warning(
            "Open-Meteo elevation raw: points=%d sample_points=%s raw_count=%d raw_sample=%s",
            len(points),
            [(p.lat, p.lon) for p in points[:3]],
            len(elevations),
            elevations[:5],
        )
        if len(normalized) != len(points):
            normalized = normalized[: len(points)]
            while len(normalized) < len(points):
                normalized.append(0.0)
        logger.warning(
            "Open-Meteo elevation normalized: count=%d sample=%s",
            len(normalized),
            normalized[:5],
        )
        return ElevationProfile(
            elevations_m=normalized,
            source="open-meteo-elevation",
            source_url=url,
        )


class FallbackElevationRepository(ElevationRepository):
    async def fetch(self, points: list[Point]) -> ElevationProfile:
        return ElevationProfile(
            elevations_m=[0.0 for _ in points],
            source="fallback-elevation",
            source_url=None,
        )


class CompositeElevationRepository(ElevationRepository):
    def __init__(self, primary: ElevationRepository | None, fallback: ElevationRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, points: list[Point]) -> ElevationProfile:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points)
            except Exception:
                return await self._fallback.fetch(points)
        return await self._fallback.fetch(points)
