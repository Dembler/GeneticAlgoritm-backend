from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import math
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


class OpenTopoDataElevationRepository(ElevationRepository):
    def __init__(
        self,
        base_url: str = "https://api.opentopodata.org",
        timeout_seconds: float = 10.0,
        datasets: tuple[str, ...] = ("mapzen", "aster30m", "srtm30m"),
        batch_size: int = 80,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._datasets = datasets
        self._batch_size = max(1, batch_size)

    async def fetch(self, points: list[Point]) -> ElevationProfile:
        if not points:
            return ElevationProfile(elevations_m=[], source="opentopodata", source_url=None)

        for dataset in self._datasets:
            try:
                elevations = await self._fetch_dataset(points, dataset)
                if any(value != 0.0 for value in elevations):
                    logger.warning(
                        "OpenTopoData elevation normalized: dataset=%s count=%d sample=%s",
                        dataset,
                        len(elevations),
                        elevations[:5],
                    )
                    return ElevationProfile(
                        elevations_m=elevations,
                        source=f"opentopodata-{dataset}",
                        source_url=f"{self._base_url}/v1/{dataset}",
                    )
            except Exception as exc:
                logger.warning("OpenTopoData dataset failed: dataset=%s error=%s", dataset, repr(exc))

        raise RuntimeError("No OpenTopoData dataset returned usable elevations.")

    async def _fetch_dataset(self, points: list[Point], dataset: str) -> list[float]:
        url = f"{self._base_url}/v1/{dataset}"
        normalized: list[float] = []
        async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True) as client:
            for start in range(0, len(points), self._batch_size):
                chunk = points[start : start + self._batch_size]
                params = {"locations": "|".join(f"{point.lat:.7f},{point.lon:.7f}" for point in chunk)}
                response = await client.get(url, params=params)
                if response.status_code == 429:
                    await asyncio.sleep(0.6)
                    response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results") or []
                logger.warning(
                    "OpenTopoData elevation raw: dataset=%s points=%d sample_points=%s raw_count=%d raw_sample=%s",
                    dataset,
                    len(chunk),
                    [(p.lat, p.lon) for p in chunk[:3]],
                    len(results),
                    results[:3],
                )
                batch_values = []
                for item in results[: len(chunk)]:
                    value = item.get("elevation")
                    batch_values.append(float(value) if value is not None else 0.0)
                while len(batch_values) < len(chunk):
                    batch_values.append(0.0)
                normalized.extend(batch_values)
                if start + self._batch_size < len(points):
                    await asyncio.sleep(0.15)
        return normalized[: len(points)]


class FallbackElevationRepository(ElevationRepository):
    async def fetch(self, points: list[Point]) -> ElevationProfile:
        elevations = [self._synthetic_elevation(point) for point in points]
        return ElevationProfile(
            elevations_m=elevations,
            source="fallback-elevation",
            source_url=None,
        )

    @staticmethod
    def _synthetic_elevation(point: Point) -> float:
        lat_rad = math.radians(point.lat)
        lon_rad = math.radians(point.lon)
        undulation = (
            170.0
            + (95.0 * math.sin(lat_rad * 3.4))
            + (70.0 * math.cos(lon_rad * 2.8))
            + (36.0 * math.sin((lat_rad + lon_rad) * 5.1))
            + (22.0 * math.cos((lat_rad - lon_rad) * 6.4))
        )
        return round(max(15.0, min(780.0, undulation)), 1)


class CompositeElevationRepository(ElevationRepository):
    def __init__(self, primary: ElevationRepository | None, fallback: ElevationRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(self, points: list[Point]) -> ElevationProfile:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points)
            except Exception as exc:
                logger.warning("Elevation provider primary failed: %s", repr(exc))
                return await self._fallback.fetch(points)
        return await self._fallback.fetch(points)
