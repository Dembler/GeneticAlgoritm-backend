from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx

from app.domain.models import Point


class GeocodingProviderError(RuntimeError):
    pass


@dataclass
class GeocodingResult:
    point: Point
    address: str | None
    score: float | None
    entity_type: str | None
    provider: str


class GeocodingRepository(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        country_set: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        radius_m: int | None = None,
        language: str | None = None,
    ) -> list[GeocodingResult]:
        raise NotImplementedError

    async def reverse(
        self,
        lat: float,
        lon: float,
        *,
        language: str | None = None,
    ) -> GeocodingResult:
        raise NotImplementedError


class DisabledGeocodingRepository(GeocodingRepository):
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        country_set: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        radius_m: int | None = None,
        language: str | None = None,
    ) -> list[GeocodingResult]:
        return []

    async def reverse(
        self,
        lat: float,
        lon: float,
        *,
        language: str | None = None,
    ) -> GeocodingResult:
        return GeocodingResult(
            point=Point(lat=lat, lon=lon),
            address=None,
            score=None,
            entity_type=None,
            provider="geocoding-disabled",
        )


class TomTomSearchRepository(GeocodingRepository):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        country_set: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        radius_m: int | None = None,
        language: str | None = None,
    ) -> list[GeocodingResult]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []
        url = f"{self._base_url}/search/2/search/{quote(cleaned_query)}.json"
        params: dict[str, object] = {
            "key": self._api_key,
            "limit": max(1, min(100, limit)),
        }
        if country_set:
            params["countrySet"] = country_set
        if language:
            params["language"] = language
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
            if radius_m is not None:
                params["radius"] = max(1, radius_m)

        async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True) as client:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            raise GeocodingProviderError(f"TomTom search returned status {response.status_code}.")

        results = response.json().get("results") or []
        return [result for item in results if (result := self._parse_result(item)) is not None]

    async def reverse(
        self,
        lat: float,
        lon: float,
        *,
        language: str | None = None,
    ) -> GeocodingResult:
        url = f"{self._base_url}/search/2/reverseGeocode/{lat},{lon}.json"
        params: dict[str, object] = {"key": self._api_key}
        if language:
            params["language"] = language
        async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True) as client:
            response = await client.get(url, params=params)
        if response.status_code != 200:
            raise GeocodingProviderError(f"TomTom reverse geocode returned status {response.status_code}.")

        addresses = response.json().get("addresses") or []
        if not addresses:
            raise GeocodingProviderError("TomTom reverse geocode returned no addresses.")
        first = addresses[0]
        address = first.get("address") if isinstance(first, dict) else None
        position = first.get("position") if isinstance(first, dict) else None
        point = self._point(position) or Point(lat=lat, lon=lon)
        return GeocodingResult(
            point=point,
            address=self._freeform_address(address),
            score=None,
            entity_type="Address",
            provider="tomtom-reverse-geocode",
        )

    def _parse_result(self, item: object) -> GeocodingResult | None:
        if not isinstance(item, dict):
            return None
        point = self._point(item.get("position"))
        if point is None:
            return None
        score = item.get("score")
        return GeocodingResult(
            point=point,
            address=self._freeform_address(item.get("address")),
            score=float(score) if isinstance(score, int | float) else None,
            entity_type=str(item.get("type")) if item.get("type") is not None else None,
            provider="tomtom-search",
        )

    @staticmethod
    def _point(value: object) -> Point | None:
        if not isinstance(value, dict):
            return None
        lat = value.get("lat")
        lon = value.get("lon")
        if not isinstance(lat, int | float) or not isinstance(lon, int | float):
            return None
        return Point(lat=float(lat), lon=float(lon))

    @staticmethod
    def _freeform_address(value: object) -> str | None:
        if not isinstance(value, dict):
            return None
        freeform = value.get("freeformAddress")
        if isinstance(freeform, str) and freeform:
            return freeform
        parts = [
            value.get("streetName"),
            value.get("municipality"),
            value.get("countrySubdivision"),
            value.get("country"),
        ]
        normalized = [str(part) for part in parts if part]
        return ", ".join(normalized) if normalized else None


class CompositeGeocodingRepository(GeocodingRepository):
    def __init__(self, primary: GeocodingRepository | None, fallback: GeocodingRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        country_set: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        radius_m: int | None = None,
        language: str | None = None,
    ) -> list[GeocodingResult]:
        if self._primary is not None:
            try:
                return await self._primary.search(
                    query,
                    limit=limit,
                    country_set=country_set,
                    lat=lat,
                    lon=lon,
                    radius_m=radius_m,
                    language=language,
                )
            except Exception:
                return await self._fallback.search(
                    query,
                    limit=limit,
                    country_set=country_set,
                    lat=lat,
                    lon=lon,
                    radius_m=radius_m,
                    language=language,
                )
        return await self._fallback.search(
            query,
            limit=limit,
            country_set=country_set,
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            language=language,
        )

    async def reverse(
        self,
        lat: float,
        lon: float,
        *,
        language: str | None = None,
    ) -> GeocodingResult:
        if self._primary is not None:
            try:
                return await self._primary.reverse(lat, lon, language=language)
            except Exception:
                return await self._fallback.reverse(lat, lon, language=language)
        return await self._fallback.reverse(lat, lon, language=language)
