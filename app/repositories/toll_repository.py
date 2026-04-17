from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Protocol

import httpx

from app.domain.models import Point, TransportProfile, VehicleClass

logger = logging.getLogger(__name__)


@dataclass
class TollSnapshot:
    toll_matrix: list[list[float]]
    currency: str | None
    source: str
    source_url: str | None
    observed_at: datetime


class TollRepository(Protocol):
    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        vehicle_class: VehicleClass,
        departure_at: datetime | None = None,
    ) -> TollSnapshot:
        raise NotImplementedError


class DisabledTollRepository(TollRepository):
    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        vehicle_class: VehicleClass,
        departure_at: datetime | None = None,
    ) -> TollSnapshot:
        _ = profile, vehicle_class
        size = len(points)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        return TollSnapshot(
            toll_matrix=matrix,
            currency="RUB",
            source="toll-disabled",
            source_url=None,
            observed_at=departure_at or datetime.now(timezone.utc),
        )


class TollGuruTollRepository(TollRepository):
    _VEHICLE_TYPE = {
        VehicleClass.passenger: "2AxlesAuto",
        VehicleClass.light_truck: "2AxlesTruck",
        VehicleClass.heavy_truck: "2AxlesTruck",
    }

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        max_concurrency: int = 8,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_concurrency = max(1, max_concurrency)

    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        vehicle_class: VehicleClass,
        departure_at: datetime | None = None,
    ) -> TollSnapshot:
        observed_at = self._to_utc(departure_at or datetime.now(timezone.utc))
        size = len(points)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        if size == 0:
            return TollSnapshot(
                toll_matrix=matrix,
                currency=None,
                source="tollguru",
                source_url=self._endpoint,
                observed_at=observed_at,
            )
        if profile != TransportProfile.driving:
            return TollSnapshot(
                toll_matrix=matrix,
                currency=None,
                source="toll-disabled-for-non-driving",
                source_url=None,
                observed_at=observed_at,
            )

        semaphore = asyncio.Semaphore(self._max_concurrency)
        headers = {"x-api-key": self._api_key, "content-type": "application/json"}
        tasks: list[asyncio.Task[tuple[int, int, float, str | None]]] = []

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            for i in range(size):
                for j in range(size):
                    if i == j:
                        continue
                    tasks.append(
                        asyncio.create_task(
                            self._edge_toll(
                                client=client,
                                semaphore=semaphore,
                                headers=headers,
                                origin=points[i],
                                destination=points[j],
                                vehicle_class=vehicle_class,
                                departure_at=observed_at,
                                i=i,
                                j=j,
                            )
                        )
                    )

            edge_values = await asyncio.gather(*tasks)

        currency: str | None = None
        for i, j, toll_cost, edge_currency in edge_values:
            matrix[i][j] = max(0.0, float(toll_cost))
            if currency is None and edge_currency:
                currency = edge_currency

        logger.warning(
            "TollGuru matrix: points=%d vehicle=%s currency=%s sample_row=%s",
            size,
            vehicle_class.value,
            currency,
            matrix[0][: min(4, len(matrix[0]))] if matrix else [],
        )
        return TollSnapshot(
            toll_matrix=matrix,
            currency=currency,
            source="tollguru",
            source_url=self._endpoint,
            observed_at=observed_at,
        )

    async def _edge_toll(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        headers: dict[str, str],
        origin: Point,
        destination: Point,
        vehicle_class: VehicleClass,
        departure_at: datetime,
        i: int,
        j: int,
    ) -> tuple[int, int, float, str | None]:
        payload = {
            "from": {"lat": origin.lat, "lng": origin.lon},
            "to": {"lat": destination.lat, "lng": destination.lon},
            "vehicleType": self._VEHICLE_TYPE[vehicle_class],
            "departure_time": departure_at.isoformat().replace("+00:00", "Z"),
        }
        async with semaphore:
            response = await client.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        toll_cost, currency = self._extract_toll_cost(body)
        return i, j, toll_cost, currency

    @property
    def _endpoint(self) -> str:
        return f"{self._base_url}/v1/calc/here"

    @staticmethod
    def _to_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _extract_toll_cost(cls, payload: dict[str, Any]) -> tuple[float, str | None]:
        routes = payload.get("routes") or []
        best_cost: float | None = None
        best_currency: str | None = None

        for route in routes:
            cost, currency = cls._extract_route_cost(route)
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_currency = currency

        return (0.0, best_currency) if best_cost is None else (max(0.0, best_cost), best_currency)

    @classmethod
    def _extract_route_cost(cls, route: dict[str, Any]) -> tuple[float | None, str | None]:
        candidates = cls._collect_numeric_costs(route.get("costs"))
        currency = cls._extract_currency(route)
        if not candidates:
            return None, currency
        return min(candidates), currency

    @staticmethod
    def _collect_numeric_costs(value: Any) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)] if value > 0 else []
        if isinstance(value, list):
            costs: list[float] = []
            for item in value:
                costs.extend(TollGuruTollRepository._collect_numeric_costs(item))
            return costs
        if not isinstance(value, dict):
            return []

        costs: list[float] = []
        for key, nested in value.items():
            key_lower = str(key).lower()
            if key_lower in {"fuel", "fuelcost", "fuel_cost"}:
                continue
            if "cost" in key_lower or key_lower in {"cash", "tag", "licenseplate"}:
                costs.extend(TollGuruTollRepository._collect_numeric_costs(nested))
        return costs

    @staticmethod
    def _extract_currency(route: dict[str, Any]) -> str | None:
        costs = route.get("costs") or {}
        if isinstance(costs, dict):
            currency = costs.get("currency")
            if isinstance(currency, str) and currency:
                return currency
        summary = route.get("summary") or {}
        if isinstance(summary, dict):
            currency = summary.get("currency")
            if isinstance(currency, str) and currency:
                return currency
        return None


class CompositeTollRepository(TollRepository):
    def __init__(self, primary: TollRepository | None, fallback: TollRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    async def fetch(
        self,
        points: list[Point],
        profile: TransportProfile,
        vehicle_class: VehicleClass,
        departure_at: datetime | None = None,
    ) -> TollSnapshot:
        if self._primary is not None:
            try:
                return await self._primary.fetch(points, profile, vehicle_class, departure_at)
            except Exception:
                return await self._fallback.fetch(points, profile, vehicle_class, departure_at)
        return await self._fallback.fetch(points, profile, vehicle_class, departure_at)
