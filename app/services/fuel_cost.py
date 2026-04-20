from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time

from app.domain.models import FuelCostBreakdown, FuelType, RouteRequest, VehicleClass
from app.repositories.fuel_price_repository import FuelPriceInfo, FuelPriceRepository

logger = logging.getLogger(__name__)


@dataclass
class FuelPriceSnapshot:
    petrol_rub_per_liter: float
    diesel_rub_per_liter: float
    currency: str
    source: str
    source_url: str | None
    price_date: str | None
    retrieved_at: datetime


class FuelPriceService:
    def __init__(
        self,
        repository: FuelPriceRepository | None,
        fallback_petrol: float,
        fallback_diesel: float,
        currency: str,
        source_name: str,
        source_url: str | None,
        cache_ttl_sec: int = 3600,
    ) -> None:
        self._repository = repository
        self._fallback_petrol = fallback_petrol
        self._fallback_diesel = fallback_diesel
        self._currency = currency
        self._source_name = source_name
        self._source_url = source_url
        self._cache_ttl_sec = cache_ttl_sec
        self._cache: FuelPriceSnapshot | None = None
        self._cache_expires_at = 0.0

    async def get_prices(self) -> FuelPriceSnapshot:
        now = time.time()
        if self._cache is not None and now < self._cache_expires_at:
            logger.warning(
                "Fuel price snapshot cache hit: source=%s petrol=%.3f diesel=%.3f currency=%s price_date=%s retrieved_at=%s",
                self._cache.source,
                self._cache.petrol_rub_per_liter,
                self._cache.diesel_rub_per_liter,
                self._cache.currency,
                self._cache.price_date,
                self._cache.retrieved_at.isoformat(),
            )
            return self._cache

        if self._repository is None:
            snapshot = self._fallback_snapshot("source is not configured")
            self._cache = snapshot
            self._cache_expires_at = now + self._cache_ttl_sec
            logger.warning(
                "Fuel price snapshot fallback: source=%s petrol=%.3f diesel=%.3f currency=%s",
                snapshot.source,
                snapshot.petrol_rub_per_liter,
                snapshot.diesel_rub_per_liter,
                snapshot.currency,
            )
            return snapshot

        try:
            fetched = await self._repository.fetch()
            snapshot = self._from_info(fetched)
            self._cache = snapshot
            self._cache_expires_at = now + self._cache_ttl_sec
            logger.warning(
                "Fuel price snapshot fetched: source=%s petrol=%.3f diesel=%.3f currency=%s price_date=%s retrieved_at=%s",
                snapshot.source,
                snapshot.petrol_rub_per_liter,
                snapshot.diesel_rub_per_liter,
                snapshot.currency,
                snapshot.price_date,
                snapshot.retrieved_at.isoformat(),
            )
            return snapshot
        except Exception:
            snapshot = self._fallback_snapshot("source is unavailable")
            self._cache = snapshot
            self._cache_expires_at = now + self._cache_ttl_sec
            logger.warning(
                "Fuel price snapshot fallback after error: source=%s petrol=%.3f diesel=%.3f currency=%s",
                snapshot.source,
                snapshot.petrol_rub_per_liter,
                snapshot.diesel_rub_per_liter,
                snapshot.currency,
            )
            return snapshot

    def _from_info(self, info: FuelPriceInfo) -> FuelPriceSnapshot:
        return FuelPriceSnapshot(
            petrol_rub_per_liter=info.petrol_rub_per_liter,
            diesel_rub_per_liter=info.diesel_rub_per_liter,
            currency=self._currency,
            source=info.source,
            source_url=info.source_url,
            price_date=info.price_date,
            retrieved_at=info.retrieved_at,
        )

    def _fallback_snapshot(self, note: str) -> FuelPriceSnapshot:
        return FuelPriceSnapshot(
            petrol_rub_per_liter=self._fallback_petrol,
            diesel_rub_per_liter=self._fallback_diesel,
            currency=self._currency,
            source=f"{self._source_name} ({note})",
            source_url=self._source_url,
            price_date=None,
            retrieved_at=datetime.now(timezone.utc),
        )


class FuelCostService:
    _DEFAULT_CONSUMPTION = {
        VehicleClass.passenger: {FuelType.petrol: 8.5, FuelType.diesel: 6.8},
        VehicleClass.light_truck: {FuelType.petrol: 12.5, FuelType.diesel: 10.5},
        VehicleClass.heavy_truck: {FuelType.petrol: 26.0, FuelType.diesel: 22.0},
    }

    _UPHILL_PENALTY = {
        VehicleClass.passenger: 0.25,
        VehicleClass.light_truck: 0.3,
        VehicleClass.heavy_truck: 0.35,
    }

    _DOWNHILL_BONUS = {
        VehicleClass.passenger: 0.08,
        VehicleClass.light_truck: 0.1,
        VehicleClass.heavy_truck: 0.12,
    }

    _CO2_KG_PER_LITER = {
        FuelType.petrol: 2.31,
        FuelType.diesel: 2.68,
    }
    _REF_TEMP_C = 25.0
    # Based on DOE/FuelEconomy observations: around 20 F (-6.7 C),
    # fuel economy can drop by ~15% (city) and ~24% (short trips).
    _COLD_ANCHOR_TEMP_C = -6.7  # 20 F
    _COLD_EXTREME_TEMP_C = -20.0
    _COLD_CITY_LOSS_AT_ANCHOR = 0.15
    _COLD_SHORT_LOSS_AT_ANCHOR = 0.24
    _COLD_CITY_LOSS_AT_EXTREME = 0.20
    _COLD_SHORT_LOSS_AT_EXTREME = 0.33
    # Hot-weather losses are modeled conservatively from DOE notes
    # about A/C-related economy losses in very hot conditions.
    _HOT_START_TEMP_C = 30.0
    _HOT_EXTREME_TEMP_C = 40.0
    _HOT_CITY_LOSS_AT_EXTREME = 0.10
    _HOT_SHORT_LOSS_AT_EXTREME = 0.25
    _SHORT_TRIP_KM = 6.4  # about 4 miles
    _CITY_TRIP_KM = 20.0
    _MOUNTAIN_REF_CONSUMPTION_L100 = 24.815
    _MOUNTAIN_REF_SLOPE_COEFF = 2.246
    _MOUNTAIN_BASE_L100 = {
        VehicleClass.passenger: 8.2,
        VehicleClass.light_truck: 11.8,
        VehicleClass.heavy_truck: 24.815,
    }
    _MOUNTAIN_SENSITIVITY_SCALE = {
        VehicleClass.passenger: 0.90,
        VehicleClass.light_truck: 0.97,
        VehicleClass.heavy_truck: 1.0,
    }
    _PRESSURE_SENSITIVITY = {
        VehicleClass.passenger: 0.28,
        VehicleClass.light_truck: 0.24,
        VehicleClass.heavy_truck: 0.2,
    }
    _SEA_LEVEL_PRESSURE_KPA = 101.325

    def __init__(self, price_service: FuelPriceService) -> None:
        self._price_service = price_service

    async def get_price_snapshot(self) -> FuelPriceSnapshot:
        return await self._price_service.get_prices()

    def resolve_consumption_l_per_100km(self, request: RouteRequest) -> float:
        if request.fuel_consumption_l_per_100km is not None:
            return request.fuel_consumption_l_per_100km
        return self._DEFAULT_CONSUMPTION[request.vehicle_class][request.fuel_type]

    def terrain_multiplier(self, vehicle: VehicleClass, uphill_pct: float, downhill_pct: float) -> float:
        uphill_factor = self._UPHILL_PENALTY[vehicle]
        downhill_factor = self._DOWNHILL_BONUS[vehicle]
        multiplier = 1 + (uphill_pct / 100.0) * uphill_factor - (downhill_pct / 100.0) * downhill_factor
        return max(0.6, min(1.6, multiplier))

    def price_per_liter(self, snapshot: FuelPriceSnapshot, fuel_type: FuelType) -> float:
        if fuel_type == FuelType.petrol:
            return snapshot.petrol_rub_per_liter
        return snapshot.diesel_rub_per_liter

    def compute_liters(self, distance_km: float, consumption_l_per_100km: float, terrain_multiplier: float) -> float:
        return distance_km * consumption_l_per_100km / 100.0 * terrain_multiplier

    def estimate_co2_kg(self, liters: float, fuel_type: FuelType) -> float:
        return liters * self._CO2_KG_PER_LITER[fuel_type]

    def temperature_multiplier(self, temperature_c: float | None, trip_distance_km: float) -> float:
        if temperature_c is None:
            return 1.0

        short_trip_share = self._short_trip_share(trip_distance_km)
        city_loss = 0.0
        short_loss = 0.0

        if temperature_c < self._REF_TEMP_C:
            city_loss, short_loss = self._cold_losses(temperature_c)
        elif temperature_c > self._HOT_START_TEMP_C:
            city_loss, short_loss = self._hot_losses(temperature_c)

        effective_loss = city_loss * (1.0 - short_trip_share) + short_loss * short_trip_share
        effective_loss = self._clamp(effective_loss, 0.0, 0.5)
        return 1.0 / max(1e-9, 1.0 - effective_loss)

    def mountain_slope_multiplier(self, vehicle: VehicleClass, uphill_pct: float) -> float:
        i = self._clamp(uphill_pct, 0.0, 30.0)
        base = self._MOUNTAIN_BASE_L100[vehicle]
        sensitivity = self._MOUNTAIN_SENSITIVITY_SCALE[vehicle]
        slope_coeff = self._MOUNTAIN_REF_SLOPE_COEFF * (base / self._MOUNTAIN_REF_CONSUMPTION_L100) * sensitivity
        q_l100 = base + slope_coeff * i
        return max(0.6, min(2.2, q_l100 / base))

    def altitude_pressure_multiplier(self, vehicle: VehicleClass, mean_elevation_m: float | None) -> float:
        if mean_elevation_m is None:
            return 1.0
        h = self._clamp(mean_elevation_m, -430.0, 5000.0)
        pressure_kpa = self._pressure_at_altitude_kpa(h)
        pressure_ratio = pressure_kpa / self._SEA_LEVEL_PRESSURE_KPA
        if pressure_ratio <= 0:
            return 1.0
        sensitivity = self._PRESSURE_SENSITIVITY[vehicle]
        return max(0.8, min(1.4, 1.0 + sensitivity * ((1.0 / pressure_ratio) - 1.0)))

    def mountain_multiplier(
        self,
        vehicle: VehicleClass,
        uphill_pct: float,
        mean_elevation_m: float | None,
    ) -> float:
        return self.mountain_slope_multiplier(vehicle, uphill_pct) * self.altitude_pressure_multiplier(
            vehicle,
            mean_elevation_m,
        )

    def _cold_losses(self, temperature_c: float) -> tuple[float, float]:
        if temperature_c >= self._COLD_ANCHOR_TEMP_C:
            scale = (self._REF_TEMP_C - temperature_c) / (self._REF_TEMP_C - self._COLD_ANCHOR_TEMP_C)
            scale = self._clamp(scale, 0.0, 1.0)
            return (
                self._COLD_CITY_LOSS_AT_ANCHOR * scale,
                self._COLD_SHORT_LOSS_AT_ANCHOR * scale,
            )

        if temperature_c <= self._COLD_EXTREME_TEMP_C:
            return (
                self._COLD_CITY_LOSS_AT_EXTREME,
                self._COLD_SHORT_LOSS_AT_EXTREME,
            )

        extra_scale = (self._COLD_ANCHOR_TEMP_C - temperature_c) / (
            self._COLD_ANCHOR_TEMP_C - self._COLD_EXTREME_TEMP_C
        )
        extra_scale = self._clamp(extra_scale, 0.0, 1.0)
        city = self._lerp(self._COLD_CITY_LOSS_AT_ANCHOR, self._COLD_CITY_LOSS_AT_EXTREME, extra_scale)
        short = self._lerp(self._COLD_SHORT_LOSS_AT_ANCHOR, self._COLD_SHORT_LOSS_AT_EXTREME, extra_scale)
        return city, short

    def _hot_losses(self, temperature_c: float) -> tuple[float, float]:
        if temperature_c <= self._HOT_START_TEMP_C:
            return 0.0, 0.0
        scale = (temperature_c - self._HOT_START_TEMP_C) / (self._HOT_EXTREME_TEMP_C - self._HOT_START_TEMP_C)
        scale = self._clamp(scale, 0.0, 1.0)
        return (
            self._HOT_CITY_LOSS_AT_EXTREME * scale,
            self._HOT_SHORT_LOSS_AT_EXTREME * scale,
        )

    def _short_trip_share(self, trip_distance_km: float) -> float:
        if trip_distance_km <= self._SHORT_TRIP_KM:
            return 1.0
        if trip_distance_km >= self._CITY_TRIP_KM:
            return 0.0
        span = self._CITY_TRIP_KM - self._SHORT_TRIP_KM
        return self._clamp((self._CITY_TRIP_KM - trip_distance_km) / span, 0.0, 1.0)

    def _pressure_at_altitude_kpa(self, altitude_m: float) -> float:
        if altitude_m <= 0:
            return self._SEA_LEVEL_PRESSURE_KPA
        return self._SEA_LEVEL_PRESSURE_KPA * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    async def compute(
        self,
        request: RouteRequest,
        distance_km: float,
        uphill_pct: float = 0.0,
        downhill_pct: float = 0.0,
        temperature_c: float | None = None,
        congestion_index: float = 0.0,
        mean_elevation_m: float | None = None,
    ) -> FuelCostBreakdown:
        prices = await self._price_service.get_prices()
        base_consumption = self.resolve_consumption_l_per_100km(request)
        terrain_multiplier = self.terrain_multiplier(
            request.vehicle_class, uphill_pct, downhill_pct
        )
        mountain_multiplier = self.mountain_multiplier(
            request.vehicle_class,
            uphill_pct,
            mean_elevation_m,
        )
        temperature_multiplier = self.temperature_multiplier(temperature_c, distance_km)
        congestion_multiplier = 1.0 + (0.2 * self._clamp(congestion_index, 0.0, 1.0))
        liters_total = self.compute_liters(distance_km, base_consumption, terrain_multiplier)
        liters_total *= mountain_multiplier
        liters_total *= temperature_multiplier
        liters_total *= congestion_multiplier
        price_per_liter = self.price_per_liter(prices, request.fuel_type)
        total_cost = liters_total * price_per_liter
        logger.warning(
            "Fuel cost debug: fuel_type=%s vehicle_class=%s distance_km=%.3f base_consumption=%.3f uphill_pct=%.3f downhill_pct=%.3f mean_elevation_m=%s temperature_c=%s congestion_index=%.3f terrain_multiplier=%.4f mountain_multiplier=%.4f temperature_multiplier=%.4f congestion_multiplier=%.4f liters_total=%.3f price_per_liter=%.3f total_cost=%.3f source=%s",
            request.fuel_type.value,
            request.vehicle_class.value,
            distance_km,
            base_consumption,
            uphill_pct,
            downhill_pct,
            None if mean_elevation_m is None else round(mean_elevation_m, 3),
            None if temperature_c is None else round(temperature_c, 3),
            congestion_index,
            terrain_multiplier,
            mountain_multiplier,
            temperature_multiplier,
            congestion_multiplier,
            liters_total,
            price_per_liter,
            total_cost,
            prices.source,
        )

        return FuelCostBreakdown(
            fuel_type=request.fuel_type,
            vehicle_class=request.vehicle_class,
            consumption_l_per_100km=base_consumption,
            distance_km=distance_km,
            uphill_share_pct=uphill_pct,
            downhill_share_pct=downhill_pct,
            terrain_multiplier=terrain_multiplier,
            mountain_multiplier=mountain_multiplier,
            temperature_multiplier=temperature_multiplier,
            congestion_multiplier=congestion_multiplier,
            liters_total=liters_total,
            price_per_liter=price_per_liter,
            total_cost=total_cost,
            currency=prices.currency,
            price_source=prices.source,
            price_source_url=prices.source_url,
            price_date=prices.price_date,
            price_retrieved_at=prices.retrieved_at.isoformat(),
        )
