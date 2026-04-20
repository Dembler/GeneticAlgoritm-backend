from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings
from app.controllers.route_controller import RouteController
from app.repositories.cache_repository import InMemoryRouteCacheRepository
from app.repositories.elevation_repository import (
    CompositeElevationRepository,
    FallbackElevationRepository,
    OpenTopoDataElevationRepository,
    OpenMeteoElevationRepository,
)
from app.repositories.fuel_price_repository import RosstatFuelPriceRepository
from app.repositories.routing_repository import (
    CompositeRoutingRepository,
    FallbackRoutingRepository,
    OsrmRoutingRepository,
)
from app.repositories.run_repository import SqliteRouteRunRepository
from app.repositories.traffic_repository import (
    CompositeTrafficRepository,
    DisabledTrafficRepository,
)
from app.repositories.toll_repository import (
    CompositeTollRepository,
    DisabledTollRepository,
    TollGuruTollRepository,
)
from app.repositories.weather_repository import (
    CompositeWeatherRepository,
    FallbackWeatherRepository,
    MetNoWeatherRepository,
    OpenMeteoWeatherRepository,
)
from app.services.context_service import ContextService
from app.services.criteria_service import CriteriaService
from app.services.dynamic_weights_service import DynamicWeightsService
from app.services.fuel_cost import FuelCostService, FuelPriceService
from app.services.route_analysis_service import RouteAnalysisService
from app.services.route_optimizer import RouteOptimizer
from app.services.route_service import RouteService
from app.services.terrain_profile_service import TerrainProfileService


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_cache_repository() -> InMemoryRouteCacheRepository:
    settings = get_settings()
    return InMemoryRouteCacheRepository(ttl_seconds=settings.cache_ttl_sec)


@lru_cache
def get_run_repository() -> SqliteRouteRunRepository:
    settings = get_settings()
    return SqliteRouteRunRepository(db_path=settings.route_runs_db_path)


@lru_cache
def get_routing_repository() -> CompositeRoutingRepository:
    settings = get_settings()
    primary = None
    if settings.osrm_enabled:
        primary = OsrmRoutingRepository(
            base_url=settings.osrm_base_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    fallback = FallbackRoutingRepository()
    return CompositeRoutingRepository(primary=primary, fallback=fallback)


@lru_cache
def get_weather_repository() -> CompositeWeatherRepository:
    settings = get_settings()
    primary = None
    if settings.weather_enabled:
        primary = MetNoWeatherRepository(
            base_url=settings.metno_base_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    fallback_primary = None
    if settings.weather_enabled:
        fallback_primary = OpenMeteoWeatherRepository(
            base_url=settings.openmeteo_base_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    fallback = FallbackWeatherRepository()
    secondary = CompositeWeatherRepository(primary=fallback_primary, fallback=fallback)
    return CompositeWeatherRepository(primary=primary, fallback=secondary)


@lru_cache
def get_elevation_repository() -> CompositeElevationRepository:
    settings = get_settings()
    primary = None
    if settings.elevation_enabled:
        primary = OpenTopoDataElevationRepository(
            base_url=settings.opentopodata_base_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    fallback_primary = None
    if settings.elevation_enabled:
        fallback_primary = OpenMeteoElevationRepository(
            base_url=settings.openmeteo_base_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    fallback = FallbackElevationRepository()
    secondary = CompositeElevationRepository(primary=fallback_primary, fallback=fallback)
    return CompositeElevationRepository(primary=primary, fallback=secondary)


@lru_cache
def get_traffic_repository() -> CompositeTrafficRepository:
    primary = None
    fallback = DisabledTrafficRepository()
    return CompositeTrafficRepository(primary=primary, fallback=fallback)


@lru_cache
def get_toll_repository() -> CompositeTollRepository:
    settings = get_settings()
    primary = None
    if settings.toll_enabled and settings.toll_api_key:
        primary = TollGuruTollRepository(
            base_url=settings.toll_base_url,
            api_key=settings.toll_api_key,
            timeout_seconds=settings.request_timeout_sec,
            max_concurrency=settings.toll_max_concurrency,
        )
    fallback = DisabledTollRepository()
    return CompositeTollRepository(primary=primary, fallback=fallback)


@lru_cache
def get_fuel_price_service() -> FuelPriceService:
    settings = get_settings()
    repository = None
    if settings.fuel_price_source_url:
        repository = RosstatFuelPriceRepository(
            source_url=settings.fuel_price_source_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    return FuelPriceService(
        repository=repository,
        fallback_petrol=settings.fuel_price_fallback_petrol,
        fallback_diesel=settings.fuel_price_fallback_diesel,
        currency=settings.fuel_price_currency,
        source_name=settings.fuel_price_source_name,
        source_url=settings.fuel_price_source_url,
        cache_ttl_sec=settings.fuel_price_cache_ttl_sec,
    )


def get_fuel_cost_service(
    price_service: FuelPriceService = Depends(get_fuel_price_service),
) -> FuelCostService:
    return FuelCostService(price_service=price_service)


@lru_cache
def get_terrain_profile_service() -> TerrainProfileService:
    return TerrainProfileService(elevation_repository=get_elevation_repository())


def get_context_service(
    routing_repository: CompositeRoutingRepository = Depends(get_routing_repository),
    weather_repository: CompositeWeatherRepository = Depends(get_weather_repository),
    elevation_repository: CompositeElevationRepository = Depends(get_elevation_repository),
    traffic_repository: CompositeTrafficRepository = Depends(get_traffic_repository),
    toll_repository: CompositeTollRepository = Depends(get_toll_repository),
    terrain_profile_service: TerrainProfileService = Depends(get_terrain_profile_service),
) -> ContextService:
    return ContextService(
        routing_repository=routing_repository,
        weather_repository=weather_repository,
        elevation_repository=elevation_repository,
        traffic_repository=traffic_repository,
        toll_repository=toll_repository,
        terrain_profile_service=terrain_profile_service,
    )


def get_criteria_service(
    fuel_cost_service: FuelCostService = Depends(get_fuel_cost_service),
) -> CriteriaService:
    return CriteriaService(fuel_cost_service=fuel_cost_service)


@lru_cache
def get_dynamic_weights_service() -> DynamicWeightsService:
    return DynamicWeightsService()


@lru_cache
def get_route_analysis_service() -> RouteAnalysisService:
    return RouteAnalysisService()


def get_optimizer(
    criteria_service: CriteriaService = Depends(get_criteria_service),
) -> RouteOptimizer:
    return RouteOptimizer(criteria_service=criteria_service)


def get_route_service(
    optimizer: RouteOptimizer = Depends(get_optimizer),
    routing_repository: CompositeRoutingRepository = Depends(get_routing_repository),
    cache_repository: InMemoryRouteCacheRepository = Depends(get_cache_repository),
    fuel_cost_service: FuelCostService = Depends(get_fuel_cost_service),
    context_service: ContextService = Depends(get_context_service),
    dynamic_weights_service: DynamicWeightsService = Depends(get_dynamic_weights_service),
    route_analysis_service: RouteAnalysisService = Depends(get_route_analysis_service),
    terrain_profile_service: TerrainProfileService = Depends(get_terrain_profile_service),
    run_repository: SqliteRouteRunRepository = Depends(get_run_repository),
    settings: Settings = Depends(get_settings),
) -> RouteService:
    return RouteService(
        optimizer=optimizer,
        routing_repository=routing_repository,
        cache_repository=cache_repository,
        fuel_cost_service=fuel_cost_service,
        context_service=context_service,
        dynamic_weights_service=dynamic_weights_service,
        route_analysis_service=route_analysis_service,
        terrain_profile_service=terrain_profile_service,
        run_repository=run_repository,
        default_population=settings.ga_default_population,
        default_generations=settings.ga_default_generations,
        pareto_enabled=settings.optimizer_enable_pareto,
    )


def get_route_controller(
    service: RouteService = Depends(get_route_service),
) -> RouteController:
    return RouteController(route_service=service)
