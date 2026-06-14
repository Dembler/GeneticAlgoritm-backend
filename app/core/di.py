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
from app.repositories.geocoding_repository import (
    CompositeGeocodingRepository,
    DisabledGeocodingRepository,
    TomTomSearchRepository,
)
from app.repositories.infrastructure_repository import (
    CompositeInfrastructureRepository,
    JsonInfrastructureRepository,
    OverpassInfrastructureRepository,
    SyntheticInfrastructureRepository,
    UnrestrictedInfrastructureRepository,
)
from app.repositories.osm_overpass_repository import OverpassRoadDataClient
from app.repositories.routing_repository import (
    CompositeRoutingRepository,
    FallbackRoutingRepository,
    OpenRouteServiceRoutingRepository,
    OsrmRoutingRepository,
    TomTomRoutingRepository,
)
from app.repositories.road_quality_repository import (
    CompositeRoadQualityRepository,
    FullQualityRoadQualityRepository,
    JsonRoadQualityRepository,
    OverpassRoadQualityRepository,
    SyntheticRoadQualityRepository,
)
from app.repositories.road_event_repository import (
    CompositeRoadEventRepository,
    DisabledRoadEventRepository,
    JsonRoadEventRepository,
    OverpassRoadEventRepository,
    SyntheticRoadEventRepository,
    TomTomIncidentRepository,
)
from app.repositories.run_repository import SqliteRouteRunRepository
from app.repositories.traffic_repository import (
    CompositeTrafficRepository,
    DisabledTrafficRepository,
    JsonTrafficRepository,
    SyntheticTrafficRepository,
    TomTomTrafficRepository,
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
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.dynamic_weights_service import DynamicWeightsService
from app.services.fuel_cost import FuelCostService, FuelPriceService
from app.services.route_analysis_service import RouteAnalysisService
from app.services.route_optimizer import RouteOptimizer
from app.services.route_refinement_service import RouteRefinementService
from app.services.segment_alternative_service import SegmentAlternativeService
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
    osrm = None
    if settings.osrm_enabled:
        osrm = OsrmRoutingRepository(
            base_url=settings.osrm_base_url,
            timeout_seconds=settings.request_timeout_sec,
        )
    fallback = FallbackRoutingRepository()
    osrm_or_fallback = CompositeRoutingRepository(primary=osrm, fallback=fallback)
    ors = None
    if settings.openrouteservice_enabled and settings.openrouteservice_api_key:
        ors = OpenRouteServiceRoutingRepository(
            base_url=settings.openrouteservice_base_url,
            api_key=settings.openrouteservice_api_key,
            timeout_seconds=settings.request_timeout_sec,
        )
    ors_or_osrm = CompositeRoutingRepository(primary=ors, fallback=osrm_or_fallback)
    tomtom = None
    if settings.tomtom_enabled and settings.tomtom_routing_enabled and settings.effective_tomtom_routing_api_key:
        tomtom = TomTomRoutingRepository(
            base_url=settings.tomtom_base_url,
            api_key=settings.effective_tomtom_routing_api_key,
            timeout_seconds=settings.request_timeout_sec,
            matrix_max_cells=settings.tomtom_matrix_max_cells,
        )
    return CompositeRoutingRepository(primary=tomtom, fallback=ors_or_osrm)


@lru_cache
def get_weather_repository() -> CompositeWeatherRepository:
    settings = get_settings()
    primary = None
    if settings.weather_enabled:
        primary = MetNoWeatherRepository(
            base_url=settings.metno_base_url,
            timeout_seconds=settings.request_timeout_sec,
            user_agent=settings.external_api_user_agent,
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
def get_overpass_road_data_client() -> OverpassRoadDataClient:
    settings = get_settings()
    return OverpassRoadDataClient(
        base_url=settings.overpass_base_url,
        radius_m=settings.overpass_radius_m,
        timeout_seconds=settings.request_timeout_sec,
        cache_ttl_seconds=settings.overpass_cache_ttl_sec,
        user_agent=settings.external_api_user_agent,
    )


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
def get_geocoding_repository() -> CompositeGeocodingRepository:
    settings = get_settings()
    primary = (
        TomTomSearchRepository(
            base_url=settings.tomtom_base_url,
            api_key=settings.effective_tomtom_search_api_key,
            timeout_seconds=settings.request_timeout_sec,
        )
        if settings.tomtom_enabled and settings.tomtom_search_enabled and settings.effective_tomtom_search_api_key
        else None
    )
    fallback = DisabledGeocodingRepository()
    return CompositeGeocodingRepository(primary=primary, fallback=fallback)


@lru_cache
def get_traffic_repository() -> CompositeTrafficRepository:
    settings = get_settings()
    fallback = DisabledTrafficRepository()
    synthetic = SyntheticTrafficRepository() if settings.traffic_enabled and settings.synthetic_data_enabled else None
    synthetic_or_disabled = CompositeTrafficRepository(primary=synthetic, fallback=fallback)
    tomtom = (
        TomTomTrafficRepository(
            base_url=settings.tomtom_traffic_base_url,
            api_key=settings.effective_tomtom_traffic_api_key,
            timeout_seconds=settings.request_timeout_sec,
            max_concurrency=settings.tomtom_traffic_max_concurrency,
        )
        if settings.traffic_enabled and settings.tomtom_enabled and settings.effective_tomtom_traffic_api_key
        else None
    )
    tomtom_or_synthetic = CompositeTrafficRepository(primary=tomtom, fallback=synthetic_or_disabled)
    primary = (
        JsonTrafficRepository(settings.traffic_source_path)
        if settings.traffic_enabled and settings.traffic_source_path
        else None
    )
    return CompositeTrafficRepository(primary=primary, fallback=tomtom_or_synthetic)


@lru_cache
def get_road_quality_repository() -> CompositeRoadQualityRepository:
    settings = get_settings()
    fallback = FullQualityRoadQualityRepository()
    synthetic = SyntheticRoadQualityRepository() if settings.road_quality_enabled and settings.synthetic_data_enabled else None
    synthetic_or_full = CompositeRoadQualityRepository(primary=synthetic, fallback=fallback)
    overpass = (
        OverpassRoadQualityRepository(get_overpass_road_data_client())
        if settings.road_quality_enabled and settings.overpass_enabled
        else None
    )
    overpass_or_synthetic = CompositeRoadQualityRepository(primary=overpass, fallback=synthetic_or_full)
    primary = (
        JsonRoadQualityRepository(settings.road_quality_source_path)
        if settings.road_quality_enabled and settings.road_quality_source_path
        else None
    )
    return CompositeRoadQualityRepository(primary=primary, fallback=overpass_or_synthetic)


@lru_cache
def get_road_event_repository() -> CompositeRoadEventRepository:
    settings = get_settings()
    fallback = DisabledRoadEventRepository()
    synthetic = SyntheticRoadEventRepository() if settings.road_events_enabled and settings.synthetic_data_enabled else None
    synthetic_or_disabled = CompositeRoadEventRepository(primary=synthetic, fallback=fallback)
    overpass = (
        OverpassRoadEventRepository(get_overpass_road_data_client())
        if settings.road_events_enabled and settings.overpass_enabled
        else None
    )
    overpass_or_synthetic = CompositeRoadEventRepository(primary=overpass, fallback=synthetic_or_disabled)
    tomtom = (
        TomTomIncidentRepository(
            base_url=settings.tomtom_base_url,
            api_key=settings.effective_tomtom_incidents_api_key,
            timeout_seconds=settings.request_timeout_sec,
            bbox_padding_km=settings.tomtom_incidents_bbox_padding_km,
            match_radius_km=settings.tomtom_incidents_match_radius_km,
        )
        if settings.road_events_enabled
        and settings.tomtom_enabled
        and settings.tomtom_incidents_enabled
        and settings.effective_tomtom_incidents_api_key
        else None
    )
    tomtom_or_overpass = CompositeRoadEventRepository(primary=tomtom, fallback=overpass_or_synthetic)
    primary = (
        JsonRoadEventRepository(settings.road_events_source_path)
        if settings.road_events_enabled and settings.road_events_source_path
        else None
    )
    return CompositeRoadEventRepository(primary=primary, fallback=tomtom_or_overpass)


@lru_cache
def get_infrastructure_repository() -> CompositeInfrastructureRepository:
    settings = get_settings()
    fallback = UnrestrictedInfrastructureRepository()
    synthetic = SyntheticInfrastructureRepository() if settings.infrastructure_enabled and settings.synthetic_data_enabled else None
    synthetic_or_unrestricted = CompositeInfrastructureRepository(primary=synthetic, fallback=fallback)
    overpass = (
        OverpassInfrastructureRepository(get_overpass_road_data_client())
        if settings.infrastructure_enabled and settings.overpass_enabled
        else None
    )
    overpass_or_synthetic = CompositeInfrastructureRepository(primary=overpass, fallback=synthetic_or_unrestricted)
    primary = (
        JsonInfrastructureRepository(settings.infrastructure_source_path)
        if settings.infrastructure_enabled and settings.infrastructure_source_path
        else None
    )
    return CompositeInfrastructureRepository(primary=primary, fallback=overpass_or_synthetic)


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
    road_quality_repository: CompositeRoadQualityRepository = Depends(get_road_quality_repository),
    road_event_repository: CompositeRoadEventRepository = Depends(get_road_event_repository),
    infrastructure_repository: CompositeInfrastructureRepository = Depends(get_infrastructure_repository),
    terrain_profile_service: TerrainProfileService = Depends(get_terrain_profile_service),
) -> ContextService:
    return ContextService(
        routing_repository=routing_repository,
        weather_repository=weather_repository,
        elevation_repository=elevation_repository,
        traffic_repository=traffic_repository,
        toll_repository=toll_repository,
        road_quality_repository=road_quality_repository,
        road_event_repository=road_event_repository,
        infrastructure_repository=infrastructure_repository,
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


@lru_cache
def get_decision_explanation_service() -> DecisionExplanationService:
    return DecisionExplanationService()


def get_optimizer(
    criteria_service: CriteriaService = Depends(get_criteria_service),
    settings: Settings = Depends(get_settings),
) -> RouteOptimizer:
    return RouteOptimizer(
        criteria_service=criteria_service,
        parallel_workers=settings.ga_parallel_workers,
        parallel_min_batch_size=settings.ga_parallel_min_batch_size,
    )


def get_route_refinement_service(
    routing_repository: CompositeRoutingRepository = Depends(get_routing_repository),
    fuel_cost_service: FuelCostService = Depends(get_fuel_cost_service),
) -> RouteRefinementService:
    return RouteRefinementService(
        routing_repository=routing_repository,
        fuel_cost_service=fuel_cost_service,
    )


def get_segment_alternative_service(
    routing_repository: CompositeRoutingRepository = Depends(get_routing_repository),
    fuel_cost_service: FuelCostService = Depends(get_fuel_cost_service),
    settings: Settings = Depends(get_settings),
) -> SegmentAlternativeService:
    return SegmentAlternativeService(
        routing_repository=routing_repository,
        fuel_cost_service=fuel_cost_service,
        enabled=settings.segment_alternatives_enabled,
        max_candidates_per_edge=settings.segment_alternatives_max_candidates_per_edge,
        max_points=settings.segment_alternatives_max_points,
        max_concurrency=settings.segment_alternatives_max_concurrency,
        max_detour_ratio=settings.refinement_max_detour_ratio,
        cache_ttl_sec=settings.segment_alternative_cache_ttl_sec,
    )


def get_route_service(
    optimizer: RouteOptimizer = Depends(get_optimizer),
    routing_repository: CompositeRoutingRepository = Depends(get_routing_repository),
    cache_repository: InMemoryRouteCacheRepository = Depends(get_cache_repository),
    fuel_cost_service: FuelCostService = Depends(get_fuel_cost_service),
    context_service: ContextService = Depends(get_context_service),
    dynamic_weights_service: DynamicWeightsService = Depends(get_dynamic_weights_service),
    route_analysis_service: RouteAnalysisService = Depends(get_route_analysis_service),
    decision_explanation_service: DecisionExplanationService = Depends(get_decision_explanation_service),
    terrain_profile_service: TerrainProfileService = Depends(get_terrain_profile_service),
    run_repository: SqliteRouteRunRepository = Depends(get_run_repository),
    route_refinement_service: RouteRefinementService = Depends(get_route_refinement_service),
    segment_alternative_service: SegmentAlternativeService = Depends(get_segment_alternative_service),
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
        decision_explanation_service=decision_explanation_service,
        terrain_profile_service=terrain_profile_service,
        run_repository=run_repository,
        route_refinement_service=route_refinement_service,
        segment_alternative_service=segment_alternative_service,
        default_population=settings.ga_default_population,
        default_generations=settings.ga_default_generations,
        pareto_enabled=settings.optimizer_enable_pareto,
    )


def get_route_controller(
    service: RouteService = Depends(get_route_service),
) -> RouteController:
    return RouteController(route_service=service)
