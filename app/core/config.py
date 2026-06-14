from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


class Settings(BaseSettings):
    app_name: str = "Route Optimization Lab"
    external_api_user_agent: str = "RouteOptimizationLab/1.0 (contact: local)"
    cors_allowed_origins: str = ""
    osrm_base_url: str = "https://router.project-osrm.org"
    osrm_enabled: bool = True
    openrouteservice_enabled: bool = True
    openrouteservice_base_url: str = "https://api.openrouteservice.org"
    openrouteservice_api_key: str | None = None
    tomtom_enabled: bool = True
    tomtom_base_url: str = "https://api.tomtom.com"
    tomtom_api_key: str | None = None
    tomtom_routing_enabled: bool = True
    tomtom_routing_api_key: str | None = None
    tomtom_matrix_max_cells: int = 200
    tomtom_search_enabled: bool = True
    tomtom_search_api_key: str | None = None

    request_timeout_sec: float = 10.0
    cache_ttl_sec: int = 300

    fuel_price_source_url: str | None = "https://ssl.rosstat.gov.ru/storage/mediabank/126_20-08-2025.html"
    fuel_price_source_name: str = "Росстат"
    fuel_price_currency: str = "RUB"
    fuel_price_cache_ttl_sec: int = 3600
    fuel_price_fallback_petrol: float = 63.0
    fuel_price_fallback_diesel: float = 68.0
    weather_enabled: bool = True
    openmeteo_base_url: str = "https://api.open-meteo.com"
    metno_base_url: str = "https://api.met.no"
    elevation_enabled: bool = True
    opentopodata_base_url: str = "https://api.opentopodata.org"
    traffic_enabled: bool = True
    traffic_source_path: str | None = None
    tomtom_traffic_base_url: str = "https://api.tomtom.com"
    tomtom_traffic_api_key: str | None = None
    tomtom_traffic_max_concurrency: int = 8
    road_quality_enabled: bool = True
    road_quality_source_path: str | None = None
    road_events_enabled: bool = True
    road_events_source_path: str | None = None
    tomtom_incidents_enabled: bool = True
    tomtom_incidents_api_key: str | None = None
    tomtom_incidents_bbox_padding_km: float = 4.0
    tomtom_incidents_match_radius_km: float = 2.5
    infrastructure_enabled: bool = True
    infrastructure_source_path: str | None = None
    overpass_enabled: bool = True
    overpass_base_url: str = "https://overpass-api.de/api/interpreter"
    overpass_radius_m: float = 900.0
    overpass_cache_ttl_sec: int = 900
    synthetic_data_enabled: bool = False
    toll_enabled: bool = False
    toll_base_url: str = "https://apis.tollguru.com/toll"
    toll_api_key: str | None = None
    toll_max_concurrency: int = 8
    route_runs_db_path: str = "data/route_runs.db"
    ga_default_population: int = 96
    ga_default_generations: int = 120
    ga_parallel_workers: int | None = 1
    ga_parallel_min_batch_size: int = 48
    optimizer_enable_pareto: bool = True
    segment_alternatives_enabled: bool = False
    segment_alternatives_max_candidates_per_edge: int = 2
    segment_alternatives_max_points: int = 6
    segment_alternatives_max_concurrency: int = 8
    refinement_max_detour_ratio: float = 1.15
    refinement_min_gain_pct: float = 0.05
    segment_alternative_cache_ttl_sec: int = 300

    model_config = SettingsConfigDict(env_prefix="APP_", case_sensitive=False)

    @property
    def effective_tomtom_routing_api_key(self) -> str | None:
        return self.tomtom_routing_api_key or self.tomtom_api_key

    @property
    def effective_tomtom_search_api_key(self) -> str | None:
        return self.tomtom_search_api_key or self.tomtom_api_key

    @property
    def effective_tomtom_traffic_api_key(self) -> str | None:
        return self.tomtom_traffic_api_key or self.tomtom_api_key

    @property
    def effective_tomtom_incidents_api_key(self) -> str | None:
        return self.tomtom_incidents_api_key or self.tomtom_api_key

    @property
    def parsed_cors_allowed_origins(self) -> list[str]:
        return [
            origin.strip().rstrip("/")
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]
