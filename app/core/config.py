from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Route Optimization Lab"
    osrm_base_url: str = "https://router.project-osrm.org"
    osrm_enabled: bool = True

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
    toll_enabled: bool = False
    toll_base_url: str = "https://apis.tollguru.com/toll"
    toll_api_key: str | None = None
    toll_max_concurrency: int = 8
    route_runs_db_path: str = "data/route_runs.db"
    ga_default_population: int = 96
    ga_default_generations: int = 120
    optimizer_enable_pareto: bool = True

    model_config = SettingsConfigDict(env_prefix="APP_", case_sensitive=False)
