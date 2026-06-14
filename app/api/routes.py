from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response

from app.controllers.route_controller import RouteController
from app.core.config import Settings
from app.core.di import get_geocoding_repository, get_route_controller, get_settings
from app.domain.models import GeocodingResultDto, RouteRequest, RouteResponse, RouteRunDetails, RouteRunListItem
from app.repositories.geocoding_repository import GeocodingRepository, GeocodingResult

router = APIRouter()


@router.get("/api/health")
@router.get("/api/v1/health")
@router.get("/api/v2/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": "demo",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/integrations/status")
@router.get("/api/v1/integrations/status")
@router.get("/api/v2/integrations/status")
async def integrations_status(settings: Settings = Depends(get_settings)) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    tomtom_enabled = settings.tomtom_enabled and bool(settings.effective_tomtom_routing_api_key)
    ors_enabled = settings.openrouteservice_enabled and bool(settings.openrouteservice_api_key)

    return [
        {
            "name": "routing",
            "enabled": tomtom_enabled or ors_enabled or settings.osrm_enabled,
            "status": "connected" if tomtom_enabled or ors_enabled or settings.osrm_enabled else "disabled",
            "source": "tomtom-routing" if tomtom_enabled else "openrouteservice" if ors_enabled else "osrm" if settings.osrm_enabled else "disabled",
            "fallback": "osrm" if settings.osrm_enabled and (tomtom_enabled or ors_enabled) else "haversine fallback",
            "last_check_at": now,
        },
        {
            "name": "geocoding",
            "enabled": settings.tomtom_search_enabled and bool(settings.effective_tomtom_search_api_key),
            "status": "connected" if settings.tomtom_search_enabled and bool(settings.effective_tomtom_search_api_key) else "disabled",
            "source": "tomtom-search" if settings.tomtom_search_enabled and bool(settings.effective_tomtom_search_api_key) else "disabled",
            "fallback": "disabled",
            "last_check_at": now,
        },
        {
            "name": "traffic",
            "enabled": settings.traffic_enabled and bool(settings.effective_tomtom_traffic_api_key),
            "status": "connected" if settings.traffic_enabled and bool(settings.effective_tomtom_traffic_api_key) else "disabled",
            "source": "tomtom-traffic" if settings.traffic_enabled and bool(settings.effective_tomtom_traffic_api_key) else "disabled",
            "fallback": "synthetic" if settings.synthetic_data_enabled else "disabled",
            "last_check_at": now,
        },
        {
            "name": "weather",
            "enabled": settings.weather_enabled,
            "status": "connected" if settings.weather_enabled else "disabled",
            "source": "met.no",
            "fallback": "open-meteo",
            "last_check_at": now,
        },
        {
            "name": "elevation",
            "enabled": settings.elevation_enabled,
            "status": "connected" if settings.elevation_enabled else "disabled",
            "source": "opentopodata",
            "fallback": "open-meteo elevation",
            "last_check_at": now,
        },
        {
            "name": "osm",
            "enabled": settings.overpass_enabled,
            "status": "connected" if settings.overpass_enabled else "disabled",
            "source": "overpass",
            "fallback": "synthetic" if settings.synthetic_data_enabled else "disabled",
            "last_check_at": now,
        },
        {
            "name": "tolls",
            "enabled": settings.toll_enabled and bool(settings.toll_api_key),
            "status": "connected" if settings.toll_enabled and bool(settings.toll_api_key) else "disabled",
            "source": "tollguru" if settings.toll_enabled and bool(settings.toll_api_key) else "disabled",
            "fallback": "disabled",
            "last_check_at": now,
        },
    ]


@router.post("/api/routes", response_model=RouteResponse)
@router.post("/api/v1/routes", response_model=RouteResponse)
@router.post("/api/v2/routes", response_model=RouteResponse)
async def build_route(
    payload: RouteRequest,
    controller: RouteController = Depends(get_route_controller),
) -> RouteResponse:
    return await controller.build_route(payload)


@router.get("/api/geocode", response_model=list[GeocodingResultDto])
@router.get("/api/v1/geocode", response_model=list[GeocodingResultDto])
@router.get("/api/v2/geocode", response_model=list[GeocodingResultDto])
async def geocode(
    q: str = Query(..., min_length=1, max_length=256),
    limit: int = Query(default=5, ge=1, le=100),
    country_set: str | None = Query(default=None, max_length=128),
    lat: float | None = Query(default=None, ge=-90, le=90),
    lon: float | None = Query(default=None, ge=-180, le=180),
    radius_m: int | None = Query(default=None, ge=1, le=100000),
    language: str | None = Query(default=None, max_length=16),
    geocoding_repository: GeocodingRepository = Depends(get_geocoding_repository),
) -> list[GeocodingResultDto]:
    results = await geocoding_repository.search(
        q,
        limit=limit,
        country_set=country_set,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        language=language,
    )
    return [_geocoding_result_to_dto(result) for result in results]


@router.get("/api/reverse-geocode", response_model=GeocodingResultDto)
@router.get("/api/v1/reverse-geocode", response_model=GeocodingResultDto)
@router.get("/api/v2/reverse-geocode", response_model=GeocodingResultDto)
async def reverse_geocode(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    language: str | None = Query(default=None, max_length=16),
    geocoding_repository: GeocodingRepository = Depends(get_geocoding_repository),
) -> GeocodingResultDto:
    result = await geocoding_repository.reverse(lat, lon, language=language)
    return _geocoding_result_to_dto(result)


@router.get("/api/runs", response_model=list[RouteRunListItem])
@router.get("/api/v1/runs", response_model=list[RouteRunListItem])
@router.get("/api/v2/runs", response_model=list[RouteRunListItem])
async def list_runs(
    limit: int = Query(default=20, ge=1, le=200),
    controller: RouteController = Depends(get_route_controller),
) -> list[RouteRunListItem]:
    return controller.list_runs(limit=limit)


@router.get("/api/runs/{run_id}", response_model=RouteRunDetails)
@router.get("/api/v1/runs/{run_id}", response_model=RouteRunDetails)
@router.get("/api/v2/runs/{run_id}", response_model=RouteRunDetails)
async def get_run(
    run_id: str,
    controller: RouteController = Depends(get_route_controller),
) -> RouteRunDetails:
    run = controller.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@router.get("/api/runs/{run_id}/report.csv", response_class=PlainTextResponse)
@router.get("/api/v1/runs/{run_id}/report.csv", response_class=PlainTextResponse)
@router.get("/api/v2/runs/{run_id}/report.csv", response_class=PlainTextResponse)
async def export_run_csv(
    run_id: str,
    controller: RouteController = Depends(get_route_controller),
) -> PlainTextResponse:
    report = controller.export_run_csv(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    headers = {
        "Content-Disposition": f'attachment; filename="route_run_{run_id}.csv"',
    }
    return PlainTextResponse(content=report, media_type="text/csv; charset=utf-8", headers=headers)


@router.get("/api/runs/{run_id}/report.pdf", response_class=Response)
@router.get("/api/v1/runs/{run_id}/report.pdf", response_class=Response)
@router.get("/api/v2/runs/{run_id}/report.pdf", response_class=Response)
async def export_run_pdf(
    run_id: str,
    controller: RouteController = Depends(get_route_controller),
) -> Response:
    report = controller.export_run_pdf(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    headers = {
        "Content-Disposition": f'attachment; filename="route_run_{run_id}.pdf"',
    }
    return Response(content=report, media_type="application/pdf", headers=headers)


@router.get("/api/schema")
@router.get("/api/v1/schema")
@router.get("/api/v2/schema")
async def export_openapi_schema(request: Request) -> dict:
    schema = request.app.openapi()
    schema.setdefault("x-api-versions", ["v1", "v2"])
    schema.setdefault("x-default-api-version", "v1")
    return schema


def _geocoding_result_to_dto(result: GeocodingResult) -> GeocodingResultDto:
    return GeocodingResultDto(
        point=result.point,
        address=result.address,
        score=result.score,
        entity_type=result.entity_type,
        provider=result.provider,
    )
