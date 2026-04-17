from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.controllers.route_controller import RouteController
from app.core.di import get_route_controller
from app.domain.models import RouteRequest, RouteResponse, RouteRunDetails, RouteRunListItem

router = APIRouter()


@router.post("/api/routes", response_model=RouteResponse)
async def build_route(
    payload: RouteRequest,
    controller: RouteController = Depends(get_route_controller),
) -> RouteResponse:
    return await controller.build_route(payload)


@router.get("/api/runs", response_model=list[RouteRunListItem])
async def list_runs(
    limit: int = Query(default=20, ge=1, le=200),
    controller: RouteController = Depends(get_route_controller),
) -> list[RouteRunListItem]:
    return controller.list_runs(limit=limit)


@router.get("/api/runs/{run_id}", response_model=RouteRunDetails)
async def get_run(
    run_id: str,
    controller: RouteController = Depends(get_route_controller),
) -> RouteRunDetails:
    run = controller.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@router.get("/api/runs/{run_id}/report.csv", response_class=PlainTextResponse)
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
