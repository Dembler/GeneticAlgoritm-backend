from __future__ import annotations

from app.domain.models import RouteRequest, RouteResponse, RouteRunDetails, RouteRunListItem
from app.services.route_service import RouteService


class RouteController:
    def __init__(self, route_service: RouteService) -> None:
        self._route_service = route_service

    async def build_route(self, payload: RouteRequest) -> RouteResponse:
        return await self._route_service.compute_route(payload)

    def list_runs(self, limit: int = 20) -> list[RouteRunListItem]:
        return self._route_service.list_runs(limit=limit)

    def get_run(self, run_id: str) -> RouteRunDetails | None:
        return self._route_service.get_run(run_id)

    def export_run_csv(self, run_id: str) -> str | None:
        return self._route_service.export_run_csv(run_id)
