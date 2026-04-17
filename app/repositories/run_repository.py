from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
from typing import Protocol
from uuid import uuid4

from app.domain.models import RouteRequest, RouteResponse, RouteRunDetails, RouteRunListItem


class RouteRunRepository(Protocol):
    def save(self, request: RouteRequest, response: RouteResponse) -> str:
        raise NotImplementedError

    def list_runs(self, limit: int = 20) -> list[RouteRunListItem]:
        raise NotImplementedError

    def get_run(self, run_id: str) -> RouteRunDetails | None:
        raise NotImplementedError

    def export_csv(self, run_id: str) -> str | None:
        raise NotImplementedError


class SqliteRouteRunRepository(RouteRunRepository):
    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS route_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    provider_summary TEXT NOT NULL,
                    score REAL,
                    feasible_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS route_alternatives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    ordered_points_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES route_runs(run_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_route_runs_created_at ON route_runs(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_route_alternatives_run_id ON route_alternatives(run_id)")
            conn.commit()

    def save(self, request: RouteRequest, response: RouteResponse) -> str:
        run_id = response.run_id or str(uuid4())
        response_with_id = response.model_copy(update={"run_id": run_id})
        created_at = datetime.now(timezone.utc).isoformat()

        alternatives = response_with_id.alternatives or []
        feasible_count = sum(1 for alt in alternatives if alt.metrics.feasible)
        if response_with_id.metrics and response_with_id.metrics.feasible:
            feasible_count += 1

        score = response_with_id.metrics.objective_score if response_with_id.metrics else None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO route_runs (
                    run_id, created_at, request_json, response_json, provider_summary, score, feasible_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    json.dumps(request.model_dump(mode="json"), ensure_ascii=False),
                    json.dumps(response_with_id.model_dump(mode="json"), ensure_ascii=False),
                    response_with_id.provider,
                    score,
                    feasible_count,
                ),
            )
            for alt in alternatives:
                conn.execute(
                    """
                    INSERT INTO route_alternatives (run_id, rank, metrics_json, ordered_points_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        alt.rank,
                        json.dumps(alt.metrics.model_dump(mode="json"), ensure_ascii=False),
                        json.dumps([p.model_dump(mode="json") for p in alt.ordered_points], ensure_ascii=False),
                    ),
                )
            conn.commit()
        return run_id

    def list_runs(self, limit: int = 20) -> list[RouteRunListItem]:
        safe_limit = max(1, min(200, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, created_at, provider_summary, score, feasible_count
                FROM route_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            RouteRunListItem(
                run_id=row["run_id"],
                created_at=row["created_at"],
                provider_summary=row["provider_summary"],
                objective_score=row["score"],
                feasible_count=int(row["feasible_count"] or 0),
            )
            for row in rows
        ]

    def get_run(self, run_id: str) -> RouteRunDetails | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, created_at, request_json, response_json
                FROM route_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        request_payload = json.loads(row["request_json"])
        response_payload = json.loads(row["response_json"])
        return RouteRunDetails(
            run_id=row["run_id"],
            created_at=row["created_at"],
            request=request_payload,
            response=response_payload,
        )

    def export_csv(self, run_id: str) -> str | None:
        details = self.get_run(run_id)
        if details is None:
            return None

        response = details.response
        alternatives = response.get("alternatives") or []
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "run_id",
                "kind",
                "rank",
                "distance_km",
                "duration_min",
                "fuel_cost",
                "co2_kg",
                "congestion_index",
                "weather_risk",
                "reliability_score",
                "safety_risk",
                "objective_score",
                "constraint_penalty",
                "feasible",
            ]
        )
        main_metrics = response.get("metrics") or {}
        writer.writerow(
            self._metrics_row(
                run_id=run_id,
                kind="best",
                rank=1,
                metrics=main_metrics,
            )
        )
        for idx, alt in enumerate(alternatives, start=1):
            writer.writerow(
                self._metrics_row(
                    run_id=run_id,
                    kind="pareto",
                    rank=int(alt.get("rank") or idx),
                    metrics=alt.get("metrics") or {},
                )
            )
        return output.getvalue()

    @staticmethod
    def _metrics_row(run_id: str, kind: str, rank: int, metrics: dict) -> list[str]:
        return [
            run_id,
            kind,
            str(rank),
            str(metrics.get("distance_km", "")),
            str(metrics.get("duration_min", "")),
            str(metrics.get("fuel_cost", "")),
            str(metrics.get("co2_kg", "")),
            str(metrics.get("congestion_index", "")),
            str(metrics.get("weather_risk", "")),
            str(metrics.get("reliability_score", "")),
            str(metrics.get("safety_risk", "")),
            str(metrics.get("objective_score", "")),
            str(metrics.get("constraint_penalty", "")),
            str(metrics.get("feasible", "")),
        ]
