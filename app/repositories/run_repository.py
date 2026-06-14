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

    def export_pdf(self, run_id: str) -> bytes | None:
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
                "operational_cost",
                "driver_cost",
                "maintenance_cost",
                "cargo_risk",
                "cargo_expected_loss",
                "co2_kg",
                "congestion_index",
                "weather_risk",
                "reliability_score",
                "safety_risk",
                "road_quality_risk",
                "incident_risk",
                "roadwork_risk",
                "dynamic_event_risk",
                "objective_score",
                "constraint_penalty",
                "infrastructure_penalty",
                "temporal_restriction_penalty",
                "deadline_penalty",
                "capacity_penalty",
                "vehicle_routes_used",
                "max_route_load_t",
                "capacity_utilization",
                "capacity_feasible",
                "violated_constraints",
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

    def export_pdf(self, run_id: str) -> bytes | None:
        details = self.get_run(run_id)
        if details is None:
            return None
        return self._build_pdf_report(details)

    @staticmethod
    def _metrics_row(run_id: str, kind: str, rank: int, metrics: dict) -> list[str]:
        return [
            run_id,
            kind,
            str(rank),
            str(metrics.get("distance_km", "")),
            str(metrics.get("duration_min", "")),
            str(metrics.get("fuel_cost", "")),
            str(metrics.get("operational_cost", "")),
            str(metrics.get("driver_cost", "")),
            str(metrics.get("maintenance_cost", "")),
            str(metrics.get("cargo_risk", "")),
            str(metrics.get("cargo_expected_loss", "")),
            str(metrics.get("co2_kg", "")),
            str(metrics.get("congestion_index", "")),
            str(metrics.get("weather_risk", "")),
            str(metrics.get("reliability_score", "")),
            str(metrics.get("safety_risk", "")),
            str(metrics.get("road_quality_risk", "")),
            str(metrics.get("incident_risk", "")),
            str(metrics.get("roadwork_risk", "")),
            str(metrics.get("dynamic_event_risk", "")),
            str(metrics.get("objective_score", "")),
            str(metrics.get("constraint_penalty", "")),
            str(metrics.get("infrastructure_penalty", "")),
            str(metrics.get("temporal_restriction_penalty", "")),
            str(metrics.get("deadline_penalty", "")),
            str(metrics.get("capacity_penalty", "")),
            str(metrics.get("vehicle_routes_used", "")),
            str(metrics.get("max_route_load_t", "")),
            str(metrics.get("capacity_utilization", "")),
            str(metrics.get("capacity_feasible", "")),
            "|".join(metrics.get("violated_constraints") or []),
            str(metrics.get("feasible", "")),
        ]

    @staticmethod
    def _build_pdf_report(details: RouteRunDetails) -> bytes:
        response = details.response
        request = details.request
        metrics = response.get("metrics") or {}
        comparison = response.get("comparison") or {}
        explanation = response.get("decision_explanation") or {}
        data_sources = response.get("data_sources") or {}
        data_confidence = response.get("data_confidence") or {}
        diagnostics = response.get("diagnostics") or {}
        route_quality = response.get("route_quality_index") or {}
        stress_test = response.get("stress_test") or {}
        segment_insights = response.get("segment_insights") or []
        segment_factors = response.get("segment_factors") or []
        constraint_health = response.get("constraint_health") or {}
        analysis_matrices = response.get("analysis_matrices") or {}
        cvrp_plan = response.get("cvrp_plan") or {}
        alternatives = response.get("alternatives") or []
        ordered_points = response.get("ordered_points") or []
        baseline_points = comparison.get("baseline_ordered_points") or []
        geometry = response.get("geometry") or []

        def fmt(value: object, digits: int = 2) -> str:
            if isinstance(value, (int, float)):
                return f"{float(value):.{digits}f}"
            if value is None:
                return "-"
            return str(value)

        def point_labels(points: list[dict]) -> str:
            labels = [str(point.get("label") or f"P{idx + 1}") for idx, point in enumerate(points)]
            return " -> ".join(labels) if labels else "-"

        def geometry_bounds(points: list[list[float]]) -> str:
            coordinates = [point for point in points if isinstance(point, list) and len(point) >= 2]
            if not coordinates:
                return "-"
            lons = [float(point[0]) for point in coordinates]
            lats = [float(point[1]) for point in coordinates]
            return f"lat {min(lats):.5f}..{max(lats):.5f}; lon {min(lons):.5f}..{max(lons):.5f}"

        def matrix_shape(matrix: object) -> str:
            if not isinstance(matrix, list):
                return "-"
            rows = len(matrix)
            cols = len(matrix[0]) if rows and isinstance(matrix[0], list) else 0
            return f"{rows}x{cols}"

        problematic_insights = [
            insight
            for insight in segment_insights
            if insight.get("is_problematic") or float(insight.get("severity_score") or 0.0) >= 0.35
        ]

        lines = [
            "Route optimization analytical report",
            f"Run ID: {details.run_id}",
            f"Created at: {details.created_at}",
            f"Provider: {response.get('provider') or '-'}",
            "",
            "1. Route map summary",
            f"Geometry points: {len(geometry)}",
            f"Geometry bounds: {geometry_bounds(geometry)}",
            f"GeoJSON type: {(response.get('geojson') or {}).get('type', '-')}",
            f"Selected order: {point_labels(ordered_points)}",
            f"Baseline order: {point_labels(baseline_points)}",
            f"Segment factors: {len(segment_factors)}",
            f"Problem map overlays: {len(problematic_insights)}",
            "",
            "2. Main metrics",
            f"Distance, km: {fmt(metrics.get('distance_km'))}",
            f"Duration, min: {fmt(metrics.get('duration_min'))}",
            f"Fuel cost: {fmt(metrics.get('fuel_cost'))}",
            f"Operational cost: {fmt(metrics.get('operational_cost'))}",
            f"CO2, kg: {fmt(metrics.get('co2_kg'))}",
            f"Cargo risk: {fmt(metrics.get('cargo_risk'), 4)}",
            f"Safety risk: {fmt(metrics.get('safety_risk'), 4)}",
            f"Reliability score: {fmt(metrics.get('reliability_score'), 4)}",
            f"Constraint penalty: {fmt(metrics.get('constraint_penalty'))}",
            f"Feasible: {fmt(metrics.get('feasible'))}",
            f"CVRP enabled: {fmt(cvrp_plan.get('enabled'))}",
            f"CVRP routes used: {fmt(cvrp_plan.get('routes_used'), 0)}",
            f"Capacity feasible: {fmt(cvrp_plan.get('feasible'))}",
            "",
            "3. Baseline comparison",
        ]
        improvement = comparison.get("improvement_pct") or {}
        delta = comparison.get("delta") or {}
        for key in ["distance_km", "duration_min", "fuel_cost", "operational_cost", "cargo_risk", "co2_kg", "objective_score"]:
            lines.append(f"{key}: delta={fmt(delta.get(key))}; improvement={fmt(improvement.get(key))}%")

        lines.extend(
            [
                "",
                "4. Problem segments and map overlays",
                f"Problematic segments: {len(problematic_insights)}",
            ]
        )
        for insight in problematic_insights[:12]:
            constraints = ", ".join(insight.get("violated_constraints") or []) or "-"
            lines.append(
                f"{insight.get('start_label', '?')} -> {insight.get('end_label', '?')}: "
                f"{insight.get('dominant_factor_label') or '-'}; "
                f"severity={fmt(insight.get('severity_score'), 3)} ({insight.get('severity_level') or '-'}); "
                f"color={insight.get('map_color_hex') or insight.get('color_hex') or '-'}; "
                f"stroke={fmt(insight.get('map_stroke_weight'), 0)}; "
                f"dash={insight.get('map_dash_array') or 'solid'}; "
                f"constraints={constraints}"
            )
        if not problematic_insights:
            lines.append("No problematic segment overlays.")

        lines.extend(
            [
                "",
                "5. Constraint health",
                f"Overall status: {constraint_health.get('overall_status') or '-'}",
            ]
        )
        for item in (constraint_health.get("items") or [])[:12]:
            lines.append(
                f"{item.get('label') or item.get('key') or '-'}: "
                f"status={item.get('status') or '-'}; "
                f"value={fmt(item.get('value'))}; limit={fmt(item.get('limit'))}; "
                f"margin={fmt(item.get('margin_pct'))}%"
            )

        lines.extend(
            [
                "",
                "6. Analysis matrices",
                f"Distance matrix: {matrix_shape(analysis_matrices.get('distance_km'))}",
                f"Duration matrix: {matrix_shape(analysis_matrices.get('duration_min'))}",
                f"Traffic matrix: {matrix_shape(analysis_matrices.get('traffic_index'))}",
                f"Road quality matrix: {matrix_shape(analysis_matrices.get('road_quality'))}",
                f"Incident matrix: {matrix_shape(analysis_matrices.get('incident_risk'))}",
                f"Roadwork matrix: {matrix_shape(analysis_matrices.get('roadwork_risk'))}",
                "",
                "7. Choice explanation",
                f"Main reason: {explanation.get('main_reason') or '-'}",
                f"Selected from: {explanation.get('selected_from') or '-'}",
                f"Strategy: {explanation.get('strategy') or '-'}",
                f"Compromise accepted: {fmt(explanation.get('compromise_accepted'))}",
                "Positive factors:",
            ]
        )
        lines.extend([f"- {item}" for item in explanation.get("top_positive_factors") or ["-"]])
        lines.append("Negative factors:")
        lines.extend([f"- {item}" for item in explanation.get("top_negative_factors") or ["-"]])
        lines.append("Rejected reasons:")
        lines.extend([f"- {item}" for item in explanation.get("rejected_reasons") or ["-"]])

        lines.extend(
            [
                "",
                "8. Alternatives",
                f"Alternatives returned: {len(alternatives)}",
            ]
        )
        for alternative in alternatives[:8]:
            alt_metrics = alternative.get("metrics") or {}
            lines.append(
                "Rank "
                f"{alternative.get('rank')}: distance={fmt(alt_metrics.get('distance_km'))} km; "
                f"duration={fmt(alt_metrics.get('duration_min'))} min; "
                f"score={fmt(alt_metrics.get('objective_score'), 4)}; "
                f"feasible={fmt(alt_metrics.get('feasible'))}"
            )

        lines.extend(
            [
                "",
                "9. Sensitivity and quality",
                f"Route Quality Index: {fmt(route_quality.get('score'), 1)} ({route_quality.get('label') or '-'})",
                f"Stress resilience: {fmt(stress_test.get('resilience_index'), 4)}",
                f"On-time probability: {fmt(stress_test.get('on_time_probability'), 4)}",
                f"Within-budget probability: {fmt(stress_test.get('within_budget_probability'), 4)}",
                f"Within-safety probability: {fmt(stress_test.get('within_safety_probability'), 4)}",
                "",
                "10. Data sources",
            ]
        )
        for key, value in data_sources.items():
            lines.append(f"{key}: {value}")
        lines.extend(
            [
                f"Data Confidence Score: {fmt(data_confidence.get('score'), 1)} ({data_confidence.get('label') or '-'})",
                f"Fallback sources: {', '.join(data_confidence.get('fallback_sources') or []) or '-'}",
                "",
                "11. Diagnostics",
                f"Optimization mode: {diagnostics.get('mode') or request.get('optimize_mode') or '-'}",
                f"Optimization active: {fmt(diagnostics.get('optimization_active'))}",
                f"Optimizer reason: {diagnostics.get('optimization_reason') or '-'}",
                f"Selected candidate reason: {diagnostics.get('final_selection_reason') or '-'}",
                f"Baseline guard applied: {fmt(diagnostics.get('baseline_guard_applied'))}",
                f"Baseline guard reason: {diagnostics.get('baseline_guard_reason') or '-'}",
                f"Refinement reason: {diagnostics.get('route_refinement_reason') or '-'}",
                f"Generations: {fmt(diagnostics.get('generations'), 0)}",
                f"Population size: {fmt(diagnostics.get('population_size'), 0)}",
                f"Evaluated solutions: {fmt(diagnostics.get('evaluated_solutions'), 0)}",
            ]
        )
        timings = diagnostics.get("performance_timings") or {}
        if timings:
            lines.extend(
                [
                    "Performance timings, ms:",
                    f"context={fmt(timings.get('context_ms'), 3)}",
                    f"optimization={fmt(timings.get('optimization_ms'), 3)}",
                    f"refinement={fmt(timings.get('refinement_ms'), 3)}",
                    f"analysis={fmt(timings.get('analysis_ms'), 3)}",
                    f"total={fmt(timings.get('total_ms'), 3)}",
                ]
            )
        return SqliteRouteRunRepository._simple_pdf(lines)

    @staticmethod
    def _simple_pdf(lines: list[str]) -> bytes:
        def escape_pdf_text(value: str) -> str:
            return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        page_lines = [lines[index : index + 54] for index in range(0, len(lines), 54)] or [[]]
        objects: dict[int, bytes] = {}
        page_ids: list[int] = []
        objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        next_id = 4
        for page in page_lines:
            content_id = next_id
            page_id = next_id + 1
            next_id += 2
            page_ids.append(page_id)
            text_commands = ["BT", "/F1 9 Tf", "36 806 Td", "12 TL"]
            for line in page:
                safe_line = escape_pdf_text(line[:110])
                text_commands.append(f"({safe_line}) Tj")
                text_commands.append("T*")
            text_commands.append("ET")
            stream = "\n".join(text_commands).encode("latin-1", errors="replace")
            objects[content_id] = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
            objects[page_id] = (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                b"/Resources << /Font << /F1 3 0 R >> >> "
                + b"/Contents "
                + str(content_id).encode("ascii")
                + b" 0 R >>"
            )
        kids = b" ".join(str(page_id).encode("ascii") + b" 0 R" for page_id in page_ids)
        objects[2] = b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode("ascii") + b" >>"

        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {0: 0}
        for object_id in sorted(objects):
            offsets[object_id] = len(output)
            output.extend(str(object_id).encode("ascii") + b" 0 obj\n")
            output.extend(objects[object_id])
            output.extend(b"\nendobj\n")
        xref_offset = len(output)
        max_id = max(objects)
        output.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for object_id in range(1, max_id + 1):
            output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))
        output.extend(
            f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        )
        return bytes(output)
