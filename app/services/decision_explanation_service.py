from __future__ import annotations

from app.domain.models import DecisionExplanation, RouteAlternative, RouteComparisonInfo, RouteRequest


class DecisionExplanationService:
    def build(
        self,
        *,
        request: RouteRequest,
        comparison: RouteComparisonInfo,
        alternatives: list[RouteAlternative],
        diagnostics,
    ) -> DecisionExplanation:
        positives: list[str] = []
        negatives: list[str] = []
        metric_labels = {
            "distance_km": "distance",
            "duration_min": "travel time",
            "fuel_cost": "fuel and toll cost",
            "operational_cost": "operational cost",
            "co2_kg": "CO2 emissions",
            "objective_score": "weighted score",
        }
        for key, label in metric_labels.items():
            improvement = float(getattr(comparison.improvement_pct, key))
            delta = float(getattr(comparison.delta, key))
            if improvement > 0.05:
                positives.append(f"Reduced {label} by {improvement:.1f}%")
            elif delta > 1e-9 and key != "objective_score":
                baseline_value = getattr(comparison.baseline_metrics, key)
                negatives.append(f"Increased {label} by {self._relative_delta_pct(baseline_value, delta):.1f}%")

        selected_from = getattr(diagnostics, "final_selected_from", None) if diagnostics is not None else None
        accepted_tradeoff = bool(getattr(diagnostics, "accepted_tradeoff", False)) if diagnostics is not None else False
        if getattr(diagnostics, "baseline_guard_applied", False):
            main_reason = "Baseline route kept because candidates regressed key metrics"
        elif accepted_tradeoff:
            main_reason = "Controlled tradeoff accepted under optimization strategy"
        elif positives:
            main_reason = positives[0]
        else:
            main_reason = "Selected route has the best feasible distance, time and cost score"

        influential = [
            component.label
            for component in sorted(
                comparison.optimized_score.components,
                key=lambda item: item.contribution,
                reverse=True,
            )
            if component.weight > 0
        ][:5]

        rejected_reasons = list(getattr(diagnostics, "rejected_alternative_reasons", []) or [])
        rejected_metrics = list(getattr(diagnostics, "rejected_regression_metrics", []) or [])
        if rejected_metrics and not accepted_tradeoff:
            rejected_reasons.append(f"Rejected regressions: {', '.join(rejected_metrics)}")
        for alternative in alternatives:
            if not alternative.metrics.feasible:
                rejected_reasons.append(f"Alternative #{alternative.rank} rejected because constraints were violated")
                break

        constraints_influence: list[str] = []
        if comparison.optimized_metrics.constraint_penalty > 0:
            constraints_influence.append(
                f"Constraint penalty applied: {comparison.optimized_metrics.constraint_penalty:.1f}",
            )
        if comparison.optimized_metrics.violated_constraints:
            constraints_influence.extend(comparison.optimized_metrics.violated_constraints[:5])

        return DecisionExplanation(
            main_reason=main_reason,
            top_positive_factors=positives[:5],
            top_negative_factors=negatives[:5],
            rejected_reasons=rejected_reasons[:8],
            influential_criteria=influential,
            constraints_influence=constraints_influence,
            selected_from=selected_from,
            strategy=request.optimization_strategy,
            compromise_accepted=accepted_tradeoff,
        )

    @staticmethod
    def _relative_delta_pct(baseline: float, delta: float) -> float:
        return (delta / max(abs(float(baseline)), 1e-9)) * 100.0
