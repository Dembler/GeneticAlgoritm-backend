from __future__ import annotations

import random
from statistics import mean

from app.domain.models import (
    Point,
    RouteMetrics,
    RouteRequest,
    RouteSegmentFactor,
    RouteSegmentInsight,
    RouteStressTest,
    StressTestHighlight,
)


class RouteAnalysisService:
    _FACTOR_META = {
        "weather": {
            "label": "Погода",
            "color": "#3d8bfd",
            "note": "Погодная волатильность сильнее всего раскачивает время и устойчивость этого маршрута.",
        },
        "congestion": {
            "label": "Трафик",
            "color": "#f08a24",
            "note": "Главный риск связан с пробками и ростом задержек на перегруженных участках.",
        },
        "elevation": {
            "label": "Рельеф",
            "color": "#8b5cf6",
            "note": "Набор высоты сильнее всего влияет на экономичность и чувствительность сегмента.",
        },
        "tolls": {
            "label": "Платные дороги",
            "color": "#ef5f7a",
            "note": "Финансовая чувствительность маршрута в основном создаётся платными участками.",
        },
        "safety": {
            "label": "Безопасность",
            "color": "#dc2626",
            "note": "Участок в первую очередь уязвим по рискам безопасности и требует осторожного режима.",
        },
        "reliability": {
            "label": "Надёжность",
            "color": "#0891b2",
            "note": "Основной риск связан с непредсказуемостью времени прохождения сегмента.",
        },
    }

    def build_segment_insights(
        self,
        points: list[Point],
        segment_factors: list[RouteSegmentFactor],
    ) -> list[RouteSegmentInsight]:
        insights: list[RouteSegmentInsight] = []
        for index, segment in enumerate(segment_factors):
            start_idx = min(segment.start_index, len(points) - 1)
            end_idx = min(segment.end_index, len(points) - 1)
            start_label = points[start_idx].label or f"Точка {start_idx + 1}"
            end_label = points[end_idx].label or f"Точка {end_idx + 1}"
            factor_scores = self._factor_scores(segment)
            dominant_key, dominant_score = max(factor_scores.items(), key=lambda item: item[1])
            meta = self._FACTOR_META[dominant_key]
            insights.append(
                RouteSegmentInsight(
                    start_index=segment.start_index,
                    end_index=segment.end_index,
                    start_label=start_label,
                    end_label=end_label,
                    dominant_factor_key=dominant_key,
                    dominant_factor_label=meta["label"],
                    severity_score=dominant_score,
                    severity_level=self._severity_level(dominant_score),
                    color_hex=meta["color"],
                    narrative=self._build_segment_narrative(
                        start_label=start_label,
                        end_label=end_label,
                        factor_key=dominant_key,
                        segment=segment,
                    ),
                    distance_km=segment.distance_km,
                    duration_min=segment.duration_min,
                    congestion_index=segment.congestion_index,
                    weather_severity=segment.weather_severity,
                    reliability_risk=segment.reliability_risk,
                    safety_risk=segment.safety_risk,
                    toll_cost=segment.toll_cost,
                    elevation_gain_m=segment.elevation_gain_m,
                )
            )
        return insights

    def build_stress_test(
        self,
        request: RouteRequest,
        metrics: RouteMetrics,
        segment_factors: list[RouteSegmentFactor],
    ) -> RouteStressTest:
        if not segment_factors:
            return RouteStressTest(
                simulations=0,
                on_time_probability=1.0,
                within_budget_probability=1.0,
                within_safety_probability=1.0,
                failure_probability=0.0,
                resilience_index=1.0,
                expected_duration_min=metrics.duration_min,
                duration_p10_min=metrics.duration_min,
                duration_p90_min=metrics.duration_min,
                expected_fuel_cost=metrics.fuel_cost,
                fuel_cost_p10=metrics.fuel_cost,
                fuel_cost_p90=metrics.fuel_cost,
                expected_safety_risk=metrics.safety_risk,
                worst_case_delay_min=0.0,
                highlights=[],
            )

        rng = random.Random(request.random_seed if request.random_seed is not None else 137)
        simulations = 120
        time_limit = request.constraints.max_duration_min or (metrics.duration_min * 1.08)
        budget_limit = request.constraints.max_fuel_cost or (metrics.fuel_cost * 1.12)
        safety_limit = request.constraints.max_safety_risk
        if safety_limit is None:
            safety_limit = min(0.95, metrics.safety_risk + 0.08)

        total_distance = max(metrics.distance_km, 1e-6)
        base_fuel_cost = max(metrics.fuel_cost - metrics.toll_cost, 0.0)
        duration_samples: list[float] = []
        fuel_samples: list[float] = []
        safety_samples: list[float] = []
        factor_delay_totals = {key: 0.0 for key in self._FACTOR_META}
        factor_cost_totals = {key: 0.0 for key in self._FACTOR_META}
        failed_runs = 0
        on_time_runs = 0
        within_budget_runs = 0
        within_safety_runs = 0

        for _ in range(simulations):
            duration_total = 0.0
            fuel_total = metrics.toll_cost
            weighted_safety = 0.0

            for segment in segment_factors:
                effects = self._sample_segment_effects(segment, rng)
                duration_total += segment.duration_min * (1.0 + sum(effects.values()))

                distance_share = segment.distance_km / total_distance
                segment_base_fuel = base_fuel_cost * distance_share
                cost_multiplier = 1.0 + (
                    0.55 * effects["weather"]
                    + 0.95 * effects["congestion"]
                    + 1.15 * effects["elevation"]
                    + 0.45 * effects["reliability"]
                )
                fuel_total += segment_base_fuel * cost_multiplier + (segment.toll_cost * effects["tolls"])

                sampled_safety = min(
                    1.0,
                    segment.safety_risk
                    + (0.45 * effects["weather"])
                    + (0.25 * effects["congestion"])
                    + (0.35 * effects["reliability"])
                    + (0.60 * effects["safety"])
                    + rng.uniform(0.0, 0.02),
                )
                weighted_safety += sampled_safety * max(segment.distance_km, 0.01)

                for factor_key, factor_value in effects.items():
                    factor_delay_totals[factor_key] += segment.duration_min * factor_value
                    if factor_key == "tolls":
                        factor_cost_totals[factor_key] += segment.toll_cost * factor_value
                    else:
                        factor_cost_totals[factor_key] += segment_base_fuel * factor_value

            safety_total = weighted_safety / max(total_distance, 0.01)
            duration_samples.append(duration_total)
            fuel_samples.append(fuel_total)
            safety_samples.append(safety_total)

            is_on_time = duration_total <= time_limit
            is_within_budget = fuel_total <= budget_limit
            is_within_safety = safety_total <= safety_limit
            on_time_runs += int(is_on_time)
            within_budget_runs += int(is_within_budget)
            within_safety_runs += int(is_within_safety)
            if not (is_on_time and is_within_budget and is_within_safety):
                failed_runs += 1

        duration_p10 = self._percentile(duration_samples, 0.10)
        duration_p90 = self._percentile(duration_samples, 0.90)
        fuel_p10 = self._percentile(fuel_samples, 0.10)
        fuel_p90 = self._percentile(fuel_samples, 0.90)
        failure_probability = failed_runs / simulations
        duration_spread = max(0.0, duration_p90 - duration_p10) / max(metrics.duration_min, 1.0)
        cost_spread = max(0.0, fuel_p90 - fuel_p10) / max(metrics.fuel_cost, 1.0)
        resilience_index = max(
            0.0,
            min(
                1.0,
                (1.0 - failure_probability) * 0.55
                + (on_time_runs / simulations) * 0.20
                + (within_budget_runs / simulations) * 0.15
                + (within_safety_runs / simulations) * 0.10
                - min(0.25, duration_spread * 0.12)
                - min(0.20, cost_spread * 0.10),
            ),
        )

        highlights = self._build_stress_highlights(
            metrics=metrics,
            simulations=simulations,
            factor_delay_totals=factor_delay_totals,
            factor_cost_totals=factor_cost_totals,
        )

        return RouteStressTest(
            simulations=simulations,
            on_time_probability=on_time_runs / simulations,
            within_budget_probability=within_budget_runs / simulations,
            within_safety_probability=within_safety_runs / simulations,
            failure_probability=failure_probability,
            resilience_index=resilience_index,
            expected_duration_min=mean(duration_samples),
            duration_p10_min=duration_p10,
            duration_p90_min=duration_p90,
            expected_fuel_cost=mean(fuel_samples),
            fuel_cost_p10=fuel_p10,
            fuel_cost_p90=fuel_p90,
            expected_safety_risk=mean(safety_samples),
            worst_case_delay_min=max(duration_samples) - metrics.duration_min,
            highlights=highlights,
        )

    def _build_stress_highlights(
        self,
        metrics: RouteMetrics,
        simulations: int,
        factor_delay_totals: dict[str, float],
        factor_cost_totals: dict[str, float],
    ) -> list[StressTestHighlight]:
        ordered = sorted(
            self._FACTOR_META,
            key=lambda key: (factor_delay_totals[key] + factor_cost_totals[key]),
            reverse=True,
        )[:3]
        highlights: list[StressTestHighlight] = []
        for key in ordered:
            meta = self._FACTOR_META[key]
            highlights.append(
                StressTestHighlight(
                    factor_key=key,
                    factor_label=meta["label"],
                    expected_delay_min=factor_delay_totals[key] / simulations,
                    expected_cost_increase=factor_cost_totals[key] / simulations,
                    note=meta["note"],
                )
            )
        return highlights

    def _factor_scores(self, segment: RouteSegmentFactor) -> dict[str, float]:
        distance_m = max(segment.distance_km * 1000.0, 1.0)
        grade_ratio = segment.elevation_gain_m / distance_m
        return {
            "weather": min(1.0, segment.weather_severity),
            "congestion": min(1.0, segment.congestion_index),
            "elevation": min(1.0, grade_ratio * 20.0),
            "tolls": min(1.0, segment.toll_cost / 250.0),
            "safety": min(1.0, segment.safety_risk),
            "reliability": min(1.0, segment.reliability_risk),
        }

    def _sample_segment_effects(
        self,
        segment: RouteSegmentFactor,
        rng: random.Random,
    ) -> dict[str, float]:
        factor_scores = self._factor_scores(segment)
        return {
            "weather": factor_scores["weather"] * rng.uniform(0.03, 0.18),
            "congestion": factor_scores["congestion"] * rng.uniform(0.05, 0.30),
            "elevation": factor_scores["elevation"] * rng.uniform(0.03, 0.16),
            "tolls": factor_scores["tolls"] * rng.uniform(0.02, 0.08),
            "safety": factor_scores["safety"] * rng.uniform(0.02, 0.12),
            "reliability": factor_scores["reliability"] * rng.uniform(0.03, 0.14),
        }

    def _build_segment_narrative(
        self,
        start_label: str,
        end_label: str,
        factor_key: str,
        segment: RouteSegmentFactor,
    ) -> str:
        if factor_key == "weather":
            return (
                f"Участок {start_label} -> {end_label} сильнее всего чувствителен к погоде: "
                f"риск {segment.weather_severity:.2f}, безопасность {segment.safety_risk:.2f}."
            )
        if factor_key == "congestion":
            return (
                f"Участок {start_label} -> {end_label} теряет устойчивость из-за трафика: "
                f"загруженность {segment.congestion_index:.2f}, средняя скорость {segment.avg_speed_kph:.1f} км/ч."
            )
        if factor_key == "elevation":
            return (
                f"Участок {start_label} -> {end_label} наиболее чувствителен к рельефу: "
                f"набор высоты {segment.elevation_gain_m:.0f} м на {segment.distance_km:.1f} км."
            )
        if factor_key == "tolls":
            return (
                f"На участке {start_label} -> {end_label} заметнее всего влияет платная составляющая: "
                f"{segment.toll_cost:.0f} ₽."
            )
        if factor_key == "safety":
            return (
                f"Участок {start_label} -> {end_label} требует осторожного режима: "
                f"риск безопасности {segment.safety_risk:.2f}."
            )
        return (
            f"Участок {start_label} -> {end_label} подвержен разбросу по времени: "
            f"риск надёжности {segment.reliability_risk:.2f}."
        )

    @staticmethod
    def _severity_level(value: float) -> str:
        if value >= 0.65:
            return "high"
        if value >= 0.35:
            return "medium"
        return "low"

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
