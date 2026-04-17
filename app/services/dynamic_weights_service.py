from __future__ import annotations

from app.domain.models import CriteriaWeights, DynamicWeightsInfo, PriorityProfile, RouteRequest
from app.services.context_service import OptimizationContext


class DynamicWeightsService:
    def compute(
        self,
        request: RouteRequest,
        context: OptimizationContext,
        fuel_price_per_liter: float,
    ) -> DynamicWeightsInfo:
        base = request.criteria_weights.normalized()
        adjusted = {
            "distance": base.distance,
            "duration": base.duration,
            "fuel_cost": base.fuel_cost,
            "emissions": base.emissions,
            "congestion": base.congestion,
            "weather_risk": base.weather_risk,
            "reliability": base.reliability,
            "safety": base.safety,
            "tolls": base.tolls,
        }
        triggers: list[str] = []

        self._apply_priority_profile(adjusted, request.priority_profile)

        if request.use_dynamic_weights:
            hour = context.departure_at.hour
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                adjusted["duration"] *= 1.28
                adjusted["congestion"] *= 1.22
                adjusted["reliability"] *= 1.12
                triggers.append("peak_hour")

            weather = context.weather.severity
            if weather >= 0.45:
                adjusted["weather_risk"] *= 1.35
                adjusted["safety"] *= 1.22
                adjusted["reliability"] *= 1.15
                triggers.append("bad_weather")

            mean_traffic = context.mean_congestion()
            if mean_traffic >= 0.42:
                adjusted["duration"] *= 1.16
                adjusted["congestion"] *= 1.3
                adjusted["reliability"] *= 1.13
                triggers.append("high_congestion")

            if fuel_price_per_liter >= 70:
                adjusted["fuel_cost"] *= 1.26
                adjusted["emissions"] *= 1.1
                triggers.append("high_fuel_price")
        else:
            triggers.append("dynamic_disabled")

        normalized = CriteriaWeights(**adjusted).normalized()
        return DynamicWeightsInfo(base=base, adjusted=normalized, triggers=triggers)

    @staticmethod
    def _apply_priority_profile(adjusted: dict[str, float], profile: PriorityProfile) -> None:
        if profile == PriorityProfile.fastest:
            adjusted["duration"] *= 1.38
            adjusted["congestion"] *= 1.2
        elif profile == PriorityProfile.cheapest:
            adjusted["fuel_cost"] *= 1.42
            adjusted["tolls"] *= 1.22
        elif profile == PriorityProfile.safest:
            adjusted["safety"] *= 1.45
            adjusted["reliability"] *= 1.3
            adjusted["weather_risk"] *= 1.1
        elif profile == PriorityProfile.greenest:
            adjusted["emissions"] *= 1.55
            adjusted["fuel_cost"] *= 1.1
