from __future__ import annotations

from app.domain.models import CargoProfile, CriteriaWeights, DynamicWeightsInfo, PriorityProfile, RouteRequest
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
            "distance": max(base.distance, 0.0),
            "duration": max(base.duration, 0.0),
            "fuel_cost": 0.0,
            "emissions": 0.0,
            "congestion": 0.0,
            "weather_risk": 0.0,
            "reliability": 0.0,
            "safety": 0.0,
            "tolls": 0.0,
            "road_quality": 0.0,
            "dynamic_events": 0.0,
            "operational_cost": max(base.operational_cost, 0.0),
            "cargo_risk": 0.0,
        }
        triggers: list[str] = []

        self._apply_priority_profile(adjusted, request.priority_profile)
        cargo_trigger = self._apply_cargo_profile(adjusted, request.cargo.profile)
        if cargo_trigger is not None:
            triggers.append(cargo_trigger)

        if request.use_dynamic_weights:
            hour = context.departure_at.hour
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                adjusted["duration"] *= 1.18
                triggers.append("peak_hour")

            weather = context.weather.severity
            if weather >= 0.45:
                adjusted["duration"] *= 1.10
                adjusted["operational_cost"] *= 1.05
                triggers.append("bad_weather")

            mean_traffic = context.mean_congestion()
            if mean_traffic >= 0.42:
                adjusted["duration"] *= 1.14
                triggers.append("high_congestion")

            mean_events = context.mean_dynamic_event_risk()
            if mean_events >= 0.35:
                adjusted["duration"] *= 1.08
                adjusted["operational_cost"] *= 1.06
                triggers.append("dynamic_road_events")

            if fuel_price_per_liter >= 70:
                adjusted["operational_cost"] *= 1.12
                triggers.append("high_fuel_price")
            if request.cargo.deadline_at is not None:
                adjusted["duration"] *= 1.16
                triggers.append("cargo_deadline")
        else:
            triggers.append("dynamic_disabled")

        normalized = CriteriaWeights(**adjusted).normalized()
        return DynamicWeightsInfo(base=base, adjusted=normalized, triggers=triggers)

    @staticmethod
    def _apply_priority_profile(adjusted: dict[str, float], profile: PriorityProfile) -> None:
        if profile == PriorityProfile.fastest:
            adjusted["distance"] = 0.15
            adjusted["duration"] = 0.70
            adjusted["operational_cost"] = 0.15
        elif profile == PriorityProfile.cheapest:
            adjusted["distance"] = 0.20
            adjusted["duration"] = 0.20
            adjusted["operational_cost"] = 0.60
        elif profile == PriorityProfile.safest:
            adjusted["distance"] = 0.25
            adjusted["duration"] = 0.45
            adjusted["operational_cost"] = 0.30
        elif profile == PriorityProfile.greenest:
            adjusted["distance"] = 0.30
            adjusted["duration"] = 0.25
            adjusted["operational_cost"] = 0.45
            adjusted["emissions"] = 0.0
        else:
            adjusted["distance"] = 0.33
            adjusted["duration"] = 0.34
            adjusted["operational_cost"] = 0.33

    @staticmethod
    def _apply_cargo_profile(adjusted: dict[str, float], profile: CargoProfile) -> str | None:
        if profile == CargoProfile.standard:
            return None
        if profile == CargoProfile.perishable:
            adjusted["duration"] *= 1.22
            adjusted["operational_cost"] *= 1.08
            return "cargo_perishable"
        elif profile == CargoProfile.fragile:
            adjusted["operational_cost"] *= 1.08
            return "cargo_fragile"
        elif profile == CargoProfile.hazardous:
            adjusted["duration"] *= 1.08
            adjusted["operational_cost"] *= 1.10
            return "cargo_hazardous"
        elif profile == CargoProfile.heavy:
            adjusted["operational_cost"] *= 1.18
            return "cargo_heavy"
        elif profile == CargoProfile.high_value:
            adjusted["operational_cost"] *= 1.12
            return "cargo_high_value"
        return None
