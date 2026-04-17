"""Domain models and DTOs."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TransportProfile(str, Enum):
    driving = "driving"
    walking = "walking"
    cycling = "cycling"


class VehicleClass(str, Enum):
    passenger = "passenger"
    light_truck = "light_truck"
    heavy_truck = "heavy_truck"


class FuelType(str, Enum):
    petrol = "petrol"
    diesel = "diesel"


class OptimizationMode(str, Enum):
    weighted = "weighted"
    pareto = "pareto"


class OptimizationReason(str, Enum):
    not_enough_points = "not_enough_points"
    fixed_route = "fixed_route"
    optimize_disabled = "optimize_disabled"


class ScoreMode(str, Enum):
    absolute_single_candidate = "absolute_single_candidate"
    population_normalized = "population_normalized"


class PriorityProfile(str, Enum):
    balanced = "balanced"
    fastest = "fastest"
    cheapest = "cheapest"
    safest = "safest"
    greenest = "greenest"


class Point(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    label: str | None = Field(default=None, max_length=64)


class CriteriaWeights(BaseModel):
    distance: float = Field(default=1.0, ge=0)
    duration: float = Field(default=1.2, ge=0)
    fuel_cost: float = Field(default=1.1, ge=0)
    emissions: float = Field(default=0.9, ge=0)
    congestion: float = Field(default=1.0, ge=0)
    weather_risk: float = Field(default=0.8, ge=0)
    reliability: float = Field(default=0.9, ge=0)
    safety: float = Field(default=1.0, ge=0)
    tolls: float = Field(default=0.4, ge=0)

    @model_validator(mode="after")
    def validate_non_zero_sum(self) -> "CriteriaWeights":
        if self.total() <= 0:
            raise ValueError("At least one criteria weight must be > 0.")
        return self

    def total(self) -> float:
        return (
            self.distance
            + self.duration
            + self.fuel_cost
            + self.emissions
            + self.congestion
            + self.weather_risk
            + self.reliability
            + self.safety
            + self.tolls
        )

    def normalized(self) -> "CriteriaWeights":
        total = self.total()
        if total <= 0:
            return CriteriaWeights()
        return CriteriaWeights(
            distance=self.distance / total,
            duration=self.duration / total,
            fuel_cost=self.fuel_cost / total,
            emissions=self.emissions / total,
            congestion=self.congestion / total,
            weather_risk=self.weather_risk / total,
            reliability=self.reliability / total,
            safety=self.safety / total,
            tolls=self.tolls / total,
        )


class OptimizationConstraints(BaseModel):
    max_distance_km: float | None = Field(default=None, gt=0)
    max_duration_min: float | None = Field(default=None, gt=0)
    max_fuel_cost: float | None = Field(default=None, gt=0)
    max_co2_kg: float | None = Field(default=None, gt=0)
    max_safety_risk: float | None = Field(default=None, ge=0, le=1)


class RouteRequest(BaseModel):
    points: list[Point] = Field(..., min_length=2)
    optimize: bool = True
    fix_ends: bool = True
    profile: TransportProfile = TransportProfile.driving
    vehicle_class: VehicleClass = VehicleClass.passenger
    fuel_type: FuelType = FuelType.petrol
    fuel_consumption_l_per_100km: float | None = Field(default=None, ge=1, le=80)

    optimize_mode: OptimizationMode = OptimizationMode.weighted
    priority_profile: PriorityProfile = PriorityProfile.balanced
    criteria_weights: CriteriaWeights = Field(default_factory=CriteriaWeights)
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    use_dynamic_weights: bool = True
    departure_at: datetime | None = None

    population_size: int = Field(default=96, ge=24, le=400)
    generations: int = Field(default=120, ge=20, le=800)
    crossover_rate: float = Field(default=0.88, ge=0.1, le=1.0)
    mutation_rate: float = Field(default=0.22, ge=0.01, le=0.9)
    max_alternatives: int = Field(default=8, ge=1, le=20)
    random_seed: int | None = Field(default=None, ge=0)

    @field_validator("points")
    @classmethod
    def validate_points_unique(cls, points: list[Point]) -> list[Point]:
        if len(points) < 2:
            return points
        first = points[0]
        last = points[-1]
        if first.lat == last.lat and first.lon == last.lon:
            raise ValueError("Start and end points must differ.")
        return points


class FuelCostBreakdown(BaseModel):
    fuel_type: FuelType
    vehicle_class: VehicleClass
    consumption_l_per_100km: float
    distance_km: float
    uphill_share_pct: float
    downhill_share_pct: float
    terrain_multiplier: float
    mountain_multiplier: float = 1.0
    temperature_multiplier: float = 1.0
    congestion_multiplier: float = 1.0
    liters_total: float
    price_per_liter: float
    total_cost: float
    currency: str
    price_source: str
    price_source_url: str | None = None
    price_date: str | None = None
    price_retrieved_at: str


class RouteMetrics(BaseModel):
    distance_km: float
    duration_min: float
    fuel_liters: float
    fuel_cost: float
    co2_kg: float
    congestion_index: float
    weather_risk: float
    reliability_score: float
    safety_risk: float
    toll_cost: float
    objective_score: float
    constraint_penalty: float
    feasible: bool


class RouteSegmentFactor(BaseModel):
    start_index: int
    end_index: int
    distance_km: float
    duration_min: float
    avg_speed_kph: float
    elevation_gain_m: float
    elevation_loss_m: float
    congestion_index: float
    weather_severity: float
    reliability_risk: float
    safety_risk: float
    toll_cost: float


class RouteAlternative(BaseModel):
    ordered_points: list[Point]
    metrics: RouteMetrics
    rank: int
    crowding_distance: float | None = None


class OptimizationDiagnostics(BaseModel):
    mode: OptimizationMode
    optimization_active: bool = True
    optimization_reason: OptimizationReason | None = None
    score_mode: ScoreMode = ScoreMode.population_normalized
    generations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    stagnation_generations: int
    evaluated_solutions: int
    pareto_solutions: int


class DynamicWeightsInfo(BaseModel):
    base: CriteriaWeights
    adjusted: CriteriaWeights
    triggers: list[str] = Field(default_factory=list)


class ScoreComponent(BaseModel):
    key: str
    label: str
    weight: float
    raw_value: float
    normalized_value: float
    contribution: float


class ScoreExplanation(BaseModel):
    score_mode: ScoreMode
    total_score: float
    components: list[ScoreComponent] = Field(default_factory=list)


class RouteComparisonDelta(BaseModel):
    distance_km: float
    duration_min: float
    fuel_cost: float
    co2_kg: float
    objective_score: float


class RouteComparisonInfo(BaseModel):
    baseline_ordered_points: list[Point]
    baseline_metrics: RouteMetrics
    optimized_metrics: RouteMetrics
    delta: RouteComparisonDelta
    improvement_pct: RouteComparisonDelta
    baseline_score: ScoreExplanation
    optimized_score: ScoreExplanation


class DataSourceInfo(BaseModel):
    routing: str
    matrix: str
    weather: str
    elevation: str
    traffic: str
    tolls: str
    fuel_prices: str


class RouteResponse(BaseModel):
    run_id: str | None = None
    ordered_points: list[Point]
    total_distance_km: float
    total_duration_min: float | None
    geometry: list[list[float]]
    geojson: dict[str, Any]
    provider: str
    fuel_cost: FuelCostBreakdown | None = None
    metrics: RouteMetrics | None = None
    alternatives: list[RouteAlternative] = Field(default_factory=list)
    segment_factors: list[RouteSegmentFactor] = Field(default_factory=list)
    diagnostics: OptimizationDiagnostics | None = None
    dynamic_weights: DynamicWeightsInfo | None = None
    comparison: RouteComparisonInfo | None = None
    data_sources: DataSourceInfo | None = None


class RouteRunListItem(BaseModel):
    run_id: str
    created_at: str
    provider_summary: str
    objective_score: float | None = None
    feasible_count: int = 0


class RouteRunDetails(BaseModel):
    run_id: str
    created_at: str
    request: dict[str, Any]
    response: dict[str, Any]
