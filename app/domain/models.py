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


class OptimizationStrategy(str, Enum):
    strict = "strict"
    balanced = "balanced"
    custom = "custom"
    user_driven = "user-driven"


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


class CargoProfile(str, Enum):
    standard = "standard"
    perishable = "perishable"
    fragile = "fragile"
    hazardous = "hazardous"
    heavy = "heavy"
    high_value = "high_value"


class Point(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    label: str | None = Field(default=None, max_length=64)


class GeocodingResultDto(BaseModel):
    point: Point
    address: str | None = None
    score: float | None = None
    entity_type: str | None = None
    provider: str


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
    road_quality: float = Field(default=0.7, ge=0)
    dynamic_events: float = Field(default=0.8, ge=0)
    operational_cost: float = Field(default=0.9, ge=0)
    cargo_risk: float = Field(default=0.6, ge=0)

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
            + self.road_quality
            + self.dynamic_events
            + self.operational_cost
            + self.cargo_risk
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
            road_quality=self.road_quality / total,
            dynamic_events=self.dynamic_events / total,
            operational_cost=self.operational_cost / total,
            cargo_risk=self.cargo_risk / total,
        )


class OptimizationConstraints(BaseModel):
    max_distance_km: float | None = Field(default=None, gt=0)
    max_duration_min: float | None = Field(default=None, gt=0)
    max_fuel_cost: float | None = Field(default=None, gt=0)
    max_operational_cost: float | None = Field(default=None, gt=0)
    max_co2_kg: float | None = Field(default=None, gt=0)
    max_safety_risk: float | None = Field(default=None, ge=0, le=1)
    max_cargo_risk: float | None = Field(default=None, ge=0, le=1)


class TradeoffTolerance(BaseModel):
    max_distance_regression_pct: float = Field(default=0.0, ge=0, le=100)
    max_duration_regression_pct: float = Field(default=0.0, ge=0, le=100)
    max_fuel_cost_regression_pct: float = Field(default=0.0, ge=0, le=100)
    max_operational_cost_regression_pct: float = Field(default=0.0, ge=0, le=100)
    max_fuel_liters_regression_pct: float = Field(default=0.0, ge=0, le=100)
    max_co2_regression_pct: float = Field(default=0.0, ge=0, le=100)
    max_cargo_risk_regression_pct: float = Field(default=0.0, ge=0, le=100)
    min_risk_reduction_pct: float = Field(default=0.0, ge=0, le=100)
    min_reliability_gain_pct: float = Field(default=0.0, ge=0, le=100)
    allow_constraint_penalty_regression: bool = False


class VehicleDimensions(BaseModel):
    height_m: float | None = Field(default=None, gt=0, le=6)
    weight_t: float | None = Field(default=None, gt=0, le=80)
    width_m: float | None = Field(default=None, gt=0, le=5)
    length_m: float | None = Field(default=None, gt=0, le=30)


class CargoParameters(BaseModel):
    profile: CargoProfile = CargoProfile.standard
    weight_t: float | None = Field(default=None, ge=0, le=80)
    declared_value_rub: float | None = Field(default=None, ge=0, le=1_000_000_000)
    deadline_at: datetime | None = None


class CvrpParameters(BaseModel):
    point_demands_t: list[float] = Field(default_factory=list)
    vehicle_count: int = Field(default=1, ge=1, le=200)
    depot_index: int = Field(default=0, ge=0)
    return_to_depot: bool = True

    @field_validator("point_demands_t")
    @classmethod
    def validate_demands_non_negative(cls, demands: list[float]) -> list[float]:
        if any(value < 0 for value in demands):
            raise ValueError("Point demands must be non-negative.")
        return demands


class OperatingCostParameters(BaseModel):
    driver_cost_per_hour: float | None = Field(default=None, ge=0, le=100_000)
    maintenance_cost_per_km: float | None = Field(default=None, ge=0, le=10_000)


class RouteRequest(BaseModel):
    points: list[Point] = Field(..., min_length=2)
    optimize: bool = True
    fix_ends: bool = False
    profile: TransportProfile = TransportProfile.driving
    vehicle_class: VehicleClass = VehicleClass.passenger
    vehicle_dimensions: VehicleDimensions = Field(default_factory=VehicleDimensions)
    vehicle_capacity_t: float | None = Field(default=None, gt=0, le=80)
    cargo: CargoParameters = Field(default_factory=CargoParameters)
    cvrp: CvrpParameters = Field(default_factory=CvrpParameters)
    operating_costs: OperatingCostParameters = Field(default_factory=OperatingCostParameters)
    fuel_type: FuelType = FuelType.petrol
    fuel_consumption_l_per_100km: float | None = Field(default=None, ge=1, le=80)

    optimize_mode: OptimizationMode = OptimizationMode.weighted
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.balanced
    tradeoff_tolerance: TradeoffTolerance = Field(default_factory=TradeoffTolerance)
    priority_profile: PriorityProfile = PriorityProfile.balanced
    criteria_weights: CriteriaWeights = Field(default_factory=CriteriaWeights)
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    use_dynamic_weights: bool = True
    departure_at: datetime | None = None
    adapt_from_run_id: str | None = Field(default=None, max_length=64)
    warm_start_orders: list[list[int]] = Field(default_factory=list, exclude=True)

    population_size: int = Field(default=96, ge=20, le=400)
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

    @model_validator(mode="after")
    def validate_cvrp_shape(self) -> "RouteRequest":
        if self.cvrp.depot_index >= len(self.points):
            raise ValueError("CVRP depot_index must refer to an existing point.")
        if len(self.cvrp.point_demands_t) > len(self.points):
            raise ValueError("CVRP point_demands_t cannot contain more entries than points.")
        return self


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
    surface_multiplier: float = 1.0
    dynamic_events_multiplier: float = 1.0
    load_multiplier: float = 1.0
    load_ratio: float = 0.0
    cargo_weight_t: float | None = None
    vehicle_capacity_t: float | None = None
    liters_total: float
    price_per_liter: float
    total_cost: float
    currency: str
    price_source: str
    price_source_url: str | None = None
    price_date: str | None = None
    price_retrieved_at: str


class OperationalCostBreakdown(BaseModel):
    fuel_and_tolls: float
    fuel_only: float
    toll_cost: float
    driver_cost: float
    maintenance_cost: float
    cargo_expected_loss: float
    total_cost: float
    cargo_risk: float
    currency: str


class RouteFitnessComponents(BaseModel):
    distance: float
    time: float
    cost: float
    fuel: float
    traffic_factor: float = 0.0
    weather_factor: float = 0.0
    road_event_factor: float = 0.0
    road_quality_factor: float = 0.0
    restriction_penalty: float = 0.0
    operational_complexity: float
    penalty: float


class RouteMetrics(BaseModel):
    distance_km: float
    duration_min: float
    fuel_liters: float
    fuel_cost: float
    operational_cost: float = 0.0
    driver_cost: float = 0.0
    maintenance_cost: float = 0.0
    co2_kg: float
    congestion_index: float
    weather_risk: float
    reliability_score: float
    route_reliability_score: float = 100.0
    safety_risk: float
    toll_cost: float
    road_quality_risk: float = 0.0
    incident_risk: float = 0.0
    roadwork_risk: float = 0.0
    dynamic_event_risk: float = 0.0
    traffic_factor: float = 0.0
    weather_factor: float = 0.0
    road_event_factor: float = 0.0
    road_quality_factor: float = 0.0
    cargo_risk: float = 0.0
    cargo_expected_loss: float = 0.0
    objective_score: float
    constraint_penalty: float
    feasible: bool
    infrastructure_penalty: float = 0.0
    temporal_restriction_penalty: float = 0.0
    deadline_penalty: float = 0.0
    capacity_penalty: float = 0.0
    vehicle_routes_used: int = 1
    max_route_load_t: float = 0.0
    capacity_utilization: float = 0.0
    capacity_feasible: bool = True
    refined_segments_count: int = 0
    average_detour_ratio: float = 0.0
    segment_alternative_gain_pct: float = 0.0
    violated_constraints: list[str] = Field(default_factory=list)
    fitness_components: RouteFitnessComponents | None = None

    @model_validator(mode="after")
    def fill_analytics_aliases(self) -> "RouteMetrics":
        if self.route_reliability_score == 100.0 and self.reliability_score <= 1.0:
            self.route_reliability_score = max(0.0, min(100.0, self.reliability_score * 100.0))
        if self.traffic_factor == 0.0:
            self.traffic_factor = max(0.0, min(1.0, self.congestion_index))
        if self.weather_factor == 0.0:
            self.weather_factor = max(0.0, min(1.0, self.weather_risk))
        if self.road_event_factor == 0.0:
            self.road_event_factor = max(0.0, min(1.0, self.dynamic_event_risk))
        if self.road_quality_factor == 0.0:
            self.road_quality_factor = max(0.0, min(1.0, self.road_quality_risk))
        return self


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
    road_quality: float = 1.0
    road_quality_risk: float = 0.0
    incident_risk: float = 0.0
    roadwork_risk: float = 0.0
    dynamic_event_risk: float = 0.0
    cargo_risk: float = 0.0
    temporal_accessible: bool = True
    infrastructure_penalty: float = 0.0
    violated_constraints: list[str] = Field(default_factory=list)
    height_clearance_m: float | None = None
    weight_limit_t: float | None = None
    width_limit_m: float | None = None
    length_limit_m: float | None = None
    infrastructure_accessible: bool = True
    segment_variant_type: str | None = None
    segment_variant_score: float | None = None
    detour_ratio: float = 0.0
    refinement_applied_on_segment: bool = False


class TerrainTrend(str, Enum):
    uphill = "uphill"
    downhill = "downhill"
    flat = "flat"


class RouteTerrainSegment(BaseModel):
    trend: TerrainTrend
    geometry: list[list[float]] = Field(default_factory=list)
    distance_km: float
    elevation_delta_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    grade_pct: float


class RouteTerrainProfile(BaseModel):
    sampled_points: int = 0
    total_gain_m: float = 0.0
    total_loss_m: float = 0.0
    max_uphill_grade_pct: float = 0.0
    max_downhill_grade_pct: float = 0.0
    source: str = "unknown"
    segments: list[RouteTerrainSegment] = Field(default_factory=list)


class RouteSegmentInsight(BaseModel):
    start_index: int
    end_index: int
    start_label: str
    end_label: str
    dominant_factor_key: str
    dominant_factor_label: str
    severity_score: float
    severity_level: str
    color_hex: str
    map_color_hex: str | None = None
    map_stroke_weight: int = 8
    map_dash_array: str | None = None
    is_problematic: bool = False
    narrative: str
    distance_km: float
    duration_min: float
    congestion_index: float
    weather_severity: float
    reliability_risk: float
    safety_risk: float
    toll_cost: float
    elevation_gain_m: float
    road_quality: float = 1.0
    road_quality_risk: float = 0.0
    incident_risk: float = 0.0
    roadwork_risk: float = 0.0
    dynamic_event_risk: float = 0.0
    cargo_risk: float = 0.0
    temporal_accessible: bool = True
    infrastructure_penalty: float = 0.0
    violated_constraints: list[str] = Field(default_factory=list)


class StressTestHighlight(BaseModel):
    factor_key: str
    factor_label: str
    expected_delay_min: float
    expected_cost_increase: float
    note: str


class RouteStressTest(BaseModel):
    simulations: int
    on_time_probability: float
    within_budget_probability: float
    within_safety_probability: float
    failure_probability: float
    resilience_index: float
    expected_duration_min: float
    duration_p10_min: float
    duration_p90_min: float
    expected_fuel_cost: float
    fuel_cost_p10: float
    fuel_cost_p90: float
    expected_safety_risk: float
    worst_case_delay_min: float
    highlights: list[StressTestHighlight] = Field(default_factory=list)


class RouteAlternative(BaseModel):
    ordered_points: list[Point]
    metrics: RouteMetrics
    rank: int
    crowding_distance: float | None = None
    geometry: list[list[float]] = Field(default_factory=list)
    provider: str | None = None
    terrain_profile: RouteTerrainProfile | None = None


class RouteVehicleRoute(BaseModel):
    vehicle_index: int
    order_indices: list[int]
    ordered_points: list[Point]
    demand_t: float
    capacity_t: float
    load_ratio: float
    feasible: bool
    metrics: RouteMetrics | None = None
    geometry: list[list[float]] = Field(default_factory=list)
    provider: str | None = None
    terrain_profile: RouteTerrainProfile | None = None


class CvrpPlanInfo(BaseModel):
    enabled: bool = False
    depot_index: int = 0
    vehicle_count: int = 1
    routes_used: int = 1
    capacity_t: float = 0.0
    total_demand_t: float = 0.0
    max_route_load_t: float = 0.0
    max_load_ratio: float = 0.0
    feasible: bool = True
    capacity_penalty: float = 0.0
    total_distance_km: float = 0.0
    total_duration_min: float = 0.0
    makespan_min: float = 0.0
    routes: list[RouteVehicleRoute] = Field(default_factory=list)


class PerformanceTimings(BaseModel):
    context_ms: float = 0.0
    optimization_ms: float = 0.0
    refinement_ms: float = 0.0
    analysis_ms: float = 0.0
    total_ms: float = 0.0


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
    warm_start_solutions: int = 0
    population_memory_solutions: int = 0
    repaired_solutions: int = 0
    forbidden_edges: int = 0
    forbidden_edges_count: int = 0
    unrepairable_candidates: int = 0
    infrastructure_violations_count: int = 0
    baseline_guard_applied: bool = False
    original_best_score: float | None = None
    baseline_score: float | None = None
    baseline_guard_reason: str | None = None
    final_selected_from: str = "optimizer"
    final_selection_reason: str = "optimizer_best_selected"
    segment_alternatives_enabled: bool = False
    segment_alternatives_total_candidates: int = 0
    segment_alternatives_used: int = 0
    segment_alternative_gain_pct: float = 0.0
    route_refinement_applied: bool = False
    route_refinement_reason: str | None = None
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.strict
    accepted_tradeoff: bool = False
    rejected_regression_metrics: list[str] = Field(default_factory=list)
    rejected_alternative_reasons: list[str] = Field(default_factory=list)
    performance_timings: PerformanceTimings | None = None


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
    operational_cost: float = 0.0
    cargo_risk: float = 0.0
    co2_kg: float
    objective_score: float


class RouteComparisonInfo(BaseModel):
    baseline_ordered_points: list[Point]
    baseline_geometry: list[list[float]] = Field(default_factory=list)
    baseline_metrics: RouteMetrics
    baseline_terrain_profile: RouteTerrainProfile | None = None
    optimized_metrics: RouteMetrics
    delta: RouteComparisonDelta
    improvement_pct: RouteComparisonDelta
    baseline_score: ScoreExplanation
    optimized_score: ScoreExplanation


class RouteComparisonRouteView(BaseModel):
    label: str
    ordered_points: list[Point]
    geometry: list[list[float]] = Field(default_factory=list)
    metrics: RouteMetrics
    terrain_profile: RouteTerrainProfile | None = None
    score: ScoreExplanation | None = None
    provider: str | None = None


class RouteComparisonSummary(BaseModel):
    baseline: RouteComparisonRouteView
    selected: RouteComparisonRouteView
    delta: RouteComparisonDelta
    improvement_pct: RouteComparisonDelta
    improved_metrics: list[str] = Field(default_factory=list)


class DecisionExplanation(BaseModel):
    main_reason: str
    top_positive_factors: list[str] = Field(default_factory=list)
    top_negative_factors: list[str] = Field(default_factory=list)
    rejected_reasons: list[str] = Field(default_factory=list)
    influential_criteria: list[str] = Field(default_factory=list)
    constraints_influence: list[str] = Field(default_factory=list)
    selected_from: str | None = None
    strategy: OptimizationStrategy = OptimizationStrategy.strict
    compromise_accepted: bool = False


class RouteQualityComponent(BaseModel):
    key: str
    label: str
    score: float
    weight: float


class RouteQualityIndex(BaseModel):
    score: float
    label: str
    components: list[RouteQualityComponent] = Field(default_factory=list)


class DataConfidenceScore(BaseModel):
    score: float
    label: str
    source_scores: dict[str, float] = Field(default_factory=dict)
    fallback_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ConstraintHealthItem(BaseModel):
    key: str
    label: str
    status: str
    value: float | None = None
    limit: float | None = None
    margin_pct: float | None = None
    violated: bool = False


class ConstraintHealthReport(BaseModel):
    overall_status: str
    items: list[ConstraintHealthItem] = Field(default_factory=list)


class SegmentCandidate(BaseModel):
    variant_id: str
    variant_type: str
    from_index: int
    to_index: int
    distance_km: float
    duration_min: float
    fuel_liters: float | None = None
    fuel_cost: float | None = None
    risk_exposure: float
    road_quality_risk: float
    weather_risk: float
    dynamic_event_risk: float
    safety_risk: float
    cargo_risk: float
    detour_ratio: float
    restriction_penalty: float
    objective_score: float
    geometry: list[list[float]] | None = None
    geojson: dict[str, Any] | None = None
    data_source: str
    explanation: str | None = None


class SegmentAlternativeSet(BaseModel):
    from_index: int
    to_index: int
    candidates: list[SegmentCandidate] = Field(default_factory=list)
    best_candidate: SegmentCandidate
    baseline_candidate: SegmentCandidate


class SegmentAlternativesSummary(BaseModel):
    enabled: bool = False
    total_pairs: int = 0
    total_candidates: int = 0
    used_candidates: int = 0
    average_candidates_per_pair: float = 0.0
    average_detour_ratio: float = 0.0
    estimated_gain_pct: float = 0.0


class RouteRefinementSegmentChoice(BaseModel):
    start_index: int
    end_index: int
    from_label: str
    to_label: str
    selected_variant: str
    improvement_reason: str
    baseline_score: float
    selected_score: float
    improvement_pct: float
    distance_delta_km: float = 0.0
    duration_delta_min: float = 0.0


class RouteRefinementInfo(BaseModel):
    applied: bool = False
    reason: str = "not_requested"
    improvement_pct: float = 0.0
    changed_segments: int = 0
    candidate_count: int = 0
    max_detour_ratio: float = 0.0
    segment_choices: list[RouteRefinementSegmentChoice] = Field(default_factory=list)


class DataSourceInfo(BaseModel):
    routing: str
    matrix: str
    weather: str
    elevation: str
    traffic: str
    tolls: str
    fuel_prices: str
    infrastructure: str = "unknown"
    road_quality: str = "unknown"
    road_events: str = "unknown"


class RouteAnalysisMatrices(BaseModel):
    point_labels: list[str] = Field(default_factory=list)
    distance_km: list[list[float]] = Field(default_factory=list)
    duration_min: list[list[float]] = Field(default_factory=list)
    traffic_index: list[list[float]] = Field(default_factory=list)
    toll_cost: list[list[float]] = Field(default_factory=list)
    road_quality: list[list[float]] = Field(default_factory=list)
    incident_risk: list[list[float]] = Field(default_factory=list)
    roadwork_risk: list[list[float]] = Field(default_factory=list)
    elevation_gain_m: list[list[float]] = Field(default_factory=list)
    elevation_loss_m: list[list[float]] = Field(default_factory=list)
    mean_elevation_m: list[list[float]] = Field(default_factory=list)
    height_clearance_m: list[list[float | None]] = Field(default_factory=list)
    weight_limit_t: list[list[float | None]] = Field(default_factory=list)
    width_limit_m: list[list[float | None]] = Field(default_factory=list)
    length_limit_m: list[list[float | None]] = Field(default_factory=list)
    infrastructure_access: list[list[bool]] = Field(default_factory=list)
    temporal_access: list[list[bool]] = Field(default_factory=list)


class RouteResponse(BaseModel):
    run_id: str | None = None
    ordered_points: list[Point]
    total_distance_km: float
    total_duration_min: float | None
    geometry: list[list[float]]
    geojson: dict[str, Any]
    provider: str
    fuel_cost: FuelCostBreakdown | None = None
    operational_cost: OperationalCostBreakdown | None = None
    metrics: RouteMetrics | None = None
    alternatives: list[RouteAlternative] = Field(default_factory=list)
    segment_factors: list[RouteSegmentFactor] = Field(default_factory=list)
    segment_insights: list[RouteSegmentInsight] = Field(default_factory=list)
    terrain_profile: RouteTerrainProfile | None = None
    stress_test: RouteStressTest | None = None
    diagnostics: OptimizationDiagnostics | None = None
    decision_explanation: DecisionExplanation | None = None
    route_quality_index: RouteQualityIndex | None = None
    data_confidence: DataConfidenceScore | None = None
    constraint_health: ConstraintHealthReport | None = None
    dynamic_weights: DynamicWeightsInfo | None = None
    refinement: RouteRefinementInfo | None = None
    segment_alternatives_summary: SegmentAlternativesSummary | None = None
    comparison: RouteComparisonInfo | None = None
    comparison_summary: RouteComparisonSummary | None = None
    analysis_matrices: RouteAnalysisMatrices | None = None
    data_sources: DataSourceInfo | None = None
    population_memory_orders: list[list[int]] = Field(default_factory=list)
    cvrp_plan: CvrpPlanInfo | None = None


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
