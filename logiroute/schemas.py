"""
Pydantic v2 data contracts for LogiRoute AI models, optimization solvers,
and REST API endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DeliveryNode(BaseModel):
    node_id: str
    name: str
    city: str
    lat: float
    lon: float
    is_depot: bool = False
    demand_kg: float = Field(default=0.0, ge=0.0)
    demand_volume_m3: float = Field(default=0.0, ge=0.0)
    demand_units: int = Field(default=0, ge=0)
    time_window_start_hr: float = Field(default=6.0, ge=0.0, le=24.0)
    time_window_end_hr: float = Field(default=19.0, ge=0.0, le=24.0)
    service_duration_min: float = Field(default=20.0, ge=0.0)
    priority: str = Field(default="STANDARD")


class RouteStop(BaseModel):
    sequence_index: int
    node_id: str
    node_name: str
    city: str
    lat: float
    lon: float
    is_depot: bool
    arrival_hour: float
    departure_hour: float
    arrival_time_formatted: str
    departure_time_formatted: str
    time_window_formatted: str
    cumulative_distance_km: float
    leg_distance_km: float
    leg_travel_time_min: float
    delivered_kg: float
    cumulative_load_kg: float
    sla_on_time: bool
    delay_risk_probability: float


class VehicleRoutePlan(BaseModel):
    vehicle_id: str
    vehicle_class: str
    vehicle_name: str
    capacity_kg: float
    total_load_kg: float
    utilization_pct: float
    total_distance_km: float
    total_duration_hours: float
    total_cost_usd: float
    co2_emissions_kg: float
    num_stops: int
    sla_compliance_pct: float
    stops: List[RouteStop]
    geometry_coords: List[List[float]]


class VRPBenchmarkSummary(BaseModel):
    optimized_distance_km: float
    baseline_distance_km: float
    distance_saved_km: float
    distance_reduction_pct: float
    optimized_cost_usd: float
    baseline_cost_usd: float
    cost_saved_usd: float
    cost_reduction_pct: float
    optimized_co2_kg: float
    baseline_co2_kg: float
    co2_saved_kg: float
    vehicles_dispatched: int
    total_fleet_utilization_pct: float
    on_time_delivery_rate_pct: float
    solver_algorithm: str
    two_opt_iterations: int
    solve_time_ms: float


class VRPOptimizationResponse(BaseModel):
    depot: DeliveryNode
    routes: List[VehicleRoutePlan]
    benchmark: VRPBenchmarkSummary


class VRPOptimizationRequest(BaseModel):
    demand_multiplier: float = Field(default=1.0, ge=0.3, le=3.0)
    traffic_congestion_index: float = Field(default=0.35, ge=0.0, le=1.0)
    weather_severity_index: float = Field(default=0.15, ge=0.0, le=1.0)
    vehicle_capacity_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    enable_two_opt: bool = True


class ForecastDataPoint(BaseModel):
    date: str
    is_historical: bool
    actual_demand: Optional[float] = None
    p10_demand: float
    p50_demand: float
    p90_demand: float
    promo_active: bool = False
    anomaly_flag: bool = False


class DemandForecastResponse(BaseModel):
    sku_id: str
    sku_name: str
    category: str
    hub_id: str
    hub_name: str
    horizon_days: int
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    series: List[ForecastDataPoint]


class ETAPredictionRequest(BaseModel):
    origin_hub_id: str = "DEPOT-KOL-01"
    destination_hub_id: str = "NODE-SLT-01"
    distance_km: Optional[float] = Field(default=None, gt=0.0, le=2500.0)
    departure_hour: float = Field(default=8.5, ge=0.0, le=23.9)
    day_of_week: int = Field(default=2, ge=0, le=6)
    traffic_congestion_index: float = Field(default=0.45, ge=0.0, le=1.0)
    weather_severity_index: float = Field(default=0.20, ge=0.0, le=1.0)
    payload_weight_kg: float = Field(default=2400.0, ge=50.0, le=25000.0)
    promised_sla_minutes: float = Field(default=75.0, ge=15.0, le=2400.0)


class ETAPredictionResponse(BaseModel):
    origin_hub_id: str
    destination_hub_id: str
    route_distance_km: float
    free_flow_duration_min: float
    predicted_eta_min: float
    predicted_eta_p90_min: float
    sla_target_min: float
    delay_risk_probability: float
    risk_tier: str
    recommended_action: str
    factor_contributions_min: Dict[str, float]
    model_metrics: Dict[str, float]


class InventoryPolicyItem(BaseModel):
    sku_id: str
    sku_name: str
    category: str
    hub_id: str
    hub_name: str
    daily_demand_mean: float
    daily_demand_std: float
    lead_time_days_mean: float
    lead_time_days_std: float
    current_on_hand_units: int
    in_transit_units: int
    safety_stock_units: int
    reorder_point_units: int
    economic_order_qty_units: int
    order_up_to_level_s: int
    recommended_order_qty: int
    days_of_supply: float
    stockout_probability_pct: float
    status: str
    annual_holding_cost_usd: float
    annual_ordering_cost_usd: float
    total_annual_policy_cost_usd: float


class InventoryOptimizationResponse(BaseModel):
    target_service_level_pct: float
    total_skus_evaluated: int
    critical_stockout_alerts: int
    reorder_triggered_count: int
    total_recommended_po_value_usd: float
    total_annual_inventory_cost_usd: float
    policies: List[InventoryPolicyItem]


class ScenarioSimulationRequest(BaseModel):
    scenario_name: str = "Custom Stress Scenario"
    demand_multiplier: float = Field(default=1.25, ge=0.4, le=3.0)
    traffic_congestion_index: float = Field(default=0.60, ge=0.0, le=1.0)
    weather_severity_index: float = Field(default=0.45, ge=0.0, le=1.0)
    lead_time_shock_multiplier: float = Field(default=1.30, ge=0.5, le=3.0)
    target_service_level: float = Field(default=0.96, ge=0.80, le=0.999)


class ScenarioSimulationResponse(BaseModel):
    scenario_name: str
    parameters: ScenarioSimulationRequest
    routing_summary: VRPBenchmarkSummary
    inventory_summary: Dict[str, float]
    network_sla_risk_pct: float
    resilience_score: float
    executive_recommendations: List[str]
