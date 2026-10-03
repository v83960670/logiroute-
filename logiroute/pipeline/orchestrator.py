"""
Supply Chain Control Tower Orchestrator.

Coordinates data generation, ML model training/inference (Quantile Demand
Forecasting + Traffic-Aware ETA & SLA Risk + IsolationForest Anomaly Detection),
MLOps Observability (PSI / KS / Wasserstein Drift Monitoring), and Operations
Research solvers (CVRPTW Fleet Routing + Multi-Objective Pareto Frontier +
Multi-Echelon Stochastic Inventory Optimization).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from logiroute.config import DEFAULT_FLEET_CLASSES, DEFAULT_REGIONAL_NODES, DEFAULT_SKUS, SETTINGS
from logiroute.data.synthetic_generator import SupplyChainDataGenerator
from logiroute.mlops.drift_monitor import MLOpsDriftMonitor
from logiroute.models.anomaly_detector import SupplyChainAnomalyDetector
from logiroute.models.demand_forecaster import ProbabilisticDemandForecaster
from logiroute.models.eta_predictor import ETADelayRiskPredictor
from logiroute.optimization.inventory_optimizer import StochasticInventoryOptimizer
from logiroute.optimization.pareto_frontier import ParetoFrontierOptimizer
from logiroute.optimization.vrp_solver import CVRPTWSolver
from logiroute.schemas import (
    AnomalyDetectionResponse,
    DemandForecastResponse,
    ETAPredictionRequest,
    ETAPredictionResponse,
    InventoryOptimizationResponse,
    MLOpsDriftResponse,
    ParetoFrontierResponse,
    ScenarioSimulationRequest,
    ScenarioSimulationResponse,
    VRPOptimizationRequest,
    VRPOptimizationResponse,
)


class ControlTowerOrchestrator:
    """Singleton-style orchestrator managing the end-to-end LogiRoute AI engine."""

    def __init__(self, seed: int = SETTINGS.random_seed) -> None:
        self.seed = seed
        self.data_generator = SupplyChainDataGenerator(seed=seed)
        self.demand_forecaster = ProbabilisticDemandForecaster(random_state=seed)
        self.eta_predictor = ETADelayRiskPredictor(random_state=seed)
        self.anomaly_detector = SupplyChainAnomalyDetector(random_state=seed)
        self.drift_monitor = MLOpsDriftMonitor(seed=seed)
        self.vrp_solver = CVRPTWSolver()
        self.pareto_optimizer = ParetoFrontierOptimizer(self.vrp_solver)
        self.inventory_optimizer = StochasticInventoryOptimizer(seed=seed)
        self.is_ready: bool = False
        self.startup_time_ms: float = 0.0
        self._cached_nodes = self.data_generator.generate_daily_dispatch_nodes()

    def initialize(self, history_days: int = 365, shipment_samples: int = 3500) -> Dict[str, Any]:
        """Generate datasets, fit all ML models, and warm up optimization solvers."""
        t0 = time.perf_counter()
        demand_df = self.data_generator.generate_demand_history(days=history_days)
        shipment_df = self.data_generator.generate_shipment_telemetry(n_samples=shipment_samples)

        demand_metrics = self.demand_forecaster.fit(demand_df, holdout_days=30)
        eta_metrics = self.eta_predictor.fit(shipment_df)
        anomaly_metrics = self.anomaly_detector.fit()

        self._cached_nodes = self.data_generator.generate_daily_dispatch_nodes(demand_multiplier=1.0)
        self.is_ready = True
        self.startup_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "status": "initialized",
            "startup_time_ms": self.startup_time_ms,
            "demand_forecaster_metrics": demand_metrics,
            "eta_predictor_metrics": eta_metrics,
            "anomaly_detector_metrics": anomaly_metrics,
        }

    def get_catalog_metadata(self) -> Dict[str, Any]:
        """Return all available SKUs, distribution hubs, and vehicle classes."""
        forecast_hubs = [n for n in DEFAULT_REGIONAL_NODES if not n.is_depot][:6]
        return {
            "skus": [
                {
                    "sku_id": s.sku_id,
                    "name": s.name,
                    "category": s.category,
                    "unit_weight_kg": s.unit_weight_kg,
                    "unit_cost_usd": s.unit_cost_usd,
                    "unit_price_usd": s.unit_price_usd,
                    "lead_time_days_mean": s.lead_time_days_mean,
                }
                for s in DEFAULT_SKUS
            ],
            "forecast_hubs": [
                {
                    "hub_id": h.hub_id,
                    "name": h.name,
                    "city": h.city,
                    "lat": h.lat,
                    "lon": h.lon,
                }
                for h in forecast_hubs
            ],
            "all_nodes": [
                {
                    "hub_id": n.hub_id,
                    "name": n.name,
                    "city": n.city,
                    "lat": n.lat,
                    "lon": n.lon,
                    "is_depot": n.is_depot,
                }
                for n in DEFAULT_REGIONAL_NODES
            ],
            "fleet_classes": [
                {
                    "class_id": v.class_id,
                    "name": v.name,
                    "capacity_kg": v.capacity_kg,
                    "cost_per_km_usd": v.cost_per_km_usd,
                    "co2_kg_per_km": v.co2_kg_per_km,
                }
                for v in DEFAULT_FLEET_CLASSES
            ],
        }

    def run_vrp_optimization(self, req: VRPOptimizationRequest) -> VRPOptimizationResponse:
        if not self.is_ready:
            self.initialize()
        return self.vrp_solver.solve(
            nodes=self._cached_nodes,
            demand_multiplier=req.demand_multiplier,
            traffic_congestion_index=req.traffic_congestion_index,
            weather_severity_index=req.weather_severity_index,
            vehicle_capacity_scale=req.vehicle_capacity_scale,
            enable_two_opt=req.enable_two_opt,
        )

    def run_demand_forecast(
        self,
        sku_id: str = "SKU-PHM-101",
        hub_id: str = "NODE-SLT-01",
        horizon_days: int = 14,
        demand_multiplier: float = 1.0,
    ) -> DemandForecastResponse:
        if not self.is_ready:
            self.initialize()
        return self.demand_forecaster.forecast_sku_hub(
            sku_id=sku_id,
            hub_id=hub_id,
            horizon_days=horizon_days,
            demand_multiplier=demand_multiplier,
        )

    def run_eta_prediction(self, req: ETAPredictionRequest) -> ETAPredictionResponse:
        if not self.is_ready:
            self.initialize()
        return self.eta_predictor.predict_shipment_eta(req)

    def run_inventory_optimization(
        self,
        target_service_level: float = 0.96,
        demand_multiplier: float = 1.0,
        lead_time_shock_multiplier: float = 1.0,
    ) -> InventoryOptimizationResponse:
        if not self.is_ready:
            self.initialize()
        stats = self.demand_forecaster.get_sku_hub_demand_stats(
            demand_multiplier=demand_multiplier
        )
        return self.inventory_optimizer.optimize_network_inventory(
            demand_stats=stats,
            target_service_level=target_service_level,
            lead_time_shock_multiplier=lead_time_shock_multiplier,
        )

    def run_anomaly_detection(self, top_k: int = 15) -> AnomalyDetectionResponse:
        if not self.is_ready:
            self.initialize()
        return self.anomaly_detector.detect_anomalies(top_k=top_k)

    def run_drift_evaluation(self, drift_regime: str = "nominal") -> MLOpsDriftResponse:
        if not self.is_ready:
            self.initialize()
        return self.drift_monitor.evaluate_drift(drift_regime=drift_regime)

    def run_pareto_frontier(self, demand_multiplier: float = 1.0) -> ParetoFrontierResponse:
        if not self.is_ready:
            self.initialize()
        return self.pareto_optimizer.compute_frontier(
            nodes=self._cached_nodes, demand_multiplier=demand_multiplier
        )

    def run_scenario_simulation(
        self, req: ScenarioSimulationRequest
    ) -> ScenarioSimulationResponse:
        """
        Stress-test the entire supply chain network under compound demand,
        traffic, weather, and supplier lead-time disruptions.
        """
        if not self.is_ready:
            self.initialize()

        vrp_res = self.vrp_solver.solve(
            nodes=self._cached_nodes,
            demand_multiplier=req.demand_multiplier,
            traffic_congestion_index=req.traffic_congestion_index,
            weather_severity_index=req.weather_severity_index,
            vehicle_capacity_scale=1.0,
            enable_two_opt=True,
        )
        inv_res = self.run_inventory_optimization(
            target_service_level=req.target_service_level,
            demand_multiplier=req.demand_multiplier,
            lead_time_shock_multiplier=req.lead_time_shock_multiplier,
        )

        avg_delay_risk = 0.0
        stop_count = 0
        for route in vrp_res.routes:
            for stop in route.stops:
                if not stop.is_depot:
                    avg_delay_risk += stop.delay_risk_probability
                    stop_count += 1
        network_sla_risk_pct = round((avg_delay_risk / max(1, stop_count)) * 100.0, 1)

        stockout_ratio = inv_res.critical_stockout_alerts / max(1, inv_res.total_skus_evaluated)
        otd_score = vrp_res.benchmark.on_time_delivery_rate_pct
        resilience_score = round(
            max(
                18.0,
                min(
                    99.5,
                    0.50 * otd_score
                    + 0.30 * (100.0 - network_sla_risk_pct)
                    + 0.20 * (100.0 * (1.0 - stockout_ratio)),
                ),
            ),
            1,
        )

        recommendations: List[str] = []
        if req.demand_multiplier >= 1.20:
            recommendations.append(
                f"Demand Surge (+{int((req.demand_multiplier - 1.0)*100)}%): Activate {vrp_res.benchmark.vehicles_dispatched} "
                f"consolidated multi-stop vehicles and pre-stage fast-moving SKUs at Salt Lake & Air Cargo Express hubs."
            )
        if req.traffic_congestion_index >= 0.55 or req.weather_severity_index >= 0.45:
            recommendations.append(
                f"Corridor Friction Alert (SLA Risk {network_sla_risk_pct}%): Shift Dankuni Depot wave dispatch "
                f"forward by 45 minutes and prioritize time-critical pharma & air-cargo stops."
            )
        if inv_res.critical_stockout_alerts > 0:
            recommendations.append(
                f"Inventory Buffer Protection: Expedite {inv_res.critical_stockout_alerts} critical SKU replenishment orders "
                f"(Total PO Value ${inv_res.total_recommended_po_value_usd:,.0f}) to maintain {inv_res.target_service_level_pct}% Cycle Service Level."
            )
        recommendations.append(
            f"Combinatorial Route Optimization saves {vrp_res.benchmark.distance_saved_km:.1f} km "
            f"(-{vrp_res.benchmark.distance_reduction_pct:.1f}%) and avoids {vrp_res.benchmark.co2_saved_kg:.1f} kg CO2e "
            f"even under {req.scenario_name} conditions."
        )

        return ScenarioSimulationResponse(
            scenario_name=req.scenario_name,
            parameters=req,
            routing_summary=vrp_res.benchmark,
            inventory_summary={
                "target_service_level_pct": inv_res.target_service_level_pct,
                "critical_stockout_alerts": float(inv_res.critical_stockout_alerts),
                "reorder_triggered_count": float(inv_res.reorder_triggered_count),
                "total_recommended_po_value_usd": inv_res.total_recommended_po_value_usd,
                "total_annual_inventory_cost_usd": inv_res.total_annual_inventory_cost_usd,
            },
            network_sla_risk_pct=network_sla_risk_pct,
            resilience_score=resilience_score,
            executive_recommendations=recommendations,
        )

    def get_control_tower_overview(self) -> Dict[str, Any]:
        """Return full executive control tower snapshot in a single call."""
        if not self.is_ready:
            self.initialize()

        vrp = self.run_vrp_optimization(VRPOptimizationRequest())
        forecast = self.run_demand_forecast()
        inventory = self.run_inventory_optimization()
        sample_eta = self.run_eta_prediction(ETAPredictionRequest())
        anomalies = self.run_anomaly_detection(top_k=12)
        drift = self.run_drift_evaluation(drift_regime="nominal")
        pareto = self.run_pareto_frontier(demand_multiplier=1.0)

        return {
            "status": "operational",
            "startup_time_ms": self.startup_time_ms,
            "catalog": self.get_catalog_metadata(),
            "model_diagnostics": {
                "demand_forecaster": self.demand_forecaster.metrics,
                "demand_feature_importances": self.demand_forecaster.feature_importances,
                "eta_predictor": self.eta_predictor.metrics,
                "anomaly_detector": self.anomaly_detector.validation_metrics,
            },
            "vrp": vrp.model_dump(),
            "default_forecast": forecast.model_dump(),
            "inventory": inventory.model_dump(),
            "sample_eta": sample_eta.model_dump(),
            "anomalies": anomalies.model_dump(),
            "mlops_drift": drift.model_dump(),
            "pareto_frontier": pareto.model_dump(),
        }
