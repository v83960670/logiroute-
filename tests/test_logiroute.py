"""
Automated Unit & Integration Test Suite for LogiRoute AI.

Covers:
1. Synthetic Supply Chain & Telemetry Data Generation
2. Conformalized Quantile Demand Forecaster (P10/P50/P90 monotonicity & coverage)
3. Traffic & Weather-Aware ETA & SLA Delay Risk Predictor
4. CVRPTW Combinatorial Route Optimizer (tour validity, capacity bounds, 2-Opt gain)
5. Multi-Echelon Stochastic Inventory Optimizer ((s, S) policy invariants)
6. FastAPI REST API & Control Tower Endpoints
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from logiroute.api.main import app
from logiroute.data.synthetic_generator import SupplyChainDataGenerator, haversine_km
from logiroute.models.demand_forecaster import ProbabilisticDemandForecaster
from logiroute.models.eta_predictor import ETADelayRiskPredictor
from logiroute.optimization.inventory_optimizer import StochasticInventoryOptimizer
from logiroute.optimization.vrp_solver import CVRPTWSolver
from logiroute.schemas import ETAPredictionRequest


@pytest.fixture(scope="module")
def generator() -> SupplyChainDataGenerator:
    return SupplyChainDataGenerator(seed=42)


@pytest.fixture(scope="module")
def trained_forecaster(generator: SupplyChainDataGenerator) -> ProbabilisticDemandForecaster:
    df = generator.generate_demand_history(days=240)
    forecaster = ProbabilisticDemandForecaster(random_state=42)
    forecaster.fit(df, holdout_days=24)
    return forecaster


@pytest.fixture(scope="module")
def trained_eta_predictor(generator: SupplyChainDataGenerator) -> ETADelayRiskPredictor:
    df = generator.generate_shipment_telemetry(n_samples=1800)
    predictor = ETADelayRiskPredictor(random_state=42)
    predictor.fit(df)
    return predictor


def test_haversine_distance_sanity() -> None:
    # Dankuni to Salt Lake is roughly ~18 km great-circle
    d = haversine_km(22.6764, 88.2924, 22.5769, 88.4337)
    assert 14.0 <= d <= 24.0


def test_synthetic_data_generation(generator: SupplyChainDataGenerator) -> None:
    demand_df = generator.generate_demand_history(days=90)
    assert not demand_df.empty
    assert (demand_df["demand_units"] > 0).all()
    assert {"sku_id", "hub_id", "demand_units", "promo_active", "weather_severity"}.issubset(
        demand_df.columns
    )

    telemetry_df = generator.generate_shipment_telemetry(n_samples=400)
    assert len(telemetry_df) == 400
    assert (telemetry_df["actual_duration_min"] >= 12.0).all()
    assert set(telemetry_df["sla_breached"].unique()).issubset({0, 1})

    nodes = generator.generate_daily_dispatch_nodes()
    assert len(nodes) == 16
    assert sum(1 for n in nodes if n.is_depot) == 1
    assert sum(1 for n in nodes if not n.is_depot) == 15


def test_probabilistic_demand_forecaster(
    trained_forecaster: ProbabilisticDemandForecaster,
) -> None:
    assert trained_forecaster.is_fitted
    metrics = trained_forecaster.metrics
    assert metrics["wmape_pct"] < 12.0
    assert metrics["r2_score"] > 0.90
    assert 68.0 <= metrics["interval_80_coverage_pct"] <= 95.0

    res = trained_forecaster.forecast_sku_hub(
        sku_id="SKU-PHM-101", hub_id="NODE-SLT-01", horizon_days=14
    )
    assert res.horizon_days == 14
    future_pts = [pt for pt in res.series if not pt.is_historical]
    assert len(future_pts) == 14

    # Verify non-crossing quantile invariant: P10 <= P50 <= P90 everywhere
    for pt in res.series:
        assert pt.p10_demand <= pt.p50_demand <= pt.p90_demand


def test_eta_delay_risk_predictor(
    trained_eta_predictor: ETADelayRiskPredictor,
) -> None:
    assert trained_eta_predictor.is_fitted
    m = trained_eta_predictor.metrics
    assert m["eta_r2_score"] > 0.95
    assert m["sla_roc_auc"] > 0.95

    low_friction = trained_eta_predictor.predict_shipment_eta(
        ETAPredictionRequest(
            origin_hub_id="DEPOT-KOL-01",
            destination_hub_id="NODE-SLT-01",
            traffic_congestion_index=0.10,
            weather_severity_index=0.05,
            promised_sla_minutes=70.0,
        )
    )
    high_friction = trained_eta_predictor.predict_shipment_eta(
        ETAPredictionRequest(
            origin_hub_id="DEPOT-KOL-01",
            destination_hub_id="NODE-SLT-01",
            traffic_congestion_index=0.90,
            weather_severity_index=0.85,
            promised_sla_minutes=70.0,
        )
    )

    assert high_friction.predicted_eta_min > low_friction.predicted_eta_min
    assert high_friction.delay_risk_probability > low_friction.delay_risk_probability
    assert high_friction.predicted_eta_p90_min >= high_friction.predicted_eta_min
    assert len(high_friction.factor_contributions_min) == 4


def test_cvrptw_solver_invariants(generator: SupplyChainDataGenerator) -> None:
    nodes = generator.generate_daily_dispatch_nodes(demand_multiplier=1.0)
    solver = CVRPTWSolver()

    res_opt = solver.solve(nodes=nodes, enable_two_opt=True)
    res_cw_only = solver.solve(nodes=nodes, enable_two_opt=False)

    # 1. Optimized distance must significantly beat baseline and be <= CW construction
    assert res_opt.benchmark.optimized_distance_km < res_opt.benchmark.baseline_distance_km
    assert res_opt.benchmark.distance_reduction_pct > 25.0
    assert res_opt.benchmark.optimized_distance_km <= res_cw_only.benchmark.optimized_distance_km + 1e-4

    # 2. Every non-depot customer must be visited exactly once
    expected_customers = {n.node_id for n in nodes if not n.is_depot}
    visited_customers = []
    for route in res_opt.routes:
        assert route.total_load_kg <= route.capacity_kg + 1e-5
        assert route.stops[0].is_depot
        assert route.stops[-1].is_depot
        for stop in route.stops:
            if not stop.is_depot:
                visited_customers.append(stop.node_id)

    assert len(visited_customers) == len(expected_customers)
    assert set(visited_customers) == expected_customers


def test_stochastic_inventory_optimizer(
    trained_forecaster: ProbabilisticDemandForecaster,
) -> None:
    stats = trained_forecaster.get_sku_hub_demand_stats()
    inv_opt = StochasticInventoryOptimizer(seed=42)

    res_90 = inv_opt.optimize_network_inventory(stats, target_service_level=0.90)
    res_99 = inv_opt.optimize_network_inventory(stats, target_service_level=0.99)

    assert res_90.total_skus_evaluated == len(stats)
    for p90, p99 in zip(
        sorted(res_90.policies, key=lambda x: (x.sku_id, x.hub_id)),
        sorted(res_99.policies, key=lambda x: (x.sku_id, x.hub_id)),
    ):
        assert p99.safety_stock_units > p90.safety_stock_units
        assert p99.reorder_point_units > p90.reorder_point_units
        assert p99.order_up_to_level_s == p99.reorder_point_units + p99.economic_order_qty_units


def test_fastapi_endpoints() -> None:
    client = TestClient(app)

    # Health check
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # Dashboard HTML
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "LOGIROUTE" in r_root.text

    # Overview endpoint
    r_ov = client.get("/api/v1/overview")
    assert r_ov.status_code == 200
    ov_data = r_ov.json()
    assert "vrp" in ov_data and "inventory" in ov_data

    # VRP Optimization endpoint
    r_vrp = client.post(
        "/api/v1/routes/optimize",
        json={
            "demand_multiplier": 1.15,
            "traffic_congestion_index": 0.50,
            "weather_severity_index": 0.25,
            "vehicle_capacity_scale": 1.0,
            "enable_two_opt": True,
        },
    )
    assert r_vrp.status_code == 200
    assert r_vrp.json()["benchmark"]["vehicles_dispatched"] >= 1

    # Demand Forecast endpoint
    r_fc = client.get("/api/v1/forecast?sku_id=SKU-PHM-101&hub_id=NODE-SLT-01&horizon_days=10")
    assert r_fc.status_code == 200
    assert r_fc.json()["horizon_days"] == 10

    # ETA Prediction endpoint
    r_eta = client.post(
        "/api/v1/eta/predict",
        json={
            "origin_hub_id": "DEPOT-KOL-01",
            "destination_hub_id": "NODE-SLT-01",
            "departure_hour": 9.0,
            "day_of_week": 1,
            "traffic_congestion_index": 0.65,
            "weather_severity_index": 0.40,
            "payload_weight_kg": 3200.0,
            "promised_sla_minutes": 60.0,
        },
    )
    assert r_eta.status_code == 200
    assert r_eta.json()["predicted_eta_min"] > 0

    # Scenario Simulation endpoint
    r_sim = client.post(
        "/api/v1/scenarios/simulate",
        json={
            "scenario_name": "Monsoon Surge Test",
            "demand_multiplier": 1.30,
            "traffic_congestion_index": 0.75,
            "weather_severity_index": 0.70,
            "lead_time_shock_multiplier": 1.40,
            "target_service_level": 0.96,
        },
    )
    assert r_sim.status_code == 200
    assert 0.0 <= r_sim.json()["resilience_score"] <= 100.0
