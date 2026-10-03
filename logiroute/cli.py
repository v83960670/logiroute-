"""
Command-Line Interface (CLI) for LogiRoute AI MLOps & Optimization Pipelines.

Usage:
    python -m logiroute.cli --evaluate
    python -m logiroute.cli --simulate "Monsoon Surge" --demand-mult 1.35 --weather 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from logiroute.pipeline.orchestrator import ControlTowerOrchestrator
from logiroute.schemas import ScenarioSimulationRequest, VRPOptimizationRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logiroute",
        description="LogiRoute AI — Supply Chain & Logistics ML/OR Pipeline CLI",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Train all ML models, run CVRPTW & Inventory solvers, and print benchmark metrics.",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        default=None,
        help="Run a named disruption stress-test scenario.",
    )
    parser.add_argument(
        "--demand-mult",
        type=float,
        default=1.25,
        help="Demand multiplier for simulation (default: 1.25).",
    )
    parser.add_argument(
        "--traffic",
        type=float,
        default=0.60,
        help="Traffic congestion index [0..1] (default: 0.60).",
    )
    parser.add_argument(
        "--weather",
        type=float,
        default=0.45,
        help="Weather severity index [0..1] (default: 0.45).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output as formatted JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    orchestrator = ControlTowerOrchestrator()
    init_summary = orchestrator.initialize()

    if args.simulate:
        sim_req = ScenarioSimulationRequest(
            scenario_name=args.simulate,
            demand_multiplier=args.demand_mult,
            traffic_congestion_index=args.traffic,
            weather_severity_index=args.weather,
        )
        sim_res = orchestrator.run_scenario_simulation(sim_req)
        if args.json:
            print(json.dumps(sim_res.model_dump(), indent=2))
        else:
            print("=" * 72)
            print(f"LOGIROUTE AI — SCENARIO SIMULATION: {sim_res.scenario_name}")
            print("=" * 72)
            print(f"Resilience Score       : {sim_res.resilience_score:.1f} / 100")
            print(f"Network SLA Risk       : {sim_res.network_sla_risk_pct:.1f}%")
            print(
                f"Optimized Fleet Dist   : {sim_res.routing_summary.optimized_distance_km:.1f} km "
                f"(-{sim_res.routing_summary.distance_reduction_pct:.1f}% vs baseline)"
            )
            print(
                f"Cost Saved             : ${sim_res.routing_summary.cost_saved_usd:,.2f} "
                f"(-{sim_res.routing_summary.cost_reduction_pct:.1f}%)"
            )
            print("\nExecutive Recommendations:")
            for idx, rec in enumerate(sim_res.executive_recommendations, 1):
                print(f"  {idx}. {rec}")
        return 0

    # Default / --evaluate report
    vrp_res = orchestrator.run_vrp_optimization(VRPOptimizationRequest())
    inv_res = orchestrator.run_inventory_optimization()

    report = {
        "initialization": init_summary,
        "vrp_benchmark": vrp_res.benchmark.model_dump(),
        "inventory_summary": {
            "target_service_level_pct": inv_res.target_service_level_pct,
            "total_skus_evaluated": inv_res.total_skus_evaluated,
            "critical_stockout_alerts": inv_res.critical_stockout_alerts,
            "reorder_triggered_count": inv_res.reorder_triggered_count,
            "total_recommended_po_value_usd": inv_res.total_recommended_po_value_usd,
            "total_annual_inventory_cost_usd": inv_res.total_annual_inventory_cost_usd,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        dm = init_summary["demand_forecaster_metrics"]
        em = init_summary["eta_predictor_metrics"]
        vb = vrp_res.benchmark
        print("=" * 76)
        print("LOGIROUTE AI — ENTERPRISE SUPPLY CHAIN & LOGISTICS ML BENCHMARK REPORT")
        print("=" * 76)
        print(f"Pipeline Initialization Time : {init_summary['startup_time_ms']:.1f} ms\n")
        print("[1] Probabilistic Demand Forecaster (Quantile GBDT: P10 / P50 / P90)")
        print(f"    - Holdout WMAPE          : {dm['wmape_pct']:.2f}%")
        print(f"    - Holdout MAPE           : {dm['mape_pct']:.2f}%")
        print(f"    - Holdout RMSE           : {dm['rmse_units']:.2f} units")
        print(f"    - Holdout R^2 Score      : {dm['r2_score']:.4f}")
        print(f"    - P10-P90 Coverage       : {dm['interval_80_coverage_pct']:.2f}%\n")
        print("[2] Traffic & Weather-Aware ETA & SLA Delay Risk Model")
        print(f"    - ETA Holdout MAE        : {em['eta_mae_min']:.2f} min")
        print(f"    - ETA Holdout RMSE       : {em['eta_rmse_min']:.2f} min")
        print(f"    - ETA R^2 Score          : {em['eta_r2_score']:.4f}")
        print(f"    - SLA Breach ROC-AUC     : {em['sla_roc_auc']:.4f}")
        print(f"    - SLA Brier Calibration  : {em['sla_brier_score']:.4f}\n")
        print("[3] Capacitated Vehicle Routing with Time Windows (CVRPTW)")
        print(f"    - Algorithm              : {vb.solver_algorithm}")
        print(f"    - Baseline Distance      : {vb.baseline_distance_km:.2f} km")
        print(f"    - Optimized Distance     : {vb.optimized_distance_km:.2f} km (-{vb.distance_reduction_pct:.1f}%)")
        print(f"    - Daily Cost Saved       : ${vb.cost_saved_usd:,.2f} (-{vb.cost_reduction_pct:.1f}%)")
        print(f"    - Carbon Saved           : {vb.co2_saved_kg:.2f} kg CO2e")
        print(f"    - Fleet Utilization      : {vb.total_fleet_utilization_pct:.1f}% across {vb.vehicles_dispatched} vehicles\n")
        print("[4] Multi-Echelon Stochastic Inventory Optimization ((s, S) Policy)")
        print(f"    - Target Service Level   : {inv_res.target_service_level_pct:.1f}%")
        print(f"    - SKU x Hub Nodes        : {inv_res.total_skus_evaluated}")
        print(f"    - Reorder Triggered      : {inv_res.reorder_triggered_count} ({inv_res.critical_stockout_alerts} critical)")
        print(f"    - Recommended PO Value   : ${inv_res.total_recommended_po_value_usd:,.2f}")
        print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
