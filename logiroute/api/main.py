"""
FastAPI Application Server for LogiRoute AI Supply Chain & Logistics Platform.

Exposes RESTful endpoints for:
- /api/v1/health              : Liveness & model readiness probe
- /api/v1/overview            : Unified Control Tower state & KPIs
- /api/v1/catalog             : Network hubs, SKUs, and fleet vehicle classes
- /api/v1/routes/optimize     : CVRPTW multi-stop route optimization
- /api/v1/forecast            : Conformalized Quantile Demand Forecasting (P10/P50/P90)
- /api/v1/eta/predict         : Traffic & Weather-Aware ETA & SLA Delay Risk inference
- /api/v1/inventory/optimize  : Multi-Echelon Stochastic (s, S) Inventory Optimization
- /api/v1/scenarios/simulate  : End-to-end Supply Chain Disruption Stress-Testing
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from logiroute import __version__
from logiroute.pipeline.orchestrator import ControlTowerOrchestrator
from logiroute.schemas import (
    DemandForecastResponse,
    ETAPredictionRequest,
    ETAPredictionResponse,
    InventoryOptimizationResponse,
    ScenarioSimulationRequest,
    ScenarioSimulationResponse,
    VRPOptimizationRequest,
    VRPOptimizationResponse,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
orchestrator = ControlTowerOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not orchestrator.is_ready:
        orchestrator.initialize()
    yield


app = FastAPI(
    title="LogiRoute AI — Supply Chain & Logistics Control Tower API",
    description=(
        "Enterprise AI/ML & Operations Research Engine for Probabilistic Demand "
        "Forecasting (CQR Quantile GBDT), Capacitated Vehicle Routing with Time "
        "Windows (CVRPTW), Traffic-Aware ETA & SLA Delay Risk Prediction, and "
        "Multi-Echelon Stochastic Inventory Optimization."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard() -> Any:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>LogiRoute AI API is running. Visit /docs for OpenAPI.</h1>")


@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "logiroute-ai",
        "version": __version__,
        "models_initialized": orchestrator.is_ready,
        "startup_time_ms": orchestrator.startup_time_ms,
    }


@app.get("/api/v1/catalog")
def get_catalog() -> Dict[str, Any]:
    return orchestrator.get_catalog_metadata()


@app.get("/api/v1/overview")
def get_overview() -> Dict[str, Any]:
    return orchestrator.get_control_tower_overview()


@app.post("/api/v1/routes/optimize", response_model=VRPOptimizationResponse)
def optimize_routes(req: VRPOptimizationRequest) -> VRPOptimizationResponse:
    return orchestrator.run_vrp_optimization(req)


@app.get("/api/v1/forecast", response_model=DemandForecastResponse)
def get_demand_forecast(
    sku_id: str = Query(default="SKU-PHM-101", description="SKU identifier"),
    hub_id: str = Query(default="NODE-SLT-01", description="Distribution hub identifier"),
    horizon_days: int = Query(default=14, ge=3, le=30, description="Forecast horizon in days"),
    demand_multiplier: float = Query(
        default=1.0, ge=0.4, le=3.0, description="Scenario demand scaling factor"
    ),
) -> DemandForecastResponse:
    return orchestrator.run_demand_forecast(
        sku_id=sku_id,
        hub_id=hub_id,
        horizon_days=horizon_days,
        demand_multiplier=demand_multiplier,
    )


@app.post("/api/v1/eta/predict", response_model=ETAPredictionResponse)
def predict_eta(req: ETAPredictionRequest) -> ETAPredictionResponse:
    return orchestrator.run_eta_prediction(req)


@app.get("/api/v1/inventory/optimize", response_model=InventoryOptimizationResponse)
def optimize_inventory(
    target_service_level: float = Query(
        default=0.96, ge=0.80, le=0.999, description="Target cycle service level (0.80 - 0.999)"
    ),
    demand_multiplier: float = Query(
        default=1.0, ge=0.4, le=3.0, description="Scenario demand multiplier"
    ),
    lead_time_shock_multiplier: float = Query(
        default=1.0, ge=0.5, le=3.0, description="Supplier lead-time shock factor"
    ),
) -> InventoryOptimizationResponse:
    return orchestrator.run_inventory_optimization(
        target_service_level=target_service_level,
        demand_multiplier=demand_multiplier,
        lead_time_shock_multiplier=lead_time_shock_multiplier,
    )


@app.post("/api/v1/scenarios/simulate", response_model=ScenarioSimulationResponse)
def simulate_scenario(req: ScenarioSimulationRequest) -> ScenarioSimulationResponse:
    return orchestrator.run_scenario_simulation(req)
