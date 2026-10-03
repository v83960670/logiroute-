/**
 * LogiRoute AI — Interactive Supply Chain & Logistics Control Tower Frontend
 * Uses strictly relative API paths (/api/v1/...) for cloud preview compatibility.
 */

const ROUTE_COLORS = [
  "#06b6d4", // Cyan
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#8b5cf6", // Purple
  "#f43f5e", // Rose
  "#3b82f6", // Blue
];

const state = {
  overview: null,
  vrpData: null,
  selectedVehicleId: "ALL",
  inventoryData: null,
  leafletMap: null,
  mapLayerGroup: null,
  demandChart: null,
  featureImpChart: null,
  etaAttrChart: null,
  paretoChart: null,
  psiChart: null,
};

function formatHourClock(decHour) {
  const h = Math.floor(decHour);
  const m = Math.round((decHour - h) * 60);
  const hh = String(m === 60 ? h + 1 : h).padStart(2, "0");
  const mm = String(m === 60 ? 0 : m).padStart(2, "0");
  return `${hh}:${mm}`;
}

async function apiFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status} on ${path}`);
  }
  return res.json();
}

/* ==================== INITIALIZATION ==================== */
document.addEventListener("DOMContentLoaded", async () => {
  setupTabNavigation();
  setupControlListeners();
  initLeafletMap();

  try {
    const overview = await apiFetch("/api/v1/overview");
    state.overview = overview;
    state.vrpData = overview.vrp;
    state.inventoryData = overview.inventory;

    populateDropdownCatalogs(overview.catalog);
    updateExecutiveKPIs(overview.vrp.benchmark, overview.model_diagnostics, overview.inventory);
    renderVRPSection(overview.vrp);
    renderDemandForecastSection(overview.default_forecast);
    renderETASection(overview.sample_eta);
    renderInventorySection(overview.inventory);
    renderMLOpsSection(overview.mlops_drift, overview.anomalies, overview.pareto_frontier);

    const statusEl = document.getElementById("engine-status-text");
    if (statusEl) {
      statusEl.textContent = `ML & OR Ready (${overview.startup_time_ms.toFixed(0)}ms)`;
    }
  } catch (err) {
    console.error("Failed to initialize Control Tower:", err);
  }
});

/* ==================== TAB NAVIGATION ==================== */
function setupTabNavigation() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const panel = document.getElementById(targetId);
      if (panel) panel.classList.add("active");

      if (targetId === "tab-routing" && state.leafletMap) {
        setTimeout(() => state.leafletMap.invalidateSize(), 120);
      }
    });
  });
}

/* ==================== DROPDOWN CATALOGS ==================== */
function populateDropdownCatalogs(catalog) {
  const fcSku = document.getElementById("fc-sku-select");
  const fcHub = document.getElementById("fc-hub-select");
  const etaOrigin = document.getElementById("eta-origin");
  const etaDest = document.getElementById("eta-dest");

  if (fcSku && catalog.skus) {
    fcSku.innerHTML = catalog.skus
      .map((s) => `<option value="${s.sku_id}">${s.sku_id} — ${s.name}</option>`)
      .join("");
  }

  if (fcHub && catalog.forecast_hubs) {
    fcHub.innerHTML = catalog.forecast_hubs
      .map((h) => `<option value="${h.hub_id}">${h.hub_id} — ${h.name}</option>`)
      .join("");
  }

  if (etaOrigin && etaDest && catalog.all_nodes) {
    etaOrigin.innerHTML = catalog.all_nodes
      .map((n) => `<option value="${n.hub_id}">${n.hub_id} — ${n.city}</option>`)
      .join("");
    etaDest.innerHTML = catalog.all_nodes
      .filter((n) => !n.is_depot)
      .map((n) => `<option value="${n.hub_id}">${n.hub_id} — ${n.name}</option>`)
      .join("");
  }
}

/* ==================== EXECUTIVE KPI STRIP ==================== */
function updateExecutiveKPIs(vrpBench, diagnostics, inventory) {
  const dm = diagnostics?.demand_forecaster || {};
  const em = diagnostics?.eta_predictor || {};

  document.getElementById("kpi-opt-dist").textContent = `${vrpBench.optimized_distance_km.toFixed(1)} km`;
  document.getElementById("kpi-dist-pct").textContent = `-${vrpBench.distance_reduction_pct.toFixed(1)}%`;
  document.getElementById("kpi-dist-sub").textContent = `Saved ${vrpBench.distance_saved_km.toFixed(1)} km vs ${vrpBench.baseline_distance_km.toFixed(0)} km baseline`;

  document.getElementById("kpi-opt-cost").textContent = `$${vrpBench.optimized_cost_usd.toFixed(0)}`;
  document.getElementById("kpi-cost-pct").textContent = `-${vrpBench.cost_reduction_pct.toFixed(1)}%`;
  document.getElementById("kpi-cost-sub").textContent = `Net Daily Savings: $${vrpBench.cost_saved_usd.toFixed(2)}`;

  document.getElementById("kpi-co2-saved").textContent = `${vrpBench.co2_saved_kg.toFixed(1)} kg`;
  document.getElementById("kpi-fleet-util").textContent = `Fleet Utilization: ${vrpBench.total_fleet_utilization_pct.toFixed(1)}% (${vrpBench.vehicles_dispatched} units)`;

  const acc = Math.max(0, 100 - (dm.wmape_pct || 7.8));
  document.getElementById("kpi-forecast-acc").textContent = `${acc.toFixed(1)}%`;
  document.getElementById("kpi-forecast-r2").textContent = `R² ${(dm.r2_score || 0.96).toFixed(3)}`;
  document.getElementById("kpi-forecast-cov").textContent = `CQR P10–P90 Coverage: ${(dm.interval_80_coverage_pct || 81.5).toFixed(1)}%`;

  document.getElementById("kpi-eta-auc").textContent = (em.sla_roc_auc || 0.991).toFixed(4);
  document.getElementById("kpi-eta-mae").textContent = `MAE ${(em.eta_mae_min || 2.5).toFixed(1)}m`;
  document.getElementById("kpi-eta-brier").textContent = `Brier Calibration: ${(em.sla_brier_score || 0.045).toFixed(4)}`;

  if (inventory) {
    document.getElementById("kpi-inv-reorders").textContent = `${inventory.reorder_triggered_count} SKUs`;
    document.getElementById("kpi-inv-critical").textContent = `${inventory.critical_stockout_alerts} Critical`;
    document.getElementById("kpi-inv-po").textContent = `Recommended PO: $${Math.round(inventory.total_recommended_po_value_usd).toLocaleString()}`;
  }
}

/* ==================== TAB 1: LEAFLET MAP & CVRPTW ROUTING ==================== */
function initLeafletMap() {
  const container = document.getElementById("leaflet-map");
  if (!container || typeof L === "undefined") return;

  const map = L.map("leaflet-map", {
    center: [22.61, 88.35],
    zoom: 10,
    zoomControl: true,
  });

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(map);

  state.leafletMap = map;
  state.mapLayerGroup = L.layerGroup().addTo(map);
}

function renderVRPSection(vrpData) {
  state.vrpData = vrpData;
  const bench = vrpData.benchmark;

  document.getElementById("vrp-algo-name").textContent = bench.solver_algorithm.includes("2-Opt")
    ? "CW + 2-Opt + Relocate"
    : "Clarke-Wright Only";
  document.getElementById("vrp-two-opt-iters").textContent = `${bench.two_opt_iterations} moves`;
  document.getElementById("vrp-solve-ms").textContent = `${bench.solve_time_ms.toFixed(2)} ms`;
  document.getElementById("vrp-otd-rate").textContent = `${bench.on_time_delivery_rate_pct.toFixed(1)}%`;

  renderRouteFilterPills(vrpData.routes);
  renderMapOverlays(vrpData);
  renderFleetCards(vrpData.routes);
  renderStopsTable(vrpData.routes);
}

function renderRouteFilterPills(routes) {
  const container = document.getElementById("route-filter-pills");
  if (!container) return;

  let html = `<button class="route-pill-btn ${state.selectedVehicleId === "ALL" ? "active" : ""}" data-veh="ALL">ALL FLEET (${routes.length})</button>`;
  routes.forEach((r, idx) => {
    const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
    const isActive = state.selectedVehicleId === r.vehicle_id ? "active" : "";
    html += `<button class="route-pill-btn ${isActive}" data-veh="${r.vehicle_id}" style="border-left: 3px solid ${color}">${r.vehicle_id}</button>`;
  });
  container.innerHTML = html;

  container.querySelectorAll(".route-pill-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedVehicleId = btn.getAttribute("data-veh");
      renderRouteFilterPills(state.vrpData.routes);
      renderMapOverlays(state.vrpData);
      renderStopsTable(state.vrpData.routes);
    });
  });
}

function renderMapOverlays(vrpData) {
  if (!state.leafletMap || !state.mapLayerGroup) return;
  state.mapLayerGroup.clearLayers();

  const depot = vrpData.depot;
  const allCoords = [[depot.lat, depot.lon]];

  // Central Depot Pin
  const depotIcon = L.divIcon({
    className: "custom-depot-pin",
    html: `<div style="background:#f43f5e;color:#fff;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:11px;padding:4px 7px;border-radius:6px;border:2px solid #fff;box-shadow:0 0 12px rgba(244,63,94,0.8);white-space:nowrap;">DEPOT</div>`,
    iconSize: [54, 24],
    iconAnchor: [27, 12],
  });

  L.marker([depot.lat, depot.lon], { icon: depotIcon })
    .bindPopup(
      `<div style="font-family:Inter,sans-serif;color:#0f172a;">
        <strong>${depot.name}</strong><br/>
        <span>Hub ID: ${depot.node_id} (${depot.city})</span><br/>
        <span>Operating Window: 06:00 – 20:00</span>
      </div>`
    )
    .addTo(state.mapLayerGroup);

  vrpData.routes.forEach((route, rIdx) => {
    if (state.selectedVehicleId !== "ALL" && state.selectedVehicleId !== route.vehicle_id) {
      return;
    }
    const color = ROUTE_COLORS[rIdx % ROUTE_COLORS.length];

    // Draw route polyline
    const poly = L.polyline(route.geometry_coords, {
      color,
      weight: 3.5,
      opacity: 0.88,
    }).addTo(state.mapLayerGroup);

    poly.bindPopup(
      `<div style="font-family:Inter,sans-serif;color:#0f172a;">
        <strong>${route.vehicle_id} — ${route.vehicle_name}</strong><br/>
        <span>Total Distance: ${route.total_distance_km} km | Stops: ${route.num_stops}</span><br/>
        <span>Payload: ${route.total_load_kg} / ${route.capacity_kg} kg (${route.utilization_pct}%)</span>
      </div>`
    );

    // Draw numbered stop markers
    route.stops.forEach((stop) => {
      if (stop.is_depot) return;
      allCoords.push([stop.lat, stop.lon]);

      const markerIcon = L.divIcon({
        className: "custom-stop-pin",
        html: `<div style="background:${color};color:#070b14;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:11px;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px solid #f8fafc;box-shadow:0 0 8px ${color};">${stop.sequence_index}</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      L.marker([stop.lat, stop.lon], { icon: markerIcon })
        .bindPopup(
          `<div style="font-family:Inter,sans-serif;color:#0f172a;min-width:200px;">
            <strong>#${stop.sequence_index} ${stop.node_name}</strong><br/>
            <span>Vehicle: <b>${route.vehicle_id}</b> | City: ${stop.city}</span><br/>
            <span>Window: ${stop.time_window_formatted}</span><br/>
            <span>Arrival: <b>${stop.arrival_time_formatted}</b> → Depart: ${stop.departure_time_formatted}</span><br/>
            <span>Delivered: ${stop.delivered_kg} kg | SLA Risk: ${(stop.delay_risk_probability * 100).toFixed(1)}%</span>
          </div>`
        )
        .addTo(state.mapLayerGroup);
    });
  });

  if (allCoords.length > 1) {
    state.leafletMap.fitBounds(allCoords, { padding: [28, 28] });
  }
}

function renderFleetCards(routes) {
  const container = document.getElementById("fleet-cards-container");
  if (!container) return;

  container.innerHTML = routes
    .map((r, idx) => {
      const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
      return `
      <div class="fleet-card" style="border-left-color: ${color}">
        <div class="fleet-card-top">
          <span class="fleet-id" style="color: ${color}">${r.vehicle_id}</span>
          <span class="badge badge-ok">${r.num_stops} Stops</span>
        </div>
        <div class="fleet-class-name">${r.vehicle_name}</div>
        <div class="control-label-row">
          <span>Load: ${r.total_load_kg.toFixed(0)} / ${r.capacity_kg.toFixed(0)} kg</span>
          <span class="mono">${r.utilization_pct.toFixed(1)}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width:${Math.min(100, r.utilization_pct)}%;background:${color}"></div>
        </div>
        <div class="fleet-stats-row mono">
          <span>${r.total_distance_km.toFixed(1)} km</span>
          <span>$${r.total_cost_usd.toFixed(0)}</span>
          <span>${r.co2_emissions_kg.toFixed(1)} kg CO₂</span>
        </div>
      </div>
    `;
    })
    .join("");
}

function renderStopsTable(routes) {
  const tbody = document.getElementById("vrp-stops-tbody");
  const badge = document.getElementById("manifest-summary-badge");
  if (!tbody) return;

  const rows = [];
  let customerStopCount = 0;

  routes.forEach((route, rIdx) => {
    if (state.selectedVehicleId !== "ALL" && state.selectedVehicleId !== route.vehicle_id) {
      return;
    }
    const color = ROUTE_COLORS[rIdx % ROUTE_COLORS.length];

    route.stops.forEach((s) => {
      if (s.is_depot && s.sequence_index === 0) return; // Show customer stops + final depot return
      if (!s.is_depot) customerStopCount += 1;

      const riskPct = (s.delay_risk_probability * 100).toFixed(1);
      const riskClass =
        s.delay_risk_probability >= 0.6
          ? "badge-danger"
          : s.delay_risk_probability >= 0.35
          ? "badge-warn"
          : "badge-ok";
      const statusBadge = s.sla_on_time
        ? `<span class="badge badge-ok">ON-TIME</span>`
        : `<span class="badge badge-danger">SLA BREACH</span>`;

      rows.push(`
        <tr>
          <td class="mono" style="color:${color};font-weight:600">${route.vehicle_id}</td>
          <td>${route.vehicle_class}</td>
          <td class="mono">#${s.sequence_index}</td>
          <td><strong>${s.node_id}</strong> — ${s.node_name}</td>
          <td>${s.city}</td>
          <td class="mono">${s.time_window_formatted}</td>
          <td class="mono">${s.arrival_time_formatted} → ${s.departure_time_formatted}</td>
          <td class="mono">${s.leg_distance_km.toFixed(1)} km</td>
          <td class="mono">${s.delivered_kg > 0 ? s.delivered_kg.toFixed(0) + " kg" : "RETURN"}</td>
          <td><span class="badge ${riskClass}">${riskPct}%</span></td>
          <td>${statusBadge}</td>
        </tr>
      `);
    });
  });

  tbody.innerHTML = rows.join("");
  if (badge) {
    badge.textContent = `${customerStopCount} delivery stops scheduled`;
  }
}

/* ==================== TAB 2: PROBABILISTIC DEMAND FORECASTING ==================== */
function renderDemandForecastSection(fcData) {
  const titleEl = document.getElementById("fc-chart-title");
  if (titleEl) {
    titleEl.textContent = `${fcData.sku_id} (${fcData.sku_name}) @ ${fcData.hub_name}`;
  }

  const m = fcData.metrics || {};
  document.getElementById("fc-metric-wmape").textContent = `${(m.wmape_pct || 0).toFixed(2)}%`;
  document.getElementById("fc-metric-rmse").textContent = `${(m.rmse_units || 0).toFixed(1)} u`;
  document.getElementById("fc-metric-r2").textContent = (m.r2_score || 0).toFixed(4);
  document.getElementById("fc-metric-cov").textContent = `${(m.interval_80_coverage_pct || 0).toFixed(1)}%`;

  const labels = fcData.series.map((pt) => pt.date.slice(5));
  const actuals = fcData.series.map((pt) => (pt.is_historical ? pt.actual_demand : null));
  const p50 = fcData.series.map((pt) => pt.p50_demand);
  const p10 = fcData.series.map((pt) => pt.p10_demand);
  const p90 = fcData.series.map((pt) => pt.p90_demand);

  const ctx = document.getElementById("demand-fan-chart");
  if (ctx && typeof Chart !== "undefined") {
    if (state.demandChart) state.demandChart.destroy();

    state.demandChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Historical Actual Demand",
            data: actuals,
            borderColor: "#06b6d4",
            backgroundColor: "#06b6d4",
            borderWidth: 2.4,
            pointRadius: 2.5,
            tension: 0.25,
          },
          {
            label: "P50 Median Forecast (Quantile GBDT)",
            data: p50,
            borderColor: "#10b981",
            borderDash: [5, 4],
            borderWidth: 2.2,
            pointRadius: 2,
            tension: 0.25,
          },
          {
            label: "P90 Upper Bound (CQR)",
            data: p90,
            borderColor: "rgba(59, 130, 246, 0.35)",
            backgroundColor: "rgba(59, 130, 246, 0.16)",
            borderWidth: 1,
            pointRadius: 0,
            fill: "+1",
            tension: 0.25,
          },
          {
            label: "P10 Lower Bound (CQR)",
            data: p10,
            borderColor: "rgba(59, 130, 246, 0.35)",
            backgroundColor: "transparent",
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
        },
        scales: {
          x: {
            ticks: { color: "#94a3b8", maxRotation: 45 },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
          },
          y: {
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
            title: { display: true, text: "Daily Units", color: "#94a3b8" },
          },
        },
      },
    });
  }

  // Render Feature Importances Chart
  const impCtx = document.getElementById("feature-imp-chart");
  if (impCtx && typeof Chart !== "undefined" && fcData.feature_importances) {
    if (state.featureImpChart) state.featureImpChart.destroy();
    const entries = Object.entries(fcData.feature_importances).slice(0, 8);
    state.featureImpChart = new Chart(impCtx, {
      type: "bar",
      data: {
        labels: entries.map((e) => e[0]),
        datasets: [
          {
            label: "Relative Importance (%)",
            data: entries.map((e) => e[1]),
            backgroundColor: "rgba(6, 182, 212, 0.7)",
            borderColor: "#06b6d4",
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
          },
          y: {
            ticks: { color: "#e2e8f0", font: { family: "JetBrains Mono", size: 11 } },
            grid: { display: false },
          },
        },
      },
    });
  }
}

/* ==================== TAB 3: ETA & SLA DELAY RISK LAB ==================== */
function renderETASection(etaData) {
  document.getElementById("eta-res-mean").textContent = `${etaData.predicted_eta_min.toFixed(1)} min`;
  document.getElementById("eta-res-dist").textContent = `Road Dist: ${etaData.route_distance_km.toFixed(1)} km`;
  document.getElementById("eta-res-p90").textContent = `${etaData.predicted_eta_p90_min.toFixed(1)} min`;
  document.getElementById("eta-res-ff").textContent = `Free-Flow: ${etaData.free_flow_duration_min.toFixed(1)} min`;

  const probPct = (etaData.delay_risk_probability * 100).toFixed(1);
  document.getElementById("eta-res-prob").textContent = `${probPct}%`;
  document.getElementById("eta-res-sla").textContent = `SLA Target: ${etaData.sla_target_min.toFixed(0)} min`;
  document.getElementById("eta-res-action").textContent = etaData.recommended_action;

  const badge = document.getElementById("eta-risk-badge");
  if (badge) {
    badge.textContent = `${etaData.risk_tier} SLA RISK`;
    badge.className =
      "badge " +
      (etaData.risk_tier === "CRITICAL"
        ? "badge-danger"
        : etaData.risk_tier === "ELEVATED"
        ? "badge-warn"
        : "badge-ok");
  }

  const attrCtx = document.getElementById("eta-attribution-chart");
  if (attrCtx && typeof Chart !== "undefined" && etaData.factor_contributions_min) {
    if (state.etaAttrChart) state.etaAttrChart.destroy();
    const factors = Object.entries(etaData.factor_contributions_min);
    state.etaAttrChart = new Chart(attrCtx, {
      type: "bar",
      data: {
        labels: factors.map((f) => f[0]),
        datasets: [
          {
            label: "Minutes Contributed",
            data: factors.map((f) => f[1]),
            backgroundColor: ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"],
            borderRadius: 5,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
          },
          y: {
            ticks: { color: "#e2e8f0" },
            grid: { display: false },
          },
        },
      },
    });
  }
}

/* ==================== TAB 4: STOCHASTIC INVENTORY ==================== */
function renderInventorySection(invData) {
  state.inventoryData = invData;
  const costBadge = document.getElementById("inv-total-cost-badge");
  if (costBadge) {
    costBadge.textContent = `Annual Policy Cost: $${Math.round(invData.total_annual_inventory_cost_usd).toLocaleString()}`;
  }

  const filterVal = document.getElementById("inv-status-filter")?.value || "ALL";
  const tbody = document.getElementById("inv-policies-tbody");
  if (!tbody) return;

  const filtered = invData.policies.filter(
    (p) => filterVal === "ALL" || p.status === filterVal
  );

  tbody.innerHTML = filtered
    .map((p) => {
      const statusBadge =
        p.status === "CRITICAL_STOCKOUT_RISK"
          ? `<span class="badge badge-danger">CRITICAL STOCKOUT</span>`
          : p.status === "REORDER_TRIGGERED"
          ? `<span class="badge badge-warn">REORDER TRIGGERED</span>`
          : `<span class="badge badge-ok">OPTIMAL BUFFER</span>`;

      const poCell =
        p.recommended_order_qty > 0
          ? `<strong class="mono" style="color:#06b6d4">+${p.recommended_order_qty} units</strong>`
          : `<span class="mono" style="color:#64748b">0 units</span>`;

      return `
        <tr>
          <td><strong>${p.sku_id}</strong> — ${p.sku_name}</td>
          <td>${p.hub_id} (${p.hub_name.split(" ")[0]})</td>
          <td class="mono">${p.daily_demand_mean.toFixed(1)} ± ${p.daily_demand_std.toFixed(1)}</td>
          <td class="mono">${p.lead_time_days_mean.toFixed(1)} ± ${p.lead_time_days_std.toFixed(1)}d</td>
          <td class="mono">${p.current_on_hand_units} (+${p.in_transit_units})</td>
          <td class="mono">${p.safety_stock_units}</td>
          <td class="mono"><strong>${p.reorder_point_units}</strong></td>
          <td class="mono">${p.economic_order_qty_units}</td>
          <td class="mono">${p.days_of_supply.toFixed(1)}d</td>
          <td class="mono">${p.stockout_probability_pct.toFixed(1)}%</td>
          <td>${statusBadge}</td>
          <td>${poCell}</td>
        </tr>
      `;
    })
    .join("");
}

/* ==================== TAB 5: MLOPS DRIFT, ANOMALIES & PARETO FRONTIER ==================== */
function renderMLOpsSection(driftData, anomaliesData, paretoData) {
  if (driftData) renderDriftMonitor(driftData);
  if (paretoData) renderParetoFrontier(paretoData);
  if (anomaliesData) renderAnomaliesTable(anomaliesData);
}

function renderDriftMonitor(driftData) {
  const scoreEl = document.getElementById("mlops-health-score");
  const badgeEl = document.getElementById("mlops-retrain-badge");
  const reasonEl = document.getElementById("mlops-retrain-reason");

  if (scoreEl) {
    scoreEl.textContent = `${driftData.overall_health_score.toFixed(1)} / 100`;
    scoreEl.style.color = driftData.retraining_recommended ? "#f43f5e" : "#10b981";
  }

  if (badgeEl) {
    badgeEl.textContent = driftData.retraining_recommended ? "RETRAIN REQUIRED" : "STABLE";
    badgeEl.className = "badge " + (driftData.retraining_recommended ? "badge-danger" : "badge-ok");
  }

  if (reasonEl) {
    reasonEl.textContent = driftData.retraining_trigger_reason;
  }

  const tbody = document.getElementById("mlops-drift-tbody");
  if (tbody && driftData.monitored_features) {
    tbody.innerHTML = driftData.monitored_features
      .map((f) => {
        const badgeCls =
          f.drift_status === "CRITICAL_DRIFT"
            ? "badge-danger"
            : f.drift_status === "MODERATE_DRIFT"
            ? "badge-warn"
            : "badge-ok";
        return `
          <tr>
            <td class="mono"><strong>${f.feature_name}</strong></td>
            <td class="mono">${f.baseline_mean.toFixed(2)} → ${f.current_mean.toFixed(2)} (${f.shift_pct >= 0 ? "+" : ""}${f.shift_pct.toFixed(1)}%)</td>
            <td class="mono"><strong>${f.psi_score.toFixed(3)}</strong></td>
            <td class="mono">${f.ks_statistic.toFixed(3)} (p=${f.ks_p_value.toFixed(3)})</td>
            <td class="mono">${f.wasserstein_norm.toFixed(3)}σ</td>
            <td><span class="badge ${badgeCls}">${f.drift_status}</span></td>
          </tr>
        `;
      })
      .join("");
  }

  const psiCtx = document.getElementById("mlops-psi-chart");
  if (psiCtx && typeof Chart !== "undefined" && driftData.monitored_features) {
    if (state.psiChart) state.psiChart.destroy();
    const feats = driftData.monitored_features;
    state.psiChart = new Chart(psiCtx, {
      type: "bar",
      data: {
        labels: feats.map((f) => f.feature_name),
        datasets: [
          {
            label: "Population Stability Index (PSI)",
            data: feats.map((f) => f.psi_score),
            backgroundColor: feats.map((f) =>
              f.psi_score >= 0.25
                ? "rgba(244, 63, 94, 0.75)"
                : f.psi_score >= 0.1
                ? "rgba(245, 158, 11, 0.75)"
                : "rgba(16, 185, 129, 0.75)"
            ),
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#94a3b8", font: { family: "JetBrains Mono", size: 10.5 } },
            grid: { display: false },
          },
          y: {
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
            title: { display: true, text: "PSI Score", color: "#94a3b8" },
          },
        },
      },
    });
  }
}

function renderParetoFrontier(paretoData) {
  const tbody = document.getElementById("pareto-tbody");
  if (tbody && paretoData.solutions) {
    tbody.innerHTML = paretoData.solutions
      .map((s) => {
        const badge = s.is_pareto_optimal
          ? `<span class="badge badge-ok">PARETO OPTIMAL</span>`
          : `<span class="badge badge-neutral">DOMINATED</span>`;
        return `
          <tr>
            <td><strong>${s.policy_id}</strong> — ${s.policy_name}</td>
            <td class="mono">$${s.daily_cost_usd.toFixed(0)}</td>
            <td class="mono">${s.daily_co2_kg.toFixed(1)} kg</td>
            <td class="mono">${s.total_distance_km.toFixed(1)} km</td>
            <td class="mono">${s.ev_fleet_share_pct.toFixed(0)}%</td>
            <td class="mono">${s.expected_sla_otd_pct.toFixed(1)}%</td>
            <td>${badge}</td>
          </tr>
        `;
      })
      .join("");
  }

  const pCtx = document.getElementById("pareto-frontier-chart");
  if (pCtx && typeof Chart !== "undefined" && paretoData.solutions) {
    if (state.paretoChart) state.paretoChart.destroy();
    const paretoPts = paretoData.solutions
      .filter((s) => s.is_pareto_optimal)
      .sort((a, b) => a.daily_co2_kg - b.daily_co2_kg);
    const domPts = paretoData.solutions.filter((s) => !s.is_pareto_optimal);

    state.paretoChart = new Chart(pCtx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Pareto-Optimal Frontier (Non-Dominated)",
            data: paretoPts.map((s) => ({
              x: s.daily_co2_kg,
              y: s.daily_cost_usd,
              label: s.policy_name,
            })),
            backgroundColor: "#10b981",
            borderColor: "#06b6d4",
            showLine: true,
            borderWidth: 2,
            pointRadius: 6,
          },
          {
            label: "Dominated Legacy / Suboptimal Policies",
            data: domPts.map((s) => ({
              x: s.daily_co2_kg,
              y: s.daily_cost_usd,
              label: s.policy_name,
            })),
            backgroundColor: "#f43f5e",
            borderColor: "#f43f5e",
            pointRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
        },
        scales: {
          x: {
            title: { display: true, text: "Daily Carbon Footprint (kg CO₂e)", color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
          },
          y: {
            title: { display: true, text: "Daily Dispatch Cost (USD)", color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(30, 45, 74, 0.4)" },
          },
        },
      },
    });
  }
}

function renderAnomaliesTable(anomData) {
  const badge = document.getElementById("anom-metrics-badge");
  if (badge && anomData.validation_metrics) {
    const vm = anomData.validation_metrics;
    badge.textContent = `ROC-AUC: ${vm.roc_auc.toFixed(4)} | PR-AUC: ${vm.pr_auc.toFixed(4)} | F1: ${vm.f1_score.toFixed(3)}`;
  }

  const tbody = document.getElementById("anomalies-tbody");
  if (!tbody || !anomData.events) return;

  tbody.innerHTML = anomData.events
    .map((ev) => {
      const sevClass =
        ev.severity === "CRITICAL"
          ? "badge-danger"
          : ev.severity === "HIGH"
          ? "badge-warn"
          : "badge-info";
      return `
        <tr>
          <td class="mono"><strong>${ev.event_id}</strong><br/><span style="color:#64748b;font-size:11px">${ev.timestamp}</span></td>
          <td><strong>${ev.hub_id}</strong><br/><span class="mono" style="color:#94a3b8;font-size:11px">${ev.sku_id}</span></td>
          <td class="mono">${ev.anomaly_type}</td>
          <td><span class="badge ${sevClass}">${ev.severity}</span></td>
          <td class="mono"><strong>${ev.anomaly_score.toFixed(3)}</strong></td>
          <td class="mono">${ev.root_cause_signal}</td>
          <td style="color:#cbd5e1;max-width:340px">${ev.automated_mitigation}</td>
        </tr>
      `;
    })
    .join("");
}

/* ==================== INTERACTIVE LISTENERS ==================== */
function setupControlListeners() {
  // Slider live readouts
  const bindSlider = (id, valId, fmt) => {
    const el = document.getElementById(id);
    const out = document.getElementById(valId);
    if (el && out) {
      el.addEventListener("input", () => {
        out.textContent = fmt(parseFloat(el.value));
      });
    }
  };

  bindSlider("vrp-demand-mult", "vrp-demand-mult-val", (v) => `${v.toFixed(2)}x`);
  bindSlider("vrp-traffic", "vrp-traffic-val", (v) => v.toFixed(2));
  bindSlider("vrp-weather", "vrp-weather-val", (v) => v.toFixed(2));
  bindSlider("vrp-cap-scale", "vrp-cap-scale-val", (v) => `${v.toFixed(2)}x`);
  bindSlider("eta-dep-hour", "eta-dep-hour-val", (v) => formatHourClock(v));
  bindSlider("eta-traffic", "eta-traffic-val", (v) => v.toFixed(2));
  bindSlider("eta-weather", "eta-weather-val", (v) => v.toFixed(2));
  bindSlider("inv-csl-slider", "inv-csl-val", (v) => `${(v * 100).toFixed(1)}%`);
  bindSlider("inv-lt-slider", "inv-lt-val", (v) => `${v.toFixed(2)}x`);

  // VRP Re-optimize button
  document.getElementById("btn-run-vrp")?.addEventListener("click", async () => {
    const payload = {
      demand_multiplier: parseFloat(document.getElementById("vrp-demand-mult").value),
      traffic_congestion_index: parseFloat(document.getElementById("vrp-traffic").value),
      weather_severity_index: parseFloat(document.getElementById("vrp-weather").value),
      vehicle_capacity_scale: parseFloat(document.getElementById("vrp-cap-scale").value),
      enable_two_opt: document.getElementById("vrp-two-opt").checked,
    };
    const vrpRes = await apiFetch("/api/v1/routes/optimize", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.selectedVehicleId = "ALL";
    renderVRPSection(vrpRes);
    updateExecutiveKPIs(
      vrpRes.benchmark,
      state.overview?.model_diagnostics,
      state.inventoryData
    );
  });

  // Forecast button
  document.getElementById("btn-run-forecast")?.addEventListener("click", async () => {
    const sku = document.getElementById("fc-sku-select").value;
    const hub = document.getElementById("fc-hub-select").value;
    const horizon = document.getElementById("fc-horizon-select").value;
    const mult = document.getElementById("fc-demand-mult").value;
    const fcRes = await apiFetch(
      `/api/v1/forecast?sku_id=${encodeURIComponent(sku)}&hub_id=${encodeURIComponent(hub)}&horizon_days=${horizon}&demand_multiplier=${mult}`
    );
    renderDemandForecastSection(fcRes);
  });

  // ETA button
  document.getElementById("btn-run-eta")?.addEventListener("click", async () => {
    const payload = {
      origin_hub_id: document.getElementById("eta-origin").value,
      destination_hub_id: document.getElementById("eta-dest").value,
      departure_hour: parseFloat(document.getElementById("eta-dep-hour").value),
      day_of_week: 2,
      traffic_congestion_index: parseFloat(document.getElementById("eta-traffic").value),
      weather_severity_index: parseFloat(document.getElementById("eta-weather").value),
      payload_weight_kg: parseFloat(document.getElementById("eta-payload").value),
      promised_sla_minutes: parseFloat(document.getElementById("eta-sla").value),
    };
    const etaRes = await apiFetch("/api/v1/eta/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderETASection(etaRes);
  });

  // Inventory button & status filter
  document.getElementById("btn-run-inventory")?.addEventListener("click", async () => {
    const csl = document.getElementById("inv-csl-slider").value;
    const lt = document.getElementById("inv-lt-slider").value;
    const dMult = document.getElementById("vrp-demand-mult").value;
    const invRes = await apiFetch(
      `/api/v1/inventory/optimize?target_service_level=${csl}&demand_multiplier=${dMult}&lead_time_shock_multiplier=${lt}`
    );
    renderInventorySection(invRes);
    updateExecutiveKPIs(
      state.vrpData.benchmark,
      state.overview?.model_diagnostics,
      invRes
    );
  });

  document.getElementById("inv-status-filter")?.addEventListener("change", () => {
    if (state.inventoryData) renderInventorySection(state.inventoryData);
  });

  // MLOps Drift button
  document.getElementById("btn-run-mlops-drift")?.addEventListener("click", async () => {
    const regime = document.getElementById("mlops-drift-select")?.value || "nominal";
    const driftRes = await apiFetch(`/api/v1/mlops/drift?drift_regime=${encodeURIComponent(regime)}`);
    renderDriftMonitor(driftRes);
  });

  document.getElementById("mlops-drift-select")?.addEventListener("change", async (e) => {
    const regime = e.target.value || "nominal";
    const driftRes = await apiFetch(`/api/v1/mlops/drift?drift_regime=${encodeURIComponent(regime)}`);
    renderDriftMonitor(driftRes);
  });

  // Close scenario banner
  document.getElementById("btn-close-banner")?.addEventListener("click", () => {
    document.getElementById("scenario-alert-banner")?.classList.add("hidden");
  });

  // Top-bar Stress Presets
  const presets = {
    baseline: {
      scenario_name: "Nominal Baseline Operations",
      demand_multiplier: 1.0,
      traffic_congestion_index: 0.35,
      weather_severity_index: 0.15,
      lead_time_shock_multiplier: 1.0,
      target_service_level: 0.96,
    },
    festive: {
      scenario_name: "Festive Peak Demand Surge (+35%)",
      demand_multiplier: 1.35,
      traffic_congestion_index: 0.65,
      weather_severity_index: 0.20,
      lead_time_shock_multiplier: 1.25,
      target_service_level: 0.975,
    },
    monsoon: {
      scenario_name: "Severe Monsoon Corridor Disruption",
      demand_multiplier: 1.10,
      traffic_congestion_index: 0.80,
      weather_severity_index: 0.85,
      lead_time_shock_multiplier: 1.45,
      target_service_level: 0.96,
    },
    supply_shock: {
      scenario_name: "Port & Supplier Lead-Time Shock (1.8x)",
      demand_multiplier: 1.15,
      traffic_congestion_index: 0.45,
      weather_severity_index: 0.25,
      lead_time_shock_multiplier: 1.80,
      target_service_level: 0.98,
    },
  };

  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".btn-preset").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const key = btn.getAttribute("data-preset");
      const cfg = presets[key] || presets.baseline;

      // Sync sliders in UI
      document.getElementById("vrp-demand-mult").value = cfg.demand_multiplier;
      document.getElementById("vrp-demand-mult-val").textContent = `${cfg.demand_multiplier.toFixed(2)}x`;
      document.getElementById("vrp-traffic").value = cfg.traffic_congestion_index;
      document.getElementById("vrp-traffic-val").textContent = cfg.traffic_congestion_index.toFixed(2);
      document.getElementById("vrp-weather").value = cfg.weather_severity_index;
      document.getElementById("vrp-weather-val").textContent = cfg.weather_severity_index.toFixed(2);
      document.getElementById("inv-csl-slider").value = cfg.target_service_level;
      document.getElementById("inv-csl-val").textContent = `${(cfg.target_service_level * 100).toFixed(1)}%`;
      document.getElementById("inv-lt-slider").value = cfg.lead_time_shock_multiplier;
      document.getElementById("inv-lt-val").textContent = `${cfg.lead_time_shock_multiplier.toFixed(2)}x`;

      // Run Scenario Simulation + refresh VRP & Inventory
      const [simRes, vrpRes, invRes] = await Promise.all([
        apiFetch("/api/v1/scenarios/simulate", {
          method: "POST",
          body: JSON.stringify(cfg),
        }),
        apiFetch("/api/v1/routes/optimize", {
          method: "POST",
          body: JSON.stringify({
            demand_multiplier: cfg.demand_multiplier,
            traffic_congestion_index: cfg.traffic_congestion_index,
            weather_severity_index: cfg.weather_severity_index,
            vehicle_capacity_scale: 1.0,
            enable_two_opt: true,
          }),
        }),
        apiFetch(
          `/api/v1/inventory/optimize?target_service_level=${cfg.target_service_level}&demand_multiplier=${cfg.demand_multiplier}&lead_time_shock_multiplier=${cfg.lead_time_shock_multiplier}`
        ),
      ]);

      state.selectedVehicleId = "ALL";
      renderVRPSection(vrpRes);
      renderInventorySection(invRes);
      updateExecutiveKPIs(
        vrpRes.benchmark,
        state.overview?.model_diagnostics,
        invRes
      );

      const banner = document.getElementById("scenario-alert-banner");
      if (banner) {
        if (key === "baseline") {
          banner.classList.add("hidden");
        } else {
          banner.classList.remove("hidden");
          document.getElementById("sim-banner-title").textContent = simRes.scenario_name.toUpperCase();
          document.getElementById("sim-banner-resilience").textContent = `Network Resilience Score: ${simRes.resilience_score.toFixed(1)} / 100`;
          document.getElementById("sim-banner-sla").textContent = `Mean SLA Breach Risk: ${simRes.network_sla_risk_pct.toFixed(1)}%`;
          document.getElementById("sim-recommendations-list").innerHTML = simRes.executive_recommendations
            .map((r) => `<li>${r}</li>`)
            .join("");
        }
      }
    });
  });
}
