"""
Capacitated Vehicle Routing Problem with Time Windows & Heterogeneous Fleet (CVRPTW) Solver.

Algorithmic Pipeline:
1. Pairwise Road Network Distance & Congestion-Adjusted Travel Time Matrix
2. Clarke-Wright Parallel Savings Heuristic with Time-Window Compatibility
3. Intra-Route 2-Opt Local Search Metaheuristic
4. Inter-Route Relocate Refinement
5. Cost & Carbon-Aware Heterogeneous Fleet Assignment (EV Cargo Van, 9T Box, 16T Heavy)
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logiroute.config import DEFAULT_FLEET_CLASSES, SETTINGS, VehicleClassConfig
from logiroute.data.synthetic_generator import road_distance_km
from logiroute.schemas import (
    DeliveryNode,
    RouteStop,
    VehicleRoutePlan,
    VRPBenchmarkSummary,
    VRPOptimizationResponse,
)


def _format_hour_clock(hour_float: float) -> str:
    """Convert decimal hour (e.g. 8.75) to HH:MM 24-hour clock string."""
    clamped = max(0.0, min(23.98, hour_float))
    h = int(math.floor(clamped))
    m = int(round((clamped - h) * 60))
    if m == 60:
        h = min(23, h + 1)
        m = 0
    return f"{h:02d}:{m:02d}"


class CVRPTWSolver:
    """
    Solves the multi-stop Capacitated Vehicle Routing Problem with Time Windows
    and assigns optimal vehicle classes.
    """

    def __init__(
        self,
        fleet_classes: Optional[List[VehicleClassConfig]] = None,
    ) -> None:
        self.fleet_classes = sorted(
            fleet_classes or DEFAULT_FLEET_CLASSES,
            key=lambda v: v.capacity_kg,
        )
        self.max_vehicle_capacity_kg = max(v.capacity_kg for v in self.fleet_classes)

    @staticmethod
    def _compute_distance_matrix(nodes: List[DeliveryNode]) -> np.ndarray:
        n = len(nodes)
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = road_distance_km(nodes[i].lat, nodes[i].lon, nodes[j].lat, nodes[j].lon)
                mat[i, j] = d
                mat[j, i] = d
        return mat

    @staticmethod
    def _route_distance(route: List[int], dist_matrix: np.ndarray) -> float:
        """Compute closed-tour distance starting and ending at depot (index 0)."""
        if not route:
            return 0.0
        total = dist_matrix[0, route[0]]
        for idx in range(len(route) - 1):
            total += dist_matrix[route[idx], route[idx + 1]]
        total += dist_matrix[route[-1], 0]
        return float(total)

    def _clarke_wright_savings(
        self,
        nodes: List[DeliveryNode],
        dist_matrix: np.ndarray,
        max_capacity_kg: float,
    ) -> List[List[int]]:
        """
        Construct initial feasible routes using Clarke-Wright Parallel Savings
        with time-window urgency ordering.
        """
        n_nodes = len(nodes)
        customer_indices = list(range(1, n_nodes))

        # Initialize each customer on its own dedicated route
        routes: List[List[int]] = [[i] for i in customer_indices]
        route_loads: List[float] = [nodes[i].demand_kg for i in customer_indices]

        # Compute Clarke-Wright savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
        # penalized by large time-window gaps so compatible windows group together
        savings_list: List[Tuple[float, int, int]] = []
        for idx_a in range(len(customer_indices)):
            i = customer_indices[idx_a]
            for idx_b in range(idx_a + 1, len(customer_indices)):
                j = customer_indices[idx_b]
                raw_saving = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
                tw_gap = abs(nodes[i].time_window_start_hr - nodes[j].time_window_start_hr)
                tw_penalty = 0.85 * tw_gap
                net_saving = raw_saving - tw_penalty
                if net_saving > 0:
                    savings_list.append((net_saving, i, j))

        savings_list.sort(key=lambda x: x[0], reverse=True)

        def find_route(cust_id: int) -> Tuple[int, str]:
            for r_idx, r in enumerate(routes):
                if not r:
                    continue
                if r[0] == cust_id and r[-1] == cust_id:
                    return r_idx, "SINGLE"
                if r[0] == cust_id:
                    return r_idx, "HEAD"
                if r[-1] == cust_id:
                    return r_idx, "TAIL"
            return -1, "INTERIOR"

        for _, i, j in savings_list:
            r_i, pos_i = find_route(i)
            r_j, pos_j = find_route(j)
            if r_i == -1 or r_j == -1 or r_i == r_j:
                continue
            if pos_i == "INTERIOR" or pos_j == "INTERIOR":
                continue
            if route_loads[r_i] + route_loads[r_j] > max_capacity_kg:
                continue

            # Orient routes so i is at the tail of r_i and j is at the head of r_j
            route_a = routes[r_i][:]
            route_b = routes[r_j][:]
            if pos_i == "HEAD":
                route_a.reverse()
            if pos_j == "TAIL":
                route_b.reverse()

            merged = route_a + route_b
            # Prefer ordering where earlier time-window center comes first
            first_tw = nodes[merged[0]].time_window_end_hr
            last_tw = nodes[merged[-1]].time_window_end_hr
            if first_tw > last_tw + 1.0:
                merged.reverse()

            routes[r_i] = merged
            route_loads[r_i] += route_loads[r_j]
            routes[r_j] = []
            route_loads[r_j] = 0.0

        active_routes = [r for r in routes if len(r) > 0]
        return active_routes

    def _two_opt_route(
        self,
        route: List[int],
        nodes: List[DeliveryNode],
        dist_matrix: np.ndarray,
    ) -> Tuple[List[int], int]:
        """
        Apply Intra-route 2-Opt edge-exchange local search while respecting
        time-window monotonicity preferences.
        """
        if len(route) < 3:
            if len(route) == 2:
                if nodes[route[0]].time_window_end_hr > nodes[route[1]].time_window_end_hr + 1.5:
                    return [route[1], route[0]], 1
            return route[:], 0

        best_route = route[:]
        best_cost = self._route_distance(best_route, dist_matrix)
        improved = True
        iterations = 0

        while improved and iterations < 50:
            improved = False
            n = len(best_route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = (
                        best_route[:i]
                        + list(reversed(best_route[i : j + 1]))
                        + best_route[j + 1 :]
                    )
                    cand_cost = self._route_distance(candidate, dist_matrix)
                    if cand_cost + 1e-6 < best_cost:
                        best_route = candidate
                        best_cost = cand_cost
                        improved = True
                        iterations += 1
        return best_route, iterations

    def _inter_route_relocate(
        self,
        routes: List[List[int]],
        nodes: List[DeliveryNode],
        dist_matrix: np.ndarray,
        max_capacity_kg: float,
    ) -> Tuple[List[List[int]], int]:
        """
        Attempt to relocate individual stops across routes to reduce total network distance.
        """
        loads = [sum(nodes[idx].demand_kg for idx in r) for r in routes]
        moves = 0
        improved = True

        while improved and moves < 25:
            improved = False
            for r_from in range(len(routes)):
                if len(routes[r_from]) <= 1:
                    continue
                for pos_from in range(len(routes[r_from])):
                    cust = routes[r_from][pos_from]
                    cust_demand = nodes[cust].demand_kg

                    old_from_route = routes[r_from]
                    new_from_route = old_from_route[:pos_from] + old_from_route[pos_from + 1 :]
                    delta_remove = self._route_distance(
                        new_from_route, dist_matrix
                    ) - self._route_distance(old_from_route, dist_matrix)

                    best_delta = -0.25  # Minimum improvement threshold (km)
                    best_target: Optional[Tuple[int, int]] = None

                    for r_to in range(len(routes)):
                        if r_to == r_from:
                            continue
                        if loads[r_to] + cust_demand > max_capacity_kg:
                            continue
                        old_to_cost = self._route_distance(routes[r_to], dist_matrix)
                        for pos_to in range(len(routes[r_to]) + 1):
                            cand_to = (
                                routes[r_to][:pos_to] + [cust] + routes[r_to][pos_to:]
                            )
                            delta_insert = (
                                self._route_distance(cand_to, dist_matrix) - old_to_cost
                            )
                            total_delta = delta_remove + delta_insert
                            if total_delta < best_delta:
                                best_delta = total_delta
                                best_target = (r_to, pos_to)

                    if best_target is not None:
                        r_to, pos_to = best_target
                        routes[r_from] = new_from_route
                        routes[r_to] = (
                            routes[r_to][:pos_to] + [cust] + routes[r_to][pos_to:]
                        )
                        loads[r_from] -= cust_demand
                        loads[r_to] += cust_demand
                        moves += 1
                        improved = True
                        break
                if improved:
                    break

        return [r for r in routes if len(r) > 0], moves

    def _select_vehicle_class(
        self, total_load_kg: float, capacity_scale: float
    ) -> VehicleClassConfig:
        """Select the most cost- and carbon-efficient vehicle class that fits the route load."""
        for vc in self.fleet_classes:
            if vc.capacity_kg * capacity_scale >= total_load_kg:
                return vc
        return self.fleet_classes[-1]

    def solve(
        self,
        nodes: List[DeliveryNode],
        demand_multiplier: float = 1.0,
        traffic_congestion_index: float = 0.35,
        weather_severity_index: float = 0.15,
        vehicle_capacity_scale: float = 1.0,
        enable_two_opt: bool = True,
    ) -> VRPOptimizationResponse:
        """
        Execute the full CVRPTW optimization pipeline and return route plans
        with baseline benchmark comparisons.
        """
        t0 = time.perf_counter()

        # Ensure depot is at index 0
        depot_nodes = [n for n in nodes if n.is_depot]
        cust_nodes = [n for n in nodes if not n.is_depot]
        if not depot_nodes:
            raise ValueError("At least one depot node is required for CVRPTW.")

        ordered_nodes: List[DeliveryNode] = [depot_nodes[0]]
        for c in cust_nodes:
            scaled_c = c.model_copy(
                update={
                    "demand_kg": round(c.demand_kg * demand_multiplier, 1),
                    "demand_volume_m3": round(c.demand_volume_m3 * demand_multiplier, 2),
                    "demand_units": max(1, int(round(c.demand_units * demand_multiplier))),
                }
            )
            ordered_nodes.append(scaled_c)

        dist_matrix = self._compute_distance_matrix(ordered_nodes)
        max_cap = self.max_vehicle_capacity_kg * vehicle_capacity_scale

        # 1. Compute naive baseline (unoptimized FIFO / 2-stop dispatch batches)
        baseline_routes: List[List[int]] = []
        curr_batch: List[int] = []
        curr_load = 0.0
        for idx in range(1, len(ordered_nodes)):
            d_kg = ordered_nodes[idx].demand_kg
            if len(curr_batch) >= 2 or (curr_load + d_kg > max_cap * 0.72 and curr_batch):
                baseline_routes.append(curr_batch)
                curr_batch = [idx]
                curr_load = d_kg
            else:
                curr_batch.append(idx)
                curr_load += d_kg
        if curr_batch:
            baseline_routes.append(curr_batch)

        baseline_dist = sum(
            self._route_distance(r, dist_matrix) for r in baseline_routes
        )
        heavy_vc = self.fleet_classes[-1]
        baseline_cost = sum(
            heavy_vc.fixed_dispatch_cost_usd
            + self._route_distance(r, dist_matrix) * heavy_vc.cost_per_km_usd
            for r in baseline_routes
        )
        baseline_co2 = baseline_dist * heavy_vc.co2_kg_per_km

        # 2. Clarke-Wright Savings Construction
        cw_routes = self._clarke_wright_savings(ordered_nodes, dist_matrix, max_cap)

        # 3. Intra-route 2-Opt + Inter-route Relocate
        total_two_opt_iters = 0
        if enable_two_opt:
            refined_routes: List[List[int]] = []
            for r in cw_routes:
                opt_r, iters = self._two_opt_route(r, ordered_nodes, dist_matrix)
                refined_routes.append(opt_r)
                total_two_opt_iters += iters

            refined_routes, reloc_moves = self._inter_route_relocate(
                refined_routes, ordered_nodes, dist_matrix, max_cap
            )
            total_two_opt_iters += reloc_moves
            final_routes: List[List[int]] = []
            for r in refined_routes:
                opt_r, iters = self._two_opt_route(r, ordered_nodes, dist_matrix)
                final_routes.append(opt_r)
                total_two_opt_iters += iters
        else:
            final_routes = cw_routes

        # 4. Build detailed VehicleRoutePlan objects with congestion/weather-aware schedules
        speed_penalty_factor = max(
            0.42,
            1.0 - 0.36 * traffic_congestion_index - 0.24 * weather_severity_index,
        )

        route_plans: List[VehicleRoutePlan] = []
        depot = ordered_nodes[0]
        total_opt_dist = 0.0
        total_opt_cost = 0.0
        total_opt_co2 = 0.0
        total_load_all = 0.0
        total_cap_all = 0.0
        on_time_stops = 0
        total_cust_stops = 0

        for r_idx, r in enumerate(final_routes):
            route_load_kg = sum(ordered_nodes[i].demand_kg for i in r)
            vc = self._select_vehicle_class(route_load_kg, vehicle_capacity_scale)
            eff_cap_kg = round(vc.capacity_kg * vehicle_capacity_scale, 1)
            eff_speed_kmh = vc.avg_speed_kmh * speed_penalty_factor

            # Depart depot slightly before first customer's time window
            first_node = ordered_nodes[r[0]]
            first_leg_hr = dist_matrix[0, r[0]] / max(12.0, eff_speed_kmh)
            depart_depot_hr = max(
                SETTINGS.depot_open_hour,
                min(8.5, first_node.time_window_start_hr - first_leg_hr),
            )

            stops: List[RouteStop] = [
                RouteStop(
                    sequence_index=0,
                    node_id=depot.node_id,
                    node_name=depot.name,
                    city=depot.city,
                    lat=depot.lat,
                    lon=depot.lon,
                    is_depot=True,
                    arrival_hour=round(depart_depot_hr, 2),
                    departure_hour=round(depart_depot_hr, 2),
                    arrival_time_formatted=_format_hour_clock(depart_depot_hr),
                    departure_time_formatted=_format_hour_clock(depart_depot_hr),
                    time_window_formatted="06:00 - 20:00",
                    cumulative_distance_km=0.0,
                    leg_distance_km=0.0,
                    leg_travel_time_min=0.0,
                    delivered_kg=0.0,
                    cumulative_load_kg=0.0,
                    sla_on_time=True,
                    delay_risk_probability=0.02,
                )
            ]

            geometry_coords: List[List[float]] = [[depot.lat, depot.lon]]
            curr_hr = depart_depot_hr
            cum_dist = 0.0
            cum_load = 0.0
            prev_idx = 0
            route_on_time_count = 0

            for seq_pos, node_idx in enumerate(r, start=1):
                node = ordered_nodes[node_idx]
                leg_km = float(dist_matrix[prev_idx, node_idx])
                leg_hr = leg_km / max(12.0, eff_speed_kmh)
                arr_hr = curr_hr + leg_hr

                # Wait if arriving before time window opens
                service_start_hr = max(arr_hr, node.time_window_start_hr)
                dep_hr = service_start_hr + (node.service_duration_min / 60.0)

                cum_dist += leg_km
                cum_load += node.demand_kg
                is_on_time = arr_hr <= (node.time_window_end_hr + 0.25)

                # Delay risk increases as arrival approaches or exceeds window end
                slack_hr = node.time_window_end_hr - arr_hr
                risk_logit = -1.8 * slack_hr + 1.6 * traffic_congestion_index + 1.2 * weather_severity_index
                delay_risk = float(np.clip(1.0 / (1.0 + math.exp(-risk_logit)), 0.02, 0.98))

                if is_on_time:
                    route_on_time_count += 1
                    on_time_stops += 1
                total_cust_stops += 1

                stops.append(
                    RouteStop(
                        sequence_index=seq_pos,
                        node_id=node.node_id,
                        node_name=node.name,
                        city=node.city,
                        lat=node.lat,
                        lon=node.lon,
                        is_depot=False,
                        arrival_hour=round(arr_hr, 2),
                        departure_hour=round(dep_hr, 2),
                        arrival_time_formatted=_format_hour_clock(arr_hr),
                        departure_time_formatted=_format_hour_clock(dep_hr),
                        time_window_formatted=(
                            f"{_format_hour_clock(node.time_window_start_hr)} - "
                            f"{_format_hour_clock(node.time_window_end_hr)}"
                        ),
                        cumulative_distance_km=round(cum_dist, 2),
                        leg_distance_km=round(leg_km, 2),
                        leg_travel_time_min=round(leg_hr * 60.0, 1),
                        delivered_kg=round(node.demand_kg, 1),
                        cumulative_load_kg=round(cum_load, 1),
                        sla_on_time=is_on_time,
                        delay_risk_probability=round(delay_risk, 3),
                    )
                )
                geometry_coords.append([node.lat, node.lon])
                curr_hr = dep_hr
                prev_idx = node_idx

            # Return leg to central depot
            return_km = float(dist_matrix[prev_idx, 0])
            return_hr = return_km / max(12.0, eff_speed_kmh)
            final_arr_hr = curr_hr + return_hr
            cum_dist += return_km
            geometry_coords.append([depot.lat, depot.lon])

            stops.append(
                RouteStop(
                    sequence_index=len(r) + 1,
                    node_id=depot.node_id,
                    node_name=f"{depot.name} (Return)",
                    city=depot.city,
                    lat=depot.lat,
                    lon=depot.lon,
                    is_depot=True,
                    arrival_hour=round(final_arr_hr, 2),
                    departure_hour=round(final_arr_hr, 2),
                    arrival_time_formatted=_format_hour_clock(final_arr_hr),
                    departure_time_formatted=_format_hour_clock(final_arr_hr),
                    time_window_formatted="06:00 - 20:00",
                    cumulative_distance_km=round(cum_dist, 2),
                    leg_distance_km=round(return_km, 2),
                    leg_travel_time_min=round(return_hr * 60.0, 1),
                    delivered_kg=0.0,
                    cumulative_load_kg=round(cum_load, 1),
                    sla_on_time=True,
                    delay_risk_probability=0.02,
                )
            )

            route_duration_hr = max(0.5, final_arr_hr - depart_depot_hr)
            route_cost_usd = vc.fixed_dispatch_cost_usd + cum_dist * vc.cost_per_km_usd
            route_co2_kg = cum_dist * vc.co2_kg_per_km
            util_pct = min(100.0, (route_load_kg / max(1.0, eff_cap_kg)) * 100.0)
            sla_pct = (route_on_time_count / max(1, len(r))) * 100.0

            total_opt_dist += cum_dist
            total_opt_cost += route_cost_usd
            total_opt_co2 += route_co2_kg
            total_load_all += route_load_kg
            total_cap_all += eff_cap_kg

            route_plans.append(
                VehicleRoutePlan(
                    vehicle_id=f"LR-FLEET-{r_idx + 1:02d}",
                    vehicle_class=vc.class_id,
                    vehicle_name=vc.name,
                    capacity_kg=eff_cap_kg,
                    total_load_kg=round(route_load_kg, 1),
                    utilization_pct=round(util_pct, 1),
                    total_distance_km=round(cum_dist, 2),
                    total_duration_hours=round(route_duration_hr, 2),
                    total_cost_usd=round(route_cost_usd, 2),
                    co2_emissions_kg=round(route_co2_kg, 2),
                    num_stops=len(r),
                    sla_compliance_pct=round(sla_pct, 1),
                    stops=stops,
                    geometry_coords=geometry_coords,
                )
            )

        solve_ms = (time.perf_counter() - t0) * 1000.0
        dist_saved = max(0.0, baseline_dist - total_opt_dist)
        dist_red_pct = (dist_saved / max(1.0, baseline_dist)) * 100.0
        cost_saved = max(0.0, baseline_cost - total_opt_cost)
        cost_red_pct = (cost_saved / max(1.0, baseline_cost)) * 100.0
        co2_saved = max(0.0, baseline_co2 - total_opt_co2)
        fleet_util_pct = (total_load_all / max(1.0, total_cap_all)) * 100.0
        otd_pct = (on_time_stops / max(1, total_cust_stops)) * 100.0

        solver_label = (
            "Clarke-Wright Parallel Savings + 2-Opt + Inter-Route Relocate"
            if enable_two_opt
            else "Clarke-Wright Parallel Savings (Construction Only)"
        )

        benchmark = VRPBenchmarkSummary(
            optimized_distance_km=round(total_opt_dist, 2),
            baseline_distance_km=round(baseline_dist, 2),
            distance_saved_km=round(dist_saved, 2),
            distance_reduction_pct=round(dist_red_pct, 2),
            optimized_cost_usd=round(total_opt_cost, 2),
            baseline_cost_usd=round(baseline_cost, 2),
            cost_saved_usd=round(cost_saved, 2),
            cost_reduction_pct=round(cost_red_pct, 2),
            optimized_co2_kg=round(total_opt_co2, 2),
            baseline_co2_kg=round(baseline_co2, 2),
            co2_saved_kg=round(co2_saved, 2),
            vehicles_dispatched=len(route_plans),
            total_fleet_utilization_pct=round(fleet_util_pct, 1),
            on_time_delivery_rate_pct=round(otd_pct, 1),
            solver_algorithm=solver_label,
            two_opt_iterations=total_two_opt_iters,
            solve_time_ms=round(solve_ms, 2),
        )

        return VRPOptimizationResponse(
            depot=depot,
            routes=route_plans,
            benchmark=benchmark,
        )
