"""
Stochastic Multi-Echelon Inventory & Safety Stock Optimizer.

Implements continuous-review (s, S) / (ROP, EOQ) inventory control under joint
demand and supplier lead-time uncertainty:
  - Lead-Time Demand Variance: sigma_DL^2 = mu_L * sigma_D^2 + mu_D^2 * sigma_L^2
  - Safety Stock: SS = z_alpha * sigma_DL
  - Reorder Point: ROP = mu_D * mu_L + SS
  - Economic Order Quantity: EOQ = sqrt(2 * D_annual * K / (h * C))
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

from logiroute.config import DEFAULT_SKUS, SETTINGS, SKUConfig
from logiroute.schemas import InventoryOptimizationResponse, InventoryPolicyItem


class StochasticInventoryOptimizer:
    """Computes optimal safety stock, reorder points, EOQ, and replenishment alerts."""

    def __init__(
        self,
        skus: Optional[List[SKUConfig]] = None,
        seed: int = SETTINGS.random_seed,
    ) -> None:
        self.skus = skus or DEFAULT_SKUS
        self.sku_map: Dict[str, SKUConfig] = {s.sku_id: s for s in self.skus}
        self.seed = seed

    def optimize_network_inventory(
        self,
        demand_stats: List[Dict[str, float | str]],
        target_service_level: float = SETTINGS.default_service_level,
        lead_time_shock_multiplier: float = 1.0,
    ) -> InventoryOptimizationResponse:
        """
        Compute optimal (s, S) policies across all SKU x Hub combinations
        given forecasted demand distributions and supplier lead-time parameters.
        """
        clamped_csl = float(np.clip(target_service_level, 0.80, 0.999))
        z_score = float(norm.ppf(clamped_csl))
        rng = np.random.default_rng(self.seed + 77)

        policies: List[InventoryPolicyItem] = []
        critical_alerts = 0
        reorder_triggered = 0
        total_po_value = 0.0
        total_annual_cost = 0.0

        for idx, stat in enumerate(demand_stats):
            sku_id = str(stat["sku_id"])
            hub_id = str(stat["hub_id"])
            sku_cfg = self.sku_map.get(sku_id, self.skus[0])

            mu_d = max(1.0, float(stat["daily_demand_mean"]))
            sigma_d = max(0.5, float(stat["daily_demand_std"]))
            mu_l = sku_cfg.lead_time_days_mean * lead_time_shock_multiplier
            sigma_l = sku_cfg.lead_time_days_std * lead_time_shock_multiplier

            # Combined lead-time demand standard deviation
            sigma_dl = math.sqrt(mu_l * (sigma_d**2) + (mu_d**2) * (sigma_l**2))
            expected_lt_demand = mu_d * mu_l

            safety_stock = int(math.ceil(z_score * sigma_dl))
            reorder_point = int(math.ceil(expected_lt_demand + safety_stock))

            annual_demand = mu_d * 365.0
            annual_holding_cost_per_unit = (
                sku_cfg.unit_cost_usd * sku_cfg.holding_cost_rate_annual
            )
            eoq = int(
                math.ceil(
                    math.sqrt(
                        (2.0 * annual_demand * sku_cfg.ordering_cost_usd)
                        / max(0.05, annual_holding_cost_per_unit)
                    )
                )
            )
            order_up_to_s = reorder_point + eoq

            # Deterministic realistic current inventory snapshot across hubs
            # Mix of healthy, reorder-ready, and critical stockout risk states
            cycle_phase = ((idx * 37 + 13) % 100) / 100.0
            if cycle_phase < 0.20:
                # Critical low stock state
                on_hand = max(8, int(round(reorder_point * rng.uniform(0.42, 0.68))))
                in_transit = 0
            elif cycle_phase < 0.50:
                # Below ROP - needs replenishment
                on_hand = int(round(reorder_point * rng.uniform(0.70, 0.94)))
                in_transit = int(round(eoq * 0.25)) if cycle_phase > 0.38 else 0
            else:
                # Healthy stock state
                on_hand = int(round(reorder_point + eoq * rng.uniform(0.20, 0.85)))
                in_transit = int(round(eoq * 0.15)) if cycle_phase > 0.82 else 0

            inv_position = on_hand + in_transit
            days_of_supply = round(on_hand / max(1.0, mu_d), 1)

            # Empirical stockout probability during lead time given current on-hand + in-transit
            z_current = (inv_position - expected_lt_demand) / max(1.0, sigma_dl)
            stockout_prob_pct = round(float((1.0 - norm.cdf(z_current)) * 100.0), 2)

            if inv_position <= reorder_point:
                rec_order_qty = max(eoq, order_up_to_s - inv_position)
                reorder_triggered += 1
                total_po_value += rec_order_qty * sku_cfg.unit_cost_usd
                if on_hand < expected_lt_demand or stockout_prob_pct >= 25.0:
                    status = "CRITICAL_STOCKOUT_RISK"
                    critical_alerts += 1
                else:
                    status = "REORDER_TRIGGERED"
            else:
                rec_order_qty = 0
                status = "OPTIMAL_BUFFER"

            annual_holding_usd = (eoq / 2.0 + safety_stock) * annual_holding_cost_per_unit
            annual_ordering_usd = (annual_demand / max(1.0, eoq)) * sku_cfg.ordering_cost_usd
            total_item_cost = annual_holding_usd + annual_ordering_usd
            total_annual_cost += total_item_cost

            policies.append(
                InventoryPolicyItem(
                    sku_id=sku_id,
                    sku_name=str(stat["sku_name"]),
                    category=str(stat["category"]),
                    hub_id=hub_id,
                    hub_name=str(stat["hub_name"]),
                    daily_demand_mean=round(mu_d, 1),
                    daily_demand_std=round(sigma_d, 1),
                    lead_time_days_mean=round(mu_l, 2),
                    lead_time_days_std=round(sigma_l, 2),
                    current_on_hand_units=on_hand,
                    in_transit_units=in_transit,
                    safety_stock_units=safety_stock,
                    reorder_point_units=reorder_point,
                    economic_order_qty_units=eoq,
                    order_up_to_level_s=order_up_to_s,
                    recommended_order_qty=rec_order_qty,
                    days_of_supply=days_of_supply,
                    stockout_probability_pct=stockout_prob_pct,
                    status=status,
                    annual_holding_cost_usd=round(annual_holding_usd, 2),
                    annual_ordering_cost_usd=round(annual_ordering_usd, 2),
                    total_annual_policy_cost_usd=round(total_item_cost, 2),
                )
            )

        # Sort policies so critical and reorder-triggered items appear first
        status_rank = {
            "CRITICAL_STOCKOUT_RISK": 0,
            "REORDER_TRIGGERED": 1,
            "OPTIMAL_BUFFER": 2,
        }
        policies.sort(
            key=lambda p: (status_rank.get(p.status, 3), -p.stockout_probability_pct)
        )

        return InventoryOptimizationResponse(
            target_service_level_pct=round(clamped_csl * 100.0, 2),
            total_skus_evaluated=len(policies),
            critical_stockout_alerts=critical_alerts,
            reorder_triggered_count=reorder_triggered,
            total_recommended_po_value_usd=round(total_po_value, 2),
            total_annual_inventory_cost_usd=round(total_annual_cost, 2),
            policies=policies,
        )
