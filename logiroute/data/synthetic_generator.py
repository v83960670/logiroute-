"""
Deterministic, domain-authentic synthetic data generator for LogiRoute AI.

Generates:
1. Multi-echelon daily SKU demand time-series with seasonality, price elasticity,
   promotions, weather shocks, and autoregressive dynamics.
2. Historical shipment telemetry records for training the Traffic & Weather-Aware
   ETA & SLA Delay Risk machine learning models.
3. Daily customer delivery manifests with time windows and cargo constraints
   for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from logiroute.config import (
    DEFAULT_REGIONAL_NODES,
    DEFAULT_SKUS,
    HubConfig,
    SETTINGS,
    SKUConfig,
)
from logiroute.schemas import DeliveryNode


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 coordinates."""
    radius_earth_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_earth_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def road_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    tortuosity: float = SETTINGS.road_tortuosity_factor,
) -> float:
    """Estimate actual road network distance using regional tortuosity factor."""
    return haversine_km(lat1, lon1, lat2, lon2) * tortuosity


class SupplyChainDataGenerator:
    """Generates calibrated datasets for demand forecasting, ETA ML, and CVRPTW."""

    def __init__(
        self,
        seed: int = SETTINGS.random_seed,
        nodes: Optional[List[HubConfig]] = None,
        skus: Optional[List[SKUConfig]] = None,
    ) -> None:
        self.seed = seed
        self.nodes = nodes or DEFAULT_REGIONAL_NODES
        self.skus = skus or DEFAULT_SKUS
        self.node_map: Dict[str, HubConfig] = {n.hub_id: n for n in self.nodes}
        self.sku_map: Dict[str, SKUConfig] = {s.sku_id: s for s in self.skus}

    def generate_demand_history(
        self,
        days: int = 365,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Generate daily demand history across SKU x Hub combinations with
        calendar cyclicality, promotional uplifts, weather indices, and AR(1) noise.
        """
        rng = np.random.default_rng(self.seed)
        if end_date is None:
            end_date = date(2026, 10, 3)
        start_date = end_date - timedelta(days=days - 1)
        dates = [start_date + timedelta(days=i) for i in range(days)]

        records: List[Dict[str, object]] = []
        # Select top active distribution hubs (including central depot aggregate + 5 key nodes)
        target_hubs = [n for n in self.nodes if not n.is_depot][:6]

        for sku_idx, sku in enumerate(self.skus):
            sku_scale = {
                "SKU-PHM-101": 1.15,
                "SKU-ELC-204": 0.65,
                "SKU-FMC-309": 1.85,
                "SKU-AUT-412": 0.45,
                "SKU-APP-518": 1.30,
            }.get(sku.sku_id, 1.0)

            for hub_idx, hub in enumerate(target_hubs):
                base_level = max(25.0, (hub.base_daily_demand / 3.8) * sku_scale)
                ar_state = 0.0

                for d_idx, current_date in enumerate(dates):
                    dow = current_date.weekday()
                    day_of_year = current_date.timetuple().tm_yday

                    # Weekly pattern: commercial B2B peaks Tue-Fri, dips Sunday
                    weekly_factor = 1.0 + 0.14 * math.sin(2.0 * math.pi * (dow - 1) / 7.0)
                    if dow == 6:
                        weekly_factor *= 0.82

                    # Annual & seasonal wave (autumn festive surge + summer cycle)
                    seasonal_factor = (
                        1.0
                        + 0.18 * math.sin(2.0 * math.pi * (day_of_year - 60) / 365.25)
                        + 0.12 * math.cos(4.0 * math.pi * (day_of_year - 240) / 365.25)
                    )

                    # Secular growth trend (+12% annualized)
                    trend_factor = 1.0 + 0.12 * (d_idx / max(1, days))

                    # Promotional windows (deterministic pseudo-random bursts)
                    promo_prob = 0.14 if dow in (1, 2, 4) else 0.06
                    promo_active = int(rng.random() < promo_prob)
                    promo_uplift = 1.28 if promo_active else 1.0

                    # Weather severity index (higher during monsoon doy 165..255)
                    monsoon_boost = 0.35 if 165 <= day_of_year <= 255 else 0.08
                    weather_severity = float(
                        np.clip(rng.beta(2.0, 5.0) + monsoon_boost * rng.random(), 0.02, 0.98)
                    )

                    # Price index modulation
                    price_discount = 0.10 * promo_active + rng.normal(0.0, 0.015)
                    effective_price_index = float(np.clip(1.0 - price_discount, 0.82, 1.08))
                    price_effect = effective_price_index ** (-0.85)

                    # Weather slightly dampens retail/apparel, boosts pharma slightly
                    weather_effect = (
                        1.0 + 0.08 * weather_severity
                        if sku.category == "Pharmaceuticals"
                        else 1.0 - 0.11 * weather_severity
                    )

                    # Autoregressive AR(1) innovation
                    innovation = rng.normal(0.0, 0.065)
                    ar_state = 0.55 * ar_state + innovation

                    expected_demand = (
                        base_level
                        * weekly_factor
                        * seasonal_factor
                        * trend_factor
                        * promo_uplift
                        * price_effect
                        * weather_effect
                        * math.exp(ar_state)
                    )

                    demand_units = max(5.0, round(expected_demand, 1))

                    records.append(
                        {
                            "date": current_date.isoformat(),
                            "sku_id": sku.sku_id,
                            "sku_name": sku.name,
                            "category": sku.category,
                            "hub_id": hub.hub_id,
                            "hub_name": hub.name,
                            "day_of_week": dow,
                            "day_of_year": day_of_year,
                            "month": current_date.month,
                            "is_weekend": int(dow >= 5),
                            "promo_active": promo_active,
                            "price_index": round(effective_price_index, 4),
                            "weather_severity": round(weather_severity, 4),
                            "lead_time_days": round(
                                max(
                                    1.0,
                                    rng.normal(
                                        sku.lead_time_days_mean, sku.lead_time_days_std * 0.5
                                    ),
                                ),
                                2,
                            ),
                            "demand_units": demand_units,
                        }
                    )

        return pd.DataFrame(records)

    def generate_shipment_telemetry(self, n_samples: int = 3500) -> pd.DataFrame:
        """
        Generate realistic historical shipment records for training the
        ETA regression and SLA Delay Risk classification models.
        """
        rng = np.random.default_rng(self.seed + 101)
        customer_nodes = [n for n in self.nodes if not n.is_depot]
        depot = next(n for n in self.nodes if n.is_depot)

        rows: List[Dict[str, float | int | str]] = []
        for i in range(n_samples):
            origin = depot if rng.random() < 0.65 else customer_nodes[rng.integers(0, len(customer_nodes))]
            dest = customer_nodes[rng.integers(0, len(customer_nodes))]
            while dest.hub_id == origin.hub_id:
                dest = customer_nodes[rng.integers(0, len(customer_nodes))]

            base_dist = road_distance_km(origin.lat, origin.lon, dest.lat, dest.lon)
            # Also include regional medium-haul legs (up to 160 km) for broader coverage
            if rng.random() < 0.25:
                dist_km = float(np.clip(base_dist * rng.uniform(1.2, 3.2), 6.0, 180.0))
            else:
                dist_km = float(np.clip(base_dist * rng.uniform(0.92, 1.18), 4.0, 120.0))

            departure_hour = float(np.round(rng.uniform(5.5, 21.0), 2))
            day_of_week = int(rng.integers(0, 7))

            # Rush hour peaks around 08:30-11:00 and 17:00-20:00 on weekdays
            morning_peak = math.exp(-0.5 * ((departure_hour - 9.5) / 1.6) ** 2)
            evening_peak = math.exp(-0.5 * ((departure_hour - 18.2) / 1.8) ** 2)
            weekday_mult = 1.0 if day_of_week < 5 else 0.58

            base_traffic = (0.22 + 0.58 * max(morning_peak, evening_peak)) * weekday_mult
            traffic_congestion = float(
                np.clip(base_traffic + rng.normal(0.0, 0.12), 0.04, 0.98)
            )
            weather_severity = float(np.clip(rng.beta(1.8, 5.2), 0.01, 0.97))
            payload_weight_kg = float(np.round(rng.uniform(350.0, 6200.0), 1))

            # Free-flow travel speed ~44 km/h + loading/dock buffer
            free_flow_min = (dist_km / 44.0) * 60.0 + 10.0

            # Non-linear congestion & weather slowdowns + heavy payload penalty
            traffic_penalty_min = free_flow_min * (
                0.38 * traffic_congestion + 0.45 * (traffic_congestion**2)
            )
            weather_penalty_min = free_flow_min * (
                0.24 * weather_severity + 0.32 * traffic_congestion * weather_severity
            )
            payload_penalty_min = 0.0014 * payload_weight_kg + (dist_km * 0.04 * (payload_weight_kg / 6000.0))
            hub_dwell_min = 6.0 + 8.5 * traffic_congestion + rng.normal(0.0, 2.2)

            actual_duration_min = max(
                12.0,
                free_flow_min
                + traffic_penalty_min
                + weather_penalty_min
                + payload_penalty_min
                + hub_dwell_min,
            )

            # Promised SLA includes standard buffer over free-flow
            promised_sla_min = free_flow_min * 1.38 + 12.0
            sla_breached = int(actual_duration_min > promised_sla_min)

            rows.append(
                {
                    "origin_hub_id": origin.hub_id,
                    "destination_hub_id": dest.hub_id,
                    "distance_km": round(dist_km, 2),
                    "departure_hour": departure_hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(day_of_week >= 5),
                    "traffic_congestion_index": round(traffic_congestion, 4),
                    "weather_severity_index": round(weather_severity, 4),
                    "payload_weight_kg": payload_weight_kg,
                    "free_flow_min": round(free_flow_min, 2),
                    "promised_sla_min": round(promised_sla_min, 2),
                    "actual_duration_min": round(float(actual_duration_min), 2),
                    "sla_breached": sla_breached,
                }
            )

        return pd.DataFrame(rows)

    def generate_daily_dispatch_nodes(
        self,
        demand_multiplier: float = 1.0,
    ) -> List[DeliveryNode]:
        """
        Build daily delivery manifest for the CVRPTW solver across the regional network.
        """
        delivery_nodes: List[DeliveryNode] = []

        # Deterministic time windows & priority profiles for each node
        time_windows = [
            (6.0, 20.0, "DEPOT"),
            (7.5, 12.5, "HIGH"),       # Salt Lake Tech & Pharma Park
            (8.0, 14.0, "HIGH"),       # New Town Action Area II
            (7.0, 15.0, "STANDARD"),   # Dhulagarh Freight Terminal
            (8.5, 13.5, "CRITICAL"),   # Park Street Central Micro-Fulfillment
            (7.0, 14.5, "HIGH"),       # Kolkata Port CFS
            (8.0, 15.5, "STANDARD"),   # Behala Cold-Chain Node
            (8.5, 16.5, "STANDARD"),   # Garia Retail Hub
            (9.0, 17.5, "STANDARD"),   # Baruipur Consolidation Center
            (7.5, 16.0, "STANDARD"),   # Budge Budge Bulk Terminal
            (7.0, 12.0, "CRITICAL"),   # Netaji Subhash Air Cargo Gateway
            (8.0, 16.0, "STANDARD"),   # Barrackpore Industrial Corridor
            (8.5, 16.5, "STANDARD"),   # Barasat NH-12 Cross-Dock
            (8.0, 15.5, "STANDARD"),   # Serampore Logistics Annex
            (7.5, 16.5, "HIGH"),       # Uluberia Manufacturing Park
            (9.0, 18.0, "STANDARD"),   # Kalyani Biotech & Cold Hub
        ]

        for idx, hub in enumerate(self.nodes):
            tw_start, tw_end, priority = time_windows[idx % len(time_windows)]
            if hub.is_depot:
                delivery_nodes.append(
                    DeliveryNode(
                        node_id=hub.hub_id,
                        name=hub.name,
                        city=hub.city,
                        lat=hub.lat,
                        lon=hub.lon,
                        is_depot=True,
                        demand_kg=0.0,
                        demand_volume_m3=0.0,
                        demand_units=0,
                        time_window_start_hr=6.0,
                        time_window_end_hr=20.0,
                        service_duration_min=0.0,
                        priority="DEPOT",
                    )
                )
            else:
                scaled_units = max(40, int(round(hub.base_daily_demand * demand_multiplier)))
                # Average blended weight ~2.45 kg per unit across SKUs
                demand_kg = round(scaled_units * 2.45, 1)
                demand_vol = round(scaled_units * 0.014, 2)
                service_min = round(14.0 + 0.008 * demand_kg, 1)

                delivery_nodes.append(
                    DeliveryNode(
                        node_id=hub.hub_id,
                        name=hub.name,
                        city=hub.city,
                        lat=hub.lat,
                        lon=hub.lon,
                        is_depot=False,
                        demand_kg=demand_kg,
                        demand_volume_m3=demand_vol,
                        demand_units=scaled_units,
                        time_window_start_hr=tw_start,
                        time_window_end_hr=tw_end,
                        service_duration_min=service_min,
                        priority=priority,
                    )
                )

        return delivery_nodes
