"""
Centralized configuration, network topology definitions, SKU parameters,
and algorithmic hyperparameters for LogiRoute AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class HubConfig:
    hub_id: str
    name: str
    city: str
    region: str
    lat: float
    lon: float
    is_depot: bool
    storage_capacity_units: int
    base_daily_demand: float
    handling_cost_per_unit: float


@dataclass(frozen=True)
class SKUConfig:
    sku_id: str
    name: str
    category: str
    unit_weight_kg: float
    unit_volume_m3: float
    unit_cost_usd: float
    unit_price_usd: float
    holding_cost_rate_annual: float
    lead_time_days_mean: float
    lead_time_days_std: float
    ordering_cost_usd: float


@dataclass(frozen=True)
class VehicleClassConfig:
    class_id: str
    name: str
    capacity_kg: float
    capacity_m3: float
    cost_per_km_usd: float
    fixed_dispatch_cost_usd: float
    avg_speed_kmh: float
    co2_kg_per_km: float
    max_route_hours: float


# Regional distribution hubs & customer delivery nodes (NCR / North-Central Hub Cluster
# for realistic daily/intra-day CVRPTW routing, plus Pan-India macro corridor nodes)
DEFAULT_REGIONAL_NODES: List[HubConfig] = [
    HubConfig(
        hub_id="DEPOT-KOL-01",
        name="Dankuni Mega Fulfillment Hub (Central Depot)",
        city="Kolkata Metropolitan",
        region="East Corridor",
        lat=22.6764,
        lon=88.2924,
        is_depot=True,
        storage_capacity_units=85000,
        base_daily_demand=0.0,
        handling_cost_per_unit=0.45,
    ),
    HubConfig(
        hub_id="NODE-SLT-01",
        name="Salt Lake Sector V Tech & Pharma Park",
        city="Bidhannagar",
        region="East Corridor",
        lat=22.5769,
        lon=88.4337,
        is_depot=False,
        storage_capacity_units=12000,
        base_daily_demand=420.0,
        handling_cost_per_unit=0.55,
    ),
    HubConfig(
        hub_id="NODE-NWT-02",
        name="New Town Action Area II Distribution Center",
        city="New Town",
        region="East Corridor",
        lat=22.6015,
        lon=88.4671,
        is_depot=False,
        storage_capacity_units=15000,
        base_daily_demand=510.0,
        handling_cost_per_unit=0.50,
    ),
    HubConfig(
        hub_id="NODE-HWH-03",
        name="Dhulagarh Logistics & Freight Terminal",
        city="Howrah",
        region="East Corridor",
        lat=22.5678,
        lon=88.1745,
        is_depot=False,
        storage_capacity_units=24000,
        base_daily_demand=680.0,
        handling_cost_per_unit=0.42,
    ),
    HubConfig(
        hub_id="NODE-PKR-04",
        name="Park Street & Central Retail Micro-Fulfillment",
        city="Kolkata Central",
        region="East Corridor",
        lat=22.5513,
        lon=88.3526,
        is_depot=False,
        storage_capacity_units=8500,
        base_daily_demand=390.0,
        handling_cost_per_unit=0.68,
    ),
    HubConfig(
        hub_id="NODE-KDP-05",
        name="Kolkata Port (Syama Prasad Mookerjee Dock) CFS",
        city="Khidirpur",
        region="East Corridor",
        lat=22.5366,
        lon=88.3168,
        is_depot=False,
        storage_capacity_units=30000,
        base_daily_demand=740.0,
        handling_cost_per_unit=0.48,
    ),
    HubConfig(
        hub_id="NODE-BEL-06",
        name="Behala Industrial Estate Cold-Chain Node",
        city="Behala",
        region="East Corridor",
        lat=22.4986,
        lon=88.3108,
        is_depot=False,
        storage_capacity_units=11000,
        base_daily_demand=340.0,
        handling_cost_per_unit=0.58,
    ),
    HubConfig(
        hub_id="NODE-GAR-07",
        name="Garia & Southern Bypass Retail Hub",
        city="Garia",
        region="East Corridor",
        lat=22.4660,
        lon=88.3928,
        is_depot=False,
        storage_capacity_units=9500,
        base_daily_demand=310.0,
        handling_cost_per_unit=0.54,
    ),
    HubConfig(
        hub_id="NODE-BRP-08",
        name="Baruipur Agri & FMCG Consolidation Center",
        city="Baruipur",
        region="East Corridor",
        lat=22.3654,
        lon=88.4325,
        is_depot=False,
        storage_capacity_units=14000,
        base_daily_demand=285.0,
        handling_cost_per_unit=0.46,
    ),
    HubConfig(
        hub_id="NODE-BDG-09",
        name="Budge Budge Riverine & Bulk Terminal",
        city="Budge Budge",
        region="East Corridor",
        lat=22.4828,
        lon=88.1818,
        is_depot=False,
        storage_capacity_units=18000,
        base_daily_demand=460.0,
        handling_cost_per_unit=0.44,
    ),
    HubConfig(
        hub_id="NODE-DUM-10",
        name="Netaji Subhash Air Cargo Express Gateway",
        city="Dum Dum",
        region="East Corridor",
        lat=22.6520,
        lon=88.4463,
        is_depot=False,
        storage_capacity_units=20000,
        base_daily_demand=615.0,
        handling_cost_per_unit=0.62,
    ),
    HubConfig(
        hub_id="NODE-BRK-11",
        name="Barrackpore Northern Industrial Corridor",
        city="Barrackpore",
        region="East Corridor",
        lat=22.7674,
        lon=88.3702,
        is_depot=False,
        storage_capacity_units=13500,
        base_daily_demand=365.0,
        handling_cost_per_unit=0.49,
    ),
    HubConfig(
        hub_id="NODE-BSR-12",
        name="Barasat NH-12 Cross-Docking Station",
        city="Barasat",
        region="East Corridor",
        lat=22.7225,
        lon=88.4806,
        is_depot=False,
        storage_capacity_units=16500,
        base_daily_demand=445.0,
        handling_cost_per_unit=0.47,
    ),
    HubConfig(
        hub_id="NODE-SRM-13",
        name="Serampore Hooghly Logistics Annex",
        city="Serampore",
        region="East Corridor",
        lat=22.7516,
        lon=88.3426,
        is_depot=False,
        storage_capacity_units=10500,
        base_daily_demand=295.0,
        handling_cost_per_unit=0.51,
    ),
    HubConfig(
        hub_id="NODE-ULB-14",
        name="Uluberia NH-16 Heavy Manufacturing Park",
        city="Uluberia",
        region="East Corridor",
        lat=22.4744,
        lon=88.1000,
        is_depot=False,
        storage_capacity_units=22000,
        base_daily_demand=530.0,
        handling_cost_per_unit=0.43,
    ),
    HubConfig(
        hub_id="NODE-KLY-15",
        name="Kalyani Expressway Biotech & Cold Hub",
        city="Kalyani",
        region="East Corridor",
        lat=22.9751,
        lon=88.4345,
        is_depot=False,
        storage_capacity_units=17500,
        base_daily_demand=380.0,
        handling_cost_per_unit=0.52,
    ),
]


DEFAULT_SKUS: List[SKUConfig] = [
    SKUConfig(
        sku_id="SKU-PHM-101",
        name="Cold-Chain Biologics & Insulin Pack",
        category="Pharmaceuticals",
        unit_weight_kg=1.8,
        unit_volume_m3=0.008,
        unit_cost_usd=42.0,
        unit_price_usd=68.0,
        holding_cost_rate_annual=0.28,
        lead_time_days_mean=4.5,
        lead_time_days_std=1.2,
        ordering_cost_usd=180.0,
    ),
    SKUConfig(
        sku_id="SKU-ELC-204",
        name="Industrial IoT Telemetry Gateway",
        category="Electronics",
        unit_weight_kg=3.2,
        unit_volume_m3=0.015,
        unit_cost_usd=115.0,
        unit_price_usd=179.0,
        holding_cost_rate_annual=0.24,
        lead_time_days_mean=7.0,
        lead_time_days_std=2.1,
        ordering_cost_usd=260.0,
    ),
    SKUConfig(
        sku_id="SKU-FMC-309",
        name="Packaged Essential Nutrition Crate",
        category="FMCG & Grocery",
        unit_weight_kg=8.5,
        unit_volume_m3=0.028,
        unit_cost_usd=14.5,
        unit_price_usd=22.0,
        holding_cost_rate_annual=0.20,
        lead_time_days_mean=3.0,
        lead_time_days_std=0.8,
        ordering_cost_usd=95.0,
    ),
    SKUConfig(
        sku_id="SKU-AUT-412",
        name="EV Battery Thermal Module Assembly",
        category="Automotive & EV",
        unit_weight_kg=14.0,
        unit_volume_m3=0.042,
        unit_cost_usd=210.0,
        unit_price_usd=315.0,
        holding_cost_rate_annual=0.22,
        lead_time_days_mean=9.0,
        lead_time_days_std=2.6,
        ordering_cost_usd=340.0,
    ),
    SKUConfig(
        sku_id="SKU-APP-518",
        name="Omnichannel Retail Apparel Carton",
        category="Apparel & Softlines",
        unit_weight_kg=5.0,
        unit_volume_m3=0.032,
        unit_cost_usd=28.0,
        unit_price_usd=49.0,
        holding_cost_rate_annual=0.25,
        lead_time_days_mean=5.5,
        lead_time_days_std=1.5,
        ordering_cost_usd=130.0,
    ),
]


DEFAULT_FLEET_CLASSES: List[VehicleClassConfig] = [
    VehicleClassConfig(
        class_id="VEH-HVY-16T",
        name="Heavy Linehaul Refrigerated Truck (16T)",
        capacity_kg=6500.0,
        capacity_m3=34.0,
        cost_per_km_usd=1.42,
        fixed_dispatch_cost_usd=85.0,
        avg_speed_kmh=38.0,
        co2_kg_per_km=0.78,
        max_route_hours=10.0,
    ),
    VehicleClassConfig(
        class_id="VEH-MED-9T",
        name="Medium Regional Box Truck (9T)",
        capacity_kg=4500.0,
        capacity_m3=24.0,
        cost_per_km_usd=1.08,
        fixed_dispatch_cost_usd=60.0,
        avg_speed_kmh=41.0,
        co2_kg_per_km=0.54,
        max_route_hours=9.5,
    ),
    VehicleClassConfig(
        class_id="VEH-EV-4T",
        name="Zero-Emission Electric Cargo Van (4.5T)",
        capacity_kg=2800.0,
        capacity_m3=16.0,
        cost_per_km_usd=0.68,
        fixed_dispatch_cost_usd=42.0,
        avg_speed_kmh=44.0,
        co2_kg_per_km=0.14,  # Grid-equivalent lifecycle emissions
        max_route_hours=8.5,
    ),
]


@dataclass
class EngineSettings:
    random_seed: int = 42
    historical_days: int = 365
    forecast_horizon_days: int = 14
    road_tortuosity_factor: float = 1.28  # Realistic urban/peri-urban road winding vs Haversine
    default_service_level: float = 0.96
    depot_open_hour: float = 6.0  # 06:00 AM
    depot_close_hour: float = 20.0  # 08:00 PM


SETTINGS = EngineSettings()
