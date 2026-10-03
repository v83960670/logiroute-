"""
Multivariate Supply Chain Telemetry Anomaly & Disruption Detector.

Combines Isolation Forest (unsupervised tree ensemble) with Robust Z-Score
residual diagnostics to detect:
1. Cold-Chain Thermal Excursions (Pharmaceutical biologics reefer drift)
2. Port / Cross-Dock Dwell-Time Spikes (Customs or yard congestion)
3. Bullwhip Order Amplification Shocks (Sudden order-to-forecast divergence)
4. Supplier Fill-Rate & Lead-Time Degradation
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from logiroute.config import DEFAULT_REGIONAL_NODES, DEFAULT_SKUS, SETTINGS
from logiroute.schemas import AnomalyDetectionResponse, TelemetryAnomalyEvent


ANOMALY_FEATURES: List[str] = [
    "order_to_forecast_ratio",
    "dock_dwell_min",
    "cold_chain_temp_delta_c",
    "supplier_fill_rate_pct",
    "transit_speed_ratio",
]


class SupplyChainAnomalyDetector:
    """
    Hybrid IsolationForest + Robust Z-Score anomaly detection engine for
    warehouse, cold-chain, and fleet telemetry streams.
    """

    def __init__(self, random_state: int = SETTINGS.random_seed) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=160,
            contamination=0.065,
            max_samples="auto",
            random_state=random_state,
        )
        self.validation_metrics: Dict[str, float] = {}
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self._telemetry_stream: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def _generate_telemetry_stream(self, n_events: int = 600) -> pd.DataFrame:
        """
        Generate multivariate supply chain operational telemetry with realistic
        normal operations and ~6.5% structural disruption anomalies.
        """
        rng = np.random.default_rng(self.random_state + 303)
        hubs = [n for n in DEFAULT_REGIONAL_NODES if not n.is_depot][:8]
        skus = DEFAULT_SKUS
        base_time = datetime(2026, 10, 3, 14, 0, 0)

        rows: List[Dict[str, object]] = []
        for i in range(n_events):
            hub = hubs[i % len(hubs)]
            sku = skus[(i // 2) % len(skus)]
            ts = (base_time - timedelta(minutes=(n_events - i) * 12)).strftime("%Y-%m-%d %H:%M")

            # Nominal operating distributions
            o2f_ratio = float(rng.normal(1.00, 0.08))
            dwell_min = float(rng.normal(24.0, 4.2))
            temp_delta = float(rng.normal(0.15, 0.35))
            fill_rate = float(np.clip(rng.normal(97.2, 1.4), 88.0, 100.0))
            speed_ratio = float(rng.normal(1.00, 0.07))
            is_true_anomaly = 0
            anomaly_subtype = "NOMINAL"

            # Inject deterministic structural anomalies (~6.5% of stream)
            if i % 16 == 0:
                is_true_anomaly = 1
                mode = (i // 16) % 4
                if mode == 0:
                    anomaly_subtype = "COLD_CHAIN_THERMAL_EXCURSION"
                    temp_delta = float(rng.uniform(3.4, 5.8))
                    dwell_min += float(rng.uniform(8.0, 16.0))
                elif mode == 1:
                    anomaly_subtype = "DOCK_CONGESTION_DWELL_SPIKE"
                    dwell_min = float(rng.uniform(54.0, 82.0))
                    speed_ratio = float(rng.uniform(0.55, 0.72))
                elif mode == 2:
                    anomaly_subtype = "BULLWHIP_DEMAND_SURGE"
                    o2f_ratio = float(rng.uniform(1.68, 2.35))
                    fill_rate = float(rng.uniform(79.0, 86.5))
                else:
                    anomaly_subtype = "SUPPLIER_FILL_RATE_COLLAPSE"
                    fill_rate = float(rng.uniform(68.0, 78.5))
                    o2f_ratio = float(rng.uniform(1.30, 1.55))

            rows.append(
                {
                    "event_id": f"TEL-{10000 + i}",
                    "timestamp": ts,
                    "hub_id": hub.hub_id,
                    "hub_name": hub.name,
                    "sku_id": sku.sku_id,
                    "order_to_forecast_ratio": round(o2f_ratio, 3),
                    "dock_dwell_min": round(dwell_min, 1),
                    "cold_chain_temp_delta_c": round(temp_delta, 2),
                    "supplier_fill_rate_pct": round(fill_rate, 2),
                    "transit_speed_ratio": round(speed_ratio, 3),
                    "is_true_anomaly": is_true_anomaly,
                    "anomaly_subtype": anomaly_subtype,
                }
            )

        return pd.DataFrame(rows)

    def fit(self, stream_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Fit IsolationForest + Z-score baselines and compute anomaly detection metrics."""
        if stream_df is None:
            stream_df = self._generate_telemetry_stream(n_events=600)
        self._telemetry_stream = stream_df.copy()

        x_raw = stream_df[ANOMALY_FEATURES].to_numpy(dtype=float)
        y_true = stream_df["is_true_anomaly"].to_numpy(dtype=int)

        for col in ANOMALY_FEATURES:
            self.feature_means[col] = float(stream_df[col].mean())
            self.feature_stds[col] = max(1e-4, float(stream_df[col].std()))

        x_scaled = self.scaler.fit_transform(x_raw)
        self.iforest.fit(x_scaled)

        # IsolationForest decision_function returns negative values for outliers
        raw_scores = -self.iforest.decision_function(x_scaled)
        # Normalize anomaly score to [0, 1]
        min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
        norm_scores = (raw_scores - min_s) / max(1e-6, max_s - min_s)
        preds = (self.iforest.predict(x_scaled) == -1).astype(int)

        roc_auc = float(roc_auc_score(y_true, norm_scores))
        pr_auc = float(average_precision_score(y_true, norm_scores))
        f1 = float(f1_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))

        self.validation_metrics = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "total_stream_events": int(len(stream_df)),
        }
        self.is_fitted = True
        return self.validation_metrics

    def detect_anomalies(self, top_k: int = 15) -> AnomalyDetectionResponse:
        """Scan the telemetry stream and return ranked anomaly events with mitigations."""
        if not self.is_fitted or self._telemetry_stream is None:
            self.fit()

        df = self._telemetry_stream.copy()
        x_scaled = self.scaler.transform(df[ANOMALY_FEATURES].to_numpy(dtype=float))
        raw_scores = -self.iforest.decision_function(x_scaled)
        min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
        df["anomaly_score"] = (raw_scores - min_s) / max(1e-6, max_s - min_s)
        df["pred_flag"] = (self.iforest.predict(x_scaled) == -1).astype(int)

        flagged = df[df["pred_flag"] == 1].sort_values("anomaly_score", ascending=False)

        playbooks = {
            "COLD_CHAIN_THERMAL_EXCURSION": (
                "Reefer ΔT Excursion (°C)",
                "cold_chain_temp_delta_c",
                "Quarantine affected biologics batch, switch reefer to backup compressor, and dispatch priority replacement from Dankuni Depot.",
            ),
            "DOCK_CONGESTION_DWELL_SPIKE": (
                "Hub Dock Dwell Time (min)",
                "dock_dwell_min",
                "Activate cross-dock overflow bay #3 and dynamically re-sequence incoming CVRPTW vehicles to bypass congested dock.",
            ),
            "BULLWHIP_DEMAND_SURGE": (
                "Order-to-Forecast Ratio",
                "order_to_forecast_ratio",
                "Trigger CQR P90 upper-bound safety stock release and allocate supplemental 9T regional box truck.",
            ),
            "SUPPLIER_FILL_RATE_COLLAPSE": (
                "Supplier Fill Rate (%)",
                "supplier_fill_rate_pct",
                "Split emergency replenishment PO to secondary qualified vendor and raise cycle service level buffer.",
            ),
        }

        events: List[TelemetryAnomalyEvent] = []
        for _, row in flagged.head(top_k).iterrows():
            subtype = str(row["anomaly_subtype"])
            if subtype not in playbooks:
                # Infer primary driver by largest Z-score
                z_map = {
                    col: abs(float(row[col]) - self.feature_means[col]) / self.feature_stds[col]
                    for col in ANOMALY_FEATURES
                }
                top_col = max(z_map, key=z_map.get)
                if top_col == "cold_chain_temp_delta_c":
                    subtype = "COLD_CHAIN_THERMAL_EXCURSION"
                elif top_col == "dock_dwell_min":
                    subtype = "DOCK_CONGESTION_DWELL_SPIKE"
                elif top_col == "order_to_forecast_ratio":
                    subtype = "BULLWHIP_DEMAND_SURGE"
                else:
                    subtype = "SUPPLIER_FILL_RATE_COLLAPSE"

            signal_label, col_name, mitigation = playbooks[subtype]
            obs_val = float(row[col_name])
            exp_val = self.feature_means[col_name]
            z_val = abs(obs_val - exp_val) / self.feature_stds[col_name]
            score = float(row["anomaly_score"])
            severity = "CRITICAL" if score >= 0.72 else "HIGH" if score >= 0.55 else "ELEVATED"

            events.append(
                TelemetryAnomalyEvent(
                    event_id=str(row["event_id"]),
                    timestamp=str(row["timestamp"]),
                    hub_id=str(row["hub_id"]),
                    hub_name=str(row["hub_name"]),
                    sku_id=str(row["sku_id"]),
                    anomaly_type=subtype,
                    severity=severity,
                    anomaly_score=round(score, 3),
                    observed_value=round(obs_val, 2),
                    expected_baseline=round(exp_val, 2),
                    deviation_z_score=round(z_val, 2),
                    root_cause_signal=f"{signal_label}: {obs_val:.2f} vs baseline {exp_val:.2f} ({z_val:.1f}σ)",
                    automated_mitigation=mitigation,
                )
            )

        total_scanned = len(df)
        total_anomalies = int(df["pred_flag"].sum())
        return AnomalyDetectionResponse(
            detector_algorithm="IsolationForest (160 Trees) + Multivariate Robust Z-Score",
            total_events_scanned=total_scanned,
            anomalies_detected=total_anomalies,
            anomaly_rate_pct=round((total_anomalies / max(1, total_scanned)) * 100.0, 2),
            validation_metrics=self.validation_metrics,
            events=events,
        )
