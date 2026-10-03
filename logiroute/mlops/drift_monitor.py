"""
Production MLOps Covariate & Concept Drift Monitor.

Evaluates distribution shift between reference training distributions and
incoming production inference batches using:
1. Population Stability Index (PSI):
     PSI = sum_b (P_b - Q_b) * ln(P_b / Q_b)
2. Two-Sample Kolmogorov-Smirnov (KS) Test (statistic D_KS and p-value)
3. Normalized 1-Wasserstein (Earth Mover's) Distance
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from logiroute.config import SETTINGS
from logiroute.schemas import FeatureDriftMetric, MLOpsDriftResponse


class MLOpsDriftMonitor:
    """Computes PSI, KS-test, and Wasserstein drift metrics across production features."""

    def __init__(self, seed: int = SETTINGS.random_seed) -> None:
        self.seed = seed
        self._reference_distributions: Dict[str, np.ndarray] = {}
        self._init_reference_distributions()

    def _init_reference_distributions(self) -> None:
        rng = np.random.default_rng(self.seed + 505)
        n_ref = 1500
        self._reference_distributions = {
            "demand_units": np.clip(rng.normal(148.0, 38.0, n_ref), 15.0, 420.0),
            "traffic_congestion_index": np.clip(rng.beta(2.4, 3.6, n_ref), 0.04, 0.98),
            "weather_severity_index": np.clip(rng.beta(1.8, 5.2, n_ref), 0.01, 0.97),
            "lead_time_days": np.clip(rng.normal(5.8, 1.6, n_ref), 1.5, 14.0),
            "payload_weight_kg": np.clip(rng.normal(3150.0, 920.0, n_ref), 350.0, 6500.0),
            "predicted_eta_min": np.clip(rng.normal(52.0, 14.5, n_ref), 14.0, 145.0),
        }

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) using decile bins derived from
        the reference training distribution.
        """
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        bin_edges = np.percentile(reference, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        # Deduplicate edges if needed
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        eps = 1e-4
        ref_props = np.maximum(eps, ref_counts / max(1, len(reference)))
        cur_props = np.maximum(eps, cur_counts / max(1, len(current)))

        psi_val = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
        return max(0.0, psi_val)

    def evaluate_drift(self, drift_regime: str = "nominal") -> MLOpsDriftResponse:
        """
        Evaluate feature & prediction drift under the selected telemetry regime:
          - 'nominal'           : Normal in-distribution stream (PSI < 0.08)
          - 'seasonal_shift'    : Moderate festive/seasonal shift (0.10 <= PSI < 0.25)
          - 'monsoon_cov_shift' : Severe monsoon & bullwhip shift (PSI >= 0.25)
        """
        rng = np.random.default_rng(self.seed + 808)
        n_cur = 800

        shift_profiles: Dict[str, Dict[str, tuple[float, float]]] = {
            "nominal": {
                "demand_units": (1.01, 1.02),
                "traffic_congestion_index": (1.02, 1.00),
                "weather_severity_index": (0.99, 1.01),
                "lead_time_days": (1.01, 1.00),
                "payload_weight_kg": (1.00, 0.99),
                "predicted_eta_min": (1.02, 1.01),
            },
            "seasonal_shift": {
                "demand_units": (1.16, 1.15),
                "traffic_congestion_index": (1.18, 1.10),
                "weather_severity_index": (1.06, 1.05),
                "lead_time_days": (1.12, 1.10),
                "payload_weight_kg": (1.14, 1.08),
                "predicted_eta_min": (1.15, 1.12),
            },
            "monsoon_cov_shift": {
                "demand_units": (1.32, 1.28),
                "traffic_congestion_index": (1.48, 1.22),
                "weather_severity_index": (1.85, 1.40),
                "lead_time_days": (1.42, 1.30),
                "payload_weight_kg": (1.22, 1.18),
                "predicted_eta_min": (1.38, 1.25),
            },
        }

        profile = shift_profiles.get(drift_regime, shift_profiles["nominal"])
        feature_metrics: List[FeatureDriftMetric] = []
        critical_count = 0
        moderate_count = 0

        for feat_name, ref_arr in self._reference_distributions.items():
            mean_mult, std_mult = profile[feat_name]
            ref_mean = float(np.mean(ref_arr))
            ref_std = max(1e-4, float(np.std(ref_arr)))

            # Sample live batch with regime shift
            sampled_indices = rng.integers(0, len(ref_arr), size=n_cur)
            base_sample = ref_arr[sampled_indices]
            noise = rng.normal(0.0, ref_std * 0.12 * std_mult, size=n_cur)
            cur_arr = (base_sample - ref_mean) * std_mult + (ref_mean * mean_mult) + noise

            if "index" in feat_name:
                cur_arr = np.clip(cur_arr, 0.01, 0.99)
            else:
                cur_arr = np.maximum(1.0, cur_arr)

            cur_mean = float(np.mean(cur_arr))
            shift_pct = ((cur_mean - ref_mean) / max(1e-4, abs(ref_mean))) * 100.0
            psi_val = self.compute_psi(ref_arr, cur_arr)
            ks_res = ks_2samp(ref_arr, cur_arr)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)
            w_norm = float(wasserstein_distance(ref_arr, cur_arr) / ref_std)

            if psi_val >= 0.25 or ks_stat >= 0.28:
                status = "CRITICAL_DRIFT"
                critical_count += 1
            elif psi_val >= 0.10 or ks_stat >= 0.14:
                status = "MODERATE_DRIFT"
                moderate_count += 1
            else:
                status = "STABLE"

            feature_metrics.append(
                FeatureDriftMetric(
                    feature_name=feat_name,
                    baseline_mean=round(ref_mean, 3),
                    current_mean=round(cur_mean, 3),
                    shift_pct=round(shift_pct, 2),
                    psi_score=round(psi_val, 4),
                    ks_statistic=round(ks_stat, 4),
                    ks_p_value=round(ks_pval, 5),
                    wasserstein_norm=round(w_norm, 4),
                    drift_status=status,
                )
            )

        mean_psi = float(np.mean([f.psi_score for f in feature_metrics]))
        health_score = round(float(np.clip(100.0 - mean_psi * 145.0, 22.0, 99.4)), 1)
        retrain_needed = critical_count >= 1 or moderate_count >= 3

        if retrain_needed:
            reason = (
                f"Detected {critical_count} CRITICAL (PSI >= 0.25) and {moderate_count} MODERATE "
                f"drifted covariates. Trigger automated CQR Quantile GBDT & ETA model retraining DAG."
            )
        elif moderate_count > 0:
            reason = (
                f"Detected {moderate_count} features with moderate seasonal drift (0.10 <= PSI < 0.25). "
                f"Shadow-evaluate challenger model; conformal CQR bounds remain valid."
            )
        else:
            reason = (
                "All 6 production feature and prediction distributions are statistically stable "
                "(PSI < 0.10). No model retraining required."
            )

        return MLOpsDriftResponse(
            drift_regime=drift_regime,
            overall_health_score=health_score,
            retraining_recommended=retrain_needed,
            retraining_trigger_reason=reason,
            monitored_features=feature_metrics,
        )
