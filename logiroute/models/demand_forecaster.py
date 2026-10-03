"""
Probabilistic Multi-Horizon Supply Chain Demand Forecaster with Conformalized
Quantile Regression (CQR).

Implements Quantile Gradient Boosted Decision Trees (HistGradientBoostingRegressor
with pinball/quantile loss at tau = 0.10, 0.50, 0.90) calibrated via Split
Conformalized Quantile Regression (Romano et al., NeurIPS 2019) to guarantee
finite-sample ~80% empirical interval coverage.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_pinball_loss, mean_squared_error, r2_score

from logiroute.config import SETTINGS
from logiroute.schemas import DemandForecastResponse, ForecastDataPoint


FEATURE_COLUMNS: List[str] = [
    "sku_code",
    "hub_code",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "promo_active",
    "price_index",
    "weather_severity",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "ewma_7",
]


class ProbabilisticDemandForecaster:
    """
    Conformalized Quantile Gradient Boosted Forecaster producing calibrated
    P10, P50, and P90 demand distributions across SKU x Distribution Hub pairs.
    """

    def __init__(self, random_state: int = SETTINGS.random_seed) -> None:
        self.random_state = random_state
        self.models: Dict[str, HistGradientBoostingRegressor] = {}
        self.sku_to_code: Dict[str, int] = {}
        self.hub_to_code: Dict[str, int] = {}
        self.cqr_margin: float = 0.0
        self.metrics: Dict[str, float] = {}
        self.feature_importances: Dict[str, float] = {}
        self._raw_history: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct temporal, cyclical, lag, and rolling features per (sku_id, hub_id)."""
        work = df.copy()
        work["date_dt"] = pd.to_datetime(work["date"])
        work = work.sort_values(["sku_id", "hub_id", "date_dt"]).reset_index(drop=True)

        work["sku_code"] = work["sku_id"].map(self.sku_to_code).fillna(0).astype(float)
        work["hub_code"] = work["hub_id"].map(self.hub_to_code).fillna(0).astype(float)

        work["dow_sin"] = np.sin(2.0 * np.pi * work["day_of_week"] / 7.0)
        work["dow_cos"] = np.cos(2.0 * np.pi * work["day_of_week"] / 7.0)
        work["doy_sin"] = np.sin(2.0 * np.pi * work["day_of_year"] / 365.25)
        work["doy_cos"] = np.cos(2.0 * np.pi * work["day_of_year"] / 365.25)

        grouped = work.groupby(["sku_id", "hub_id"], sort=False)["demand_units"]
        work["lag_1"] = grouped.shift(1)
        work["lag_7"] = grouped.shift(7)
        work["lag_14"] = grouped.shift(14)
        work["lag_28"] = grouped.shift(28)

        shifted = grouped.shift(1)
        work["rolling_mean_7"] = (
            shifted.groupby([work["sku_id"], work["hub_id"]])
            .rolling(7, min_periods=2)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        work["rolling_std_7"] = (
            shifted.groupby([work["sku_id"], work["hub_id"]])
            .rolling(7, min_periods=2)
            .std()
            .fillna(2.0)
            .reset_index(level=[0, 1], drop=True)
        )
        work["rolling_mean_14"] = (
            shifted.groupby([work["sku_id"], work["hub_id"]])
            .rolling(14, min_periods=3)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        work["rolling_mean_28"] = (
            shifted.groupby([work["sku_id"], work["hub_id"]])
            .rolling(28, min_periods=5)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        work["ewma_7"] = (
            shifted.groupby([work["sku_id"], work["hub_id"]])
            .transform(lambda s: s.ewm(span=7, min_periods=1).mean())
        )

        return work

    def fit(self, history_df: pd.DataFrame, holdout_days: int = 30) -> Dict[str, float]:
        """
        Train P10, P50, and P90 quantile gradient boosted models, calibrate CQR
        nonconformity scores, and evaluate on an out-of-time holdout window.
        """
        self._raw_history = history_df.copy()
        unique_skus = sorted(history_df["sku_id"].unique())
        unique_hubs = sorted(history_df["hub_id"].unique())
        self.sku_to_code = {s: idx for idx, s in enumerate(unique_skus)}
        self.hub_to_code = {h: idx for idx, h in enumerate(unique_hubs)}

        feat_df = self._build_features(history_df)
        feat_df = feat_df.dropna(subset=FEATURE_COLUMNS + ["demand_units"]).reset_index(drop=True)

        max_date = feat_df["date_dt"].max()
        val_cutoff = max_date - pd.Timedelta(days=holdout_days)
        cal_cutoff = val_cutoff - pd.Timedelta(days=25)

        train_df = feat_df[feat_df["date_dt"] <= cal_cutoff]
        cal_df = feat_df[(feat_df["date_dt"] > cal_cutoff) & (feat_df["date_dt"] <= val_cutoff)]
        val_df = feat_df[feat_df["date_dt"] > val_cutoff]

        x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["demand_units"].to_numpy(dtype=float)
        x_cal = cal_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_cal = cal_df["demand_units"].to_numpy(dtype=float)
        x_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_val = val_df["demand_units"].to_numpy(dtype=float)

        quantiles = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
        for label, q in quantiles.items():
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=135,
                learning_rate=0.075,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=self.random_state,
            )
            model.fit(x_train, y_train)
            self.models[label] = model

        # Split Conformalized Quantile Regression (CQR) calibration on cal_df
        cal_p10 = self.models["p10"].predict(x_cal)
        cal_p90 = self.models["p90"].predict(x_cal)
        conformity_scores = np.maximum(cal_p10 - y_cal, y_cal - cal_p90)
        n_cal = len(conformity_scores)
        target_alpha = 0.20  # 80% prediction interval [P10, P90]
        q_level = min(0.99, (1.0 - target_alpha) * (1.0 + 1.0 / max(1, n_cal)))
        self.cqr_margin = float(np.quantile(conformity_scores, q_level))

        # Out-of-time validation predictions with CQR adjustment
        raw_val_p10 = self.models["p10"].predict(x_val) - self.cqr_margin
        raw_val_p50 = self.models["p50"].predict(x_val)
        raw_val_p90 = self.models["p90"].predict(x_val) + self.cqr_margin

        pred_p10, pred_p50, pred_p90 = self._enforce_monotonic_quantiles(
            raw_val_p10, raw_val_p50, raw_val_p90
        )

        mape = float(np.mean(np.abs((y_val - pred_p50) / np.maximum(y_val, 1.0))) * 100.0)
        wmape = float(np.sum(np.abs(y_val - pred_p50)) / np.maximum(np.sum(y_val), 1.0) * 100.0)
        rmse = float(math.sqrt(mean_squared_error(y_val, pred_p50)))
        r2 = float(r2_score(y_val, pred_p50))
        coverage_80 = float(np.mean((y_val >= pred_p10) & (y_val <= pred_p90)) * 100.0)
        pinball_p50 = float(mean_pinball_loss(y_val, pred_p50, alpha=0.50))

        self.metrics = {
            "mape_pct": round(mape, 2),
            "wmape_pct": round(wmape, 2),
            "rmse_units": round(rmse, 2),
            "r2_score": round(r2, 4),
            "interval_80_coverage_pct": round(coverage_80, 2),
            "cqr_calibration_margin_units": round(self.cqr_margin, 2),
            "pinball_loss_p50": round(pinball_p50, 3),
            "train_samples": int(len(train_df) + len(cal_df)),
            "validation_samples": int(len(val_df)),
        }

        # Compute permutation feature importance on P50 validation set
        perm = permutation_importance(
            self.models["p50"],
            x_val,
            y_val,
            n_repeats=3,
            random_state=self.random_state,
        )
        raw_imp = {
            col: max(0.0, float(score))
            for col, score in zip(FEATURE_COLUMNS, perm.importances_mean)
        }
        total_imp = sum(raw_imp.values()) or 1.0
        sorted_imp = sorted(
            ((k, round((v / total_imp) * 100.0, 2)) for k, v in raw_imp.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        self.feature_importances = dict(sorted_imp[:10])
        self.is_fitted = True
        return self.metrics

    @staticmethod
    def _enforce_monotonic_quantiles(
        p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Guarantee P10 <= P50 <= P90 for every prediction point."""
        stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
        return np.maximum(1.0, stacked[0]), np.maximum(1.0, stacked[1]), np.maximum(1.0, stacked[2])

    def forecast_sku_hub(
        self,
        sku_id: str,
        hub_id: str,
        horizon_days: int = SETTINGS.forecast_horizon_days,
        history_window_days: int = 35,
        demand_multiplier: float = 1.0,
    ) -> DemandForecastResponse:
        """
        Produce historical backtest + recursive multi-step probabilistic forecast
        for a specific (sku_id, hub_id) pair.
        """
        if not self.is_fitted or self._raw_history is None:
            raise RuntimeError("ProbabilisticDemandForecaster must be fitted before forecasting.")

        sub = self._raw_history[
            (self._raw_history["sku_id"] == sku_id)
            & (self._raw_history["hub_id"] == hub_id)
        ].copy()
        if sub.empty:
            sub = self._raw_history.iloc[:90].copy()
            sku_id = str(sub["sku_id"].iloc[0])
            hub_id = str(sub["hub_id"].iloc[0])

        sku_name = str(sub["sku_name"].iloc[0])
        category = str(sub["category"].iloc[0])
        hub_name = str(sub["hub_name"].iloc[0])

        # Build features on historical slice to get in-sample P10/P50/P90
        feat_sub = self._build_features(sub).dropna(subset=FEATURE_COLUMNS).tail(history_window_days)
        x_hist = feat_sub[FEATURE_COLUMNS].to_numpy(dtype=float)
        h_p10, h_p50, h_p90 = self._enforce_monotonic_quantiles(
            self.models["p10"].predict(x_hist) - self.cqr_margin,
            self.models["p50"].predict(x_hist),
            self.models["p90"].predict(x_hist) + self.cqr_margin,
        )

        series_points: List[ForecastDataPoint] = []
        for idx, (_, row) in enumerate(feat_sub.iterrows()):
            actual = float(row["demand_units"])
            p10_v = round(float(h_p10[idx]), 1)
            p50_v = round(float(h_p50[idx]), 1)
            p90_v = round(float(h_p90[idx]), 1)
            anomaly = bool(actual > p90_v * 1.05 or actual < p10_v * 0.95)
            series_points.append(
                ForecastDataPoint(
                    date=str(row["date"]),
                    is_historical=True,
                    actual_demand=round(actual, 1),
                    p10_demand=p10_v,
                    p50_demand=p50_v,
                    p90_demand=p90_v,
                    promo_active=bool(row["promo_active"]),
                    anomaly_flag=anomaly,
                )
            )

        # Recursive multi-horizon rolling forecast
        history_buffer = sub["demand_units"].tolist()
        last_date = date.fromisoformat(str(sub["date"].iloc[-1]))
        sku_code = float(self.sku_to_code.get(sku_id, 0))
        hub_code = float(self.hub_to_code.get(hub_id, 0))

        for step in range(1, horizon_days + 1):
            f_date = last_date + timedelta(days=step)
            dow = f_date.weekday()
            doy = f_date.timetuple().tm_yday
            is_weekend = float(dow >= 5)
            promo_active = 1.0 if step in (4, 5, 11) else 0.0
            price_index = 0.91 if promo_active > 0 else 1.0
            weather_severity = 0.22

            lag_1 = history_buffer[-1]
            lag_7 = history_buffer[-7] if len(history_buffer) >= 7 else lag_1
            lag_14 = history_buffer[-14] if len(history_buffer) >= 14 else lag_7
            lag_28 = history_buffer[-28] if len(history_buffer) >= 28 else lag_14

            recent_7 = history_buffer[-7:]
            recent_14 = history_buffer[-14:]
            recent_28 = history_buffer[-28:]

            rolling_mean_7 = float(np.mean(recent_7))
            rolling_std_7 = float(np.std(recent_7, ddof=1)) if len(recent_7) > 1 else 2.0
            rolling_mean_14 = float(np.mean(recent_14))
            rolling_mean_28 = float(np.mean(recent_28))
            ewma_7 = float(pd.Series(recent_14).ewm(span=7, min_periods=1).mean().iloc[-1])

            x_step = np.array(
                [
                    [
                        sku_code,
                        hub_code,
                        math.sin(2.0 * math.pi * dow / 7.0),
                        math.cos(2.0 * math.pi * dow / 7.0),
                        math.sin(2.0 * math.pi * doy / 365.25),
                        math.cos(2.0 * math.pi * doy / 365.25),
                        is_weekend,
                        promo_active,
                        price_index,
                        weather_severity,
                        lag_1,
                        lag_7,
                        lag_14,
                        lag_28,
                        rolling_mean_7,
                        rolling_std_7,
                        rolling_mean_14,
                        rolling_mean_28,
                        ewma_7,
                    ]
                ],
                dtype=float,
            )

            raw_p10 = (float(self.models["p10"].predict(x_step)[0]) - self.cqr_margin) * demand_multiplier
            raw_p50 = float(self.models["p50"].predict(x_step)[0]) * demand_multiplier
            raw_p90 = (float(self.models["p90"].predict(x_step)[0]) + self.cqr_margin) * demand_multiplier

            horizon_spread = 1.0 + 0.015 * (step - 1)
            p50_val = max(2.0, raw_p50)
            p10_val = max(1.0, min(p50_val * 0.94, p50_val - (p50_val - raw_p10) * horizon_spread))
            p90_val = max(p50_val * 1.06, p50_val + (raw_p90 - p50_val) * horizon_spread)

            history_buffer.append(p50_val)
            series_points.append(
                ForecastDataPoint(
                    date=f_date.isoformat(),
                    is_historical=False,
                    actual_demand=None,
                    p10_demand=round(p10_val, 1),
                    p50_demand=round(p50_val, 1),
                    p90_demand=round(p90_val, 1),
                    promo_active=bool(promo_active > 0),
                    anomaly_flag=False,
                )
            )

        return DemandForecastResponse(
            sku_id=sku_id,
            sku_name=sku_name,
            category=category,
            hub_id=hub_id,
            hub_name=hub_name,
            horizon_days=horizon_days,
            metrics=self.metrics,
            feature_importances=self.feature_importances,
            series=series_points,
        )

    def get_sku_hub_demand_stats(
        self, demand_multiplier: float = 1.0
    ) -> List[Dict[str, float | str]]:
        """
        Return forecasted daily mean and standard deviation of demand across
        all SKU x Hub pairs to feed into the Stochastic Inventory Optimizer.
        """
        if self._raw_history is None:
            raise RuntimeError("Forecaster history not available.")

        grouped = (
            self._raw_history.groupby(["sku_id", "sku_name", "category", "hub_id", "hub_name"])[
                "demand_units"
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        results: List[Dict[str, float | str]] = []
        for _, row in grouped.iterrows():
            results.append(
                {
                    "sku_id": str(row["sku_id"]),
                    "sku_name": str(row["sku_name"]),
                    "category": str(row["category"]),
                    "hub_id": str(row["hub_id"]),
                    "hub_name": str(row["hub_name"]),
                    "daily_demand_mean": round(float(row["mean"]) * demand_multiplier, 2),
                    "daily_demand_std": round(float(row["std"]) * demand_multiplier, 2),
                }
            )
        return results
