# LogiRoute AI — Reproducible ML & Operations Research Benchmark Report

Generated automatically via `python scripts/generate_benchmark_report.py` (Seed = `42`).

## 1. Machine Learning Model Evaluation

| Model / Engine | Primary Metric | Value | Secondary Metric | Value |
| :--- | :--- | :--- | :--- | :--- |
| **Probabilistic Demand Forecaster (CQR GBDT)** | Holdout WMAPE | **7.86%** | Holdout $R^2$ | **0.9579** |
| **Conformal Interval Calibration ($[P_{10}, P_{90}]$)** | Empirical Coverage | **81.78%** | CQR Offset $\delta$ | **4.35 units** |
| **Traffic & Weather-Aware ETA Regressor** | Holdout MAE | **2.56 min** | Holdout $R^2$ | **0.9940** |
| **SLA Delay Risk Classifier** | Holdout ROC-AUC | **0.9908** | Brier Score | **0.0455** |
| **Multivariate Telemetry Anomaly Detector** | ROC-AUC | **0.9999** | PR-AUC / F1 | **0.9987 / 0.9610** |

---

## 2. Combinatorial Route Optimization (`CVRPTW`) Benchmark

| Metric | Unconsolidated Baseline | LogiRoute CVRPTW (CW + 2-Opt + Relocate) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Daily Distance** | 612.39 km | **342.63 km** | **-44.0% (-269.75 km)** |
| **Total Daily Cost** | $1,549.59 | **$706.55** | **-54.4% (-$843.04)** |
| **Carbon Footprint** | 477.66 kg $\text{CO}_2\text{e}$ | **226.63 kg $\text{CO}_2\text{e}$** | **-251.03 kg $\text{CO}_2\text{e}$** |
| **Fleet Utilization** | 54.2% (8 trucks) | **81.6% (4 vehicles)** | **+27.4% utilization** |

---

## 3. Multi-Objective Pareto Frontier Solutions

| Policy ID | Strategy Name | Daily Cost | Daily $\text{CO}_2\text{e}$ | EV Share | Expected SLA | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `POL-EV-GREEN` | Zero-Emission EV-First Priority | $678.29 | 99.7 kg | 83% | 95.8% | **Pareto-Optimal** |
| `POL-PARETO-BAL` | LogiRoute Balanced (CW + 2-Opt + Relocate) | $706.55 | 226.6 kg | 50% | 97.6% | **Pareto-Optimal** |
| `POL-HYPER-CONSOL` | Max-Consolidation Heavy Linehaul | $628.83 | 281.0 kg | 0% | 92.4% | **Pareto-Optimal** |
| `POL-SLA-EXPRESS` | Ultra-Responsive SLA Express Wave | $861.99 | 206.2 kg | 60% | 99.6% | **Pareto-Optimal** |
| `POL-CW-GREEDY` | Clarke-Wright Construction (No 2-Opt) | $805.47 | 267.4 kg | 25% | 93.5% | **Dominated** |
| `POL-LEGACY-BASE` | Unoptimized Legacy Dispatch Baseline | $1,549.59 | 477.7 kg | 0% | 86.5% | **Dominated** |
