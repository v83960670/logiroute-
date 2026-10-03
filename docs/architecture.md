# LogiRoute AI — Technical Architecture & Mathematical Formulations

This document details the mathematical formulations, feature engineering pipelines, and combinatorial optimization algorithms powering **LogiRoute AI**.

---

## 1. Conformalized Quantile Demand Forecasting

Supply chain inventory decisions depend not just on point forecasts ($\mathbb{E}[Y|X]$), but on the upper tail of the demand distribution during supplier lead times.

### Pinball / Quantile Loss Training
For target quantiles $\tau \in \{0.10, 0.50, 0.90\}$, we train separate Gradient Boosted Decision Trees (`HistGradientBoostingRegressor`) minimizing the pinball loss:

$$\mathcal{L}_{\tau}(y, \hat{q}_{\tau}) = \max\big(\tau (y - \hat{q}_{\tau}),\; (\tau - 1)(y - \hat{q}_{\tau})\big)$$

### Split Conformalized Quantile Regression (CQR)
Raw quantile tree predictions can exhibit slight under-coverage on out-of-sample data. Following Romano et al. (*Conformalized Quantile Regression*, NeurIPS 2019), we partition a chronological calibration window $\mathcal{D}_{\text{cal}}$ of size $n$ prior to the validation holdout set and compute conformity scores:

$$E_i = \max\Big(\hat{q}_{0.10}(x_i) - y_i,\; y_i - \hat{q}_{0.90}(x_i)\Big), \quad \forall i \in \mathcal{D}_{\text{cal}}$$

For target miscoverage rate $\alpha = 0.20$ (an 80% interval), the conformal offset $\delta_{\text{CQR}}$ is the empirical quantile:

$$\delta_{\text{CQR}} = \text{Quantile}\left(\{E_i\}_{i=1}^n,\; \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)$$

The calibrated prediction interval at test point $x_{\text{test}}$ is then:

$$\hat{C}(x_{\text{test}}) = \big[\hat{q}_{0.10}(x_{\text{test}}) - \delta_{\text{CQR}},\; \hat{q}_{0.90}(x_{\text{test}}) + \delta_{\text{CQR}}\big]$$

with monotone rearrangement applied to guarantee $\hat{y}_{P10} \le \hat{y}_{P50} \le \hat{y}_{P90}$.

---

## 2. Traffic & Weather-Aware ETA & SLA Delay Risk

Given shipment covariates $x = (d_{\text{km}}, t_{\text{dep}}, \sin(2\pi t_{\text{dep}}/24), \cos(2\pi t_{\text{dep}}/24), \text{DOW}, \text{Traffic}, \text{Weather}, W_{\text{kg}}, T_{\text{free}})$:

1. **Expected Transit Time**: $\hat{T}_{\text{mean}}(x) = f_{\text{MSE}}(x)$
2. **Tail Transit Time**: $\hat{T}_{P90}(x) = f_{\tau=0.90}(x)$
3. **Hybrid SLA Breach Probability**: Combines the gradient boosted classifier probability $\hat{p}_{\text{GBDT}}(x)$ with the structural margin logistic score relative to the promised SLA $T_{\text{SLA}}$:

$$\hat{p}_{\text{SLA}}(x) = 0.45\,\hat{p}_{\text{GBDT}}(x) + 0.55\,\sigma\left(\gamma \cdot \frac{\hat{T}_{\text{mean}}(x) - T_{\text{SLA}}}{T_{\text{SLA}}}\right)$$

4. **Counterfactual Factor Attribution**: Evaluates marginal travel-time reductions when setting each friction covariate to baseline nominal levels ($\text{Traffic} \to 0.05$, $\text{Weather} \to 0.02$, $W_{\text{kg}} \to 400\text{ kg}$).

---

## 3. Capacitated Vehicle Routing with Time Windows (`CVRPTW`)

Given a central depot $v_0$ and customer nodes $V_c = \{v_1, \dots, v_N\}$, each with payload demand $q_i$, service duration $s_i$, and time window $[e_i, l_i]$:

1. **Road Network Metric**:
   $$d_{\text{road}}(i, j) = \tau_{\text{road}} \cdot 2 R_{\oplus} \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos\phi_i \cos\phi_j \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
   where $\tau_{\text{road}} = 1.28$ accounts for urban/peri-urban road winding.
2. **Clarke-Wright Parallel Savings Construction**:
   Pairs $(i, j)$ are ranked by time-window penalized savings:
   $$s_{ij} = d(0, i) + d(0, j) - d(i, j) - \lambda_{\text{TW}} |e_i - e_j|$$
   Routes are merged greedily when $i$ and $j$ are exterior endpoints of distinct routes and $\sum_{u \in R_a \cup R_b} q_u \le Q_{\max}$.
3. **Intra-Route 2-Opt Local Search**:
   For each route $R = (0, u_1, \dots, u_k, 0)$, sub-segments $R[a:b]$ are inverted whenever the resulting tour strictly reduces closed-loop distance.
4. **Inter-Route Relocate**:
   Evaluates relocating single nodes $u \in R_1$ into optimal insertion positions in $R_2$ subject to capacity feasibility and net distance reduction $\Delta d < -0.25\text{ km}$.
5. **Heterogeneous Fleet Assignment**:
   Each finalized route $R$ is assigned to the minimum-capacity vehicle class $k \in \{\text{EV-4.5T}, \text{MED-9T}, \text{HVY-16T}\}$ satisfying $\sum_{u \in R} q_u \le Q_k$, minimizing both dispatch cost and $\text{kg CO}_2\text{e}$.

---

## 4. Multi-Echelon Stochastic Inventory Control (`(s, S)` Policy)

For each SKU $\times$ Hub node with daily demand $\mathcal{N}(\mu_D, \sigma_D^2)$ and replenishment lead time $\mathcal{N}(\mu_L, \sigma_L^2)$:

- **Lead-Time Demand Mean & Standard Deviation**:
  $$\mu_{DL} = \mu_D \mu_L, \qquad \sigma_{DL} = \sqrt{\mu_L \sigma_D^2 + \mu_D^2 \sigma_L^2}$$
- **Safety Stock & Reorder Point ($s = \text{ROP}$)**:
  $$SS = \left\lceil \Phi^{-1}(\alpha) \cdot \sigma_{DL} \right\rceil, \qquad s = \left\lceil \mu_{DL} + SS \right\rceil$$
- **Economic Order Quantity & Order-Up-To Level ($S$)**:
  $$EOQ = \left\lceil \sqrt{\frac{2 \cdot (365 \mu_D) \cdot K}{h \cdot C}} \right\rceil, \qquad S = s + EOQ$$
