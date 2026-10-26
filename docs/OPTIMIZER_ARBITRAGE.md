# Optimizer Arbitrage (B2G Commercial EMS)

## 1. Overview & Purpose

`src/optimizer_arbitrage.py` implements a commercial Battery-to-Grid (**B2G**) Model Predictive Control (MPC) Energy Management System designed specifically for the Czech spot market (OTE):
- **Real Hardware Specification:** Calibrated for **Fox-ESS HV2600 HS10.4** (4 modular units, 10.24 kWh nominal capacity, LiFePO4 chemistry).
- **Configurable Discharge Floor:** Operating SoC range from 20% to 100% (8.192 kWh usable storage), preserving battery longevity per user specification.
- **High Round-Trip Efficiency:** Over 95% round-trip efficiency ($\eta_{ch} = \eta_{dis} = 0.975$).
- **Inelastic Household Load:** Real telemetry from `data/Maly.csv` is preserved strictly without synthetic shifting.
- **Commercial B2G Arbitrage:** Battery can discharge at full continuous power (up to 5.0 kW, 25A recommended current) to export into the grid during high spot price spikes.
- **Dynamic Curtailment (Zero Export):** Real-time clamping of grid feed-in to 0.0 kW whenever effective export prices are negative ($P_{sell} < 0$).
- **Czech Per-Phase Metering (Fázové řízení):** Supports asymmetric 3-phase metering as mandated by Czech grid code.

---

## 2. Core Strategies & Safeguards

### A. Zero Grid Pre-Charge Before Sunny Days
Charging the battery from the grid at night (even during low spot tariffs) before a sunny day is counter-productive:
- It occupies battery capacity with paid electricity (+ grid fees + VAT).
- It leaves no room for free daytime solar generation, directly causing massive solar curtailment (*Omezení přetoků*).

**Implementation:**
When conservative daytime solar surplus is sufficient to fill battery headroom (`is_sunny_day = True`):
$$S_{pv}^{safe} \ge \max\big(1.5, (\mathrm{SoC}_{max} - \mathrm{SoC}_{curr}) \cdot 0.85\big)$$
For all nighttime and early morning hours prior to the daytime solar peak ($t < t_{peak}$):
$$P_{ch}(t) \le P_{pv}^{pred}(t)$$
Grid electricity is physically forbidden from entering the battery (`charge[t] <= pv_pred[t]`). The battery enters the morning with maximum available capacity ready to soak up solar energy.

### B. Guarded Solar Pre-Flush (Extreme Cases Only)
Rather than aggressively dumping battery energy into the grid at speculative morning prices, the system relies on **natural morning household consumption** (from 05:00 to 09:00) to draw down battery SoC safely and create headroom.

Commercial B2G pre-flush before the solar peak is restricted to **extreme cases only**:
1. **Conservative Surplus Excess:** The forecast solar surplus exceeds battery headroom by at least 3.0 kWh ($E_{curt}^{risk} \ge 3.0\text{ kWh}$).
2. **Negative Spot Window:** Midday hours have negative prices ($P_{sell} < 0$) where curtailment is inevitable.
3. **Profitable Morning Export:** Morning export spot price is attractive ($P_{sell} \ge 2.00\text{ CZK/kWh}$).

If these conditions are not met, pre-peak export is limited strictly to instantaneous PV:
$$P_{sell}(t) \le P_{pv}^{pred}(t) \quad (\forall t < t_{peak})$$

### C. Forecast Uncertainty & Safety Margins
ML weather models and consumption forecasts have inherent error margins. To protect the household from cloudy days or demand spikes, decision logic applies conservative safety haircuts:
- **Solar Discount:** $P_{pv}^{safe}(t) = 0.80 \cdot P_{pv}^{pred}(t)$ (20% safety discount, `pv_safe = 0.80 * pv_pred`).
- **Load Buffer:** $P_{load}^{safe}(t) = 1.15 \cdot P_{load}^{pred}(t)$ (15% safety buffer, `load_safe = 1.15 * load_pred`).
- **Conservative Surplus:**
  $$S_{pv}^{safe} = \sum_{t=0}^{T-1} \max\big(0.0, P_{pv}^{safe}(t) - P_{load}^{safe}(t)\big)$$

### D. Mathematical Energy Balance with Curtailment Variable
In previous versions, sunny days with negative export prices caused the LP equality balance to become mathematically impossible when the battery was full, causing CBC solver infeasibility fallbacks.

The LP problem now includes a physical curtailment variable $P_{curt}(t) \in [0, P_{pv}^{pred}(t)]$:
$$P_{buy}(t) + P_{pv}^{pred}(t) + P_{dis}(t) = P_{load}^{pred}(t) + P_{sell}(t) + P_{ch}(t) + P_{curt}(t)$$
Objective penalty: $+0.005 \cdot P_{curt}(t)$.
The solver prioritizes self-consumption, battery charging, and commercial export, but legally and smoothly curtails solar when export is prohibited and the battery is saturated.

---

## 3. Running the Optimizer

### Full Backtest
```powershell
python -m src.optimizer_arbitrage --input data/combined_market_data.csv --output trading_log_arbitrage.csv
```

### Specific Date or Month
```powershell
python -m src.optimizer_arbitrage --month 2025-06
python -m src.optimizer_arbitrage --start-date 2025-06-15 --end-date 2025-06-20
```

### Interactive Web Dashboard
```powershell
python visualizer_web.py
# or
streamlit run visualizer_web.py
```

---

## 4. Scientific Foundations & Academic Literature Benchmarking

The algorithms implemented in `SolarEnergyOptimizer` directly correspond to the methodologies established in top-tier peer-reviewed energy journals (*Applied Energy*, *IEEE Transactions on Smart Grid*, *Journal of Energy Storage*).

### A. Academic Research Alignment

1. **Battery Degradation & Arbitrage Economics:**
   - *Reference:* Xu, B., Oudalov, A., Ulbig, A., Andersson, G., & Kirschen, D. S. (2018). *Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment.* **IEEE Transactions on Smart Grid**, 9(2), 1131–1140.
   - *Key Finding:* Naive energy arbitrage easily destroys battery capital value through micro-cycling and deep discharges. Incorporating an explicit degradation cost $C_{deg}$ into the objective function establishes a necessary profitability barrier (Hurdle Rate):
     $$\text{Spread} \ge \frac{\text{WACSE}}{\eta_{dis}} + C_{deg}$$
   - *Implementation:* Our model enforces this via the dynamic `WACSE` (Weighted Average Cost of Stored Energy) hurdle rate, forbidding export whenever the spot spread fails to cover lifecycle battery wear.

2. **Forecast-Based Curtailment Prevention:**
   - *Reference:* Zeh, A., & Witzmann, R. (2014). *Operational strategies for battery energy storage systems in low-voltage distribution grids to integrate high shares of PV.* **Applied Energy**, 136, 122–129.
   - *Key Finding:* Uncontrolled early-morning charging fills residential storage by 10:00–11:00 AM, forcing solar inverters into heavy curtailment during peak midday irradiance.
   - *Implementation:* Realized via our **Zero Grid Pre-Charge** safeguard ($S_{pv}^{safe} \ge \text{Headroom} \times 0.85$), ensuring the battery enters sunny mornings with sufficient headroom to soak up free midday generation.

3. **Smart Control & Lifetime Extension:**
   - *Reference:* Angenendt, P., Zurmühlen, S., Axelsen, H., & Sauer, D. U. (2018). *Forecasting and smart control of PV battery storage systems to provide grid relief and improve battery lifetime.* **Applied Energy**, 218, 523–537.
   - *Key Finding:* Lowering the average battery state of charge (SoC) prior to peak generation reduces both cyclic and calendar aging, cutting degradation costs in half while providing grid relief.
   - *Implementation:* Reflected in our guarded morning pre-flush logic, avoiding unnecessary full-charge resting periods at night.

4. **Dynamic Spot Pricing & Mixed-Integer Formulations:**
   - *Reference:* Khalilpour, S., & Vassallo, R. (2016). *Planning and operation schedule of PV-battery system under dynamic pricing.* **Applied Energy**, 164, 359–370.
   - *Key Finding:* Formulating storage dispatch as a rolling-horizon optimization problem under time-varying market tariffs enables coordinated self-consumption and commercial export.
   - *Implementation:* Realized via our 24-hour rolling horizon LP/MPC framework using the COIN-OR CBC solver.

### B. Methodological Comparison

| Problem / Mechanism | Academic Literature Solution | SolarEnergyOptimizer Implementation |
|---|---|---|
| **Horizon Edge Effect** | Terminal SoC Valuation ($\max \dots + V_{term} \cdot SoC(H)$) | Avoided-purchase terminal valuation ($V_{term} \approx \bar{P}_{buy} \cdot \eta_{dis} - C_{deg}$) preventing midnight dumping. |
| **Negative Spot Prices** | Curtailment variable ($P_{curt}(t)$) in LP power balance | Physical slack variable $P_{curt}(t) \in [0, P_{pv}^{pred}(t)]$ with zero-export enforcement when $P_{sell} < 0$. |
| **Forecast Uncertainty** | Robust MPC / Confidence bands | Conservative safety haircuts: $P_{pv}^{safe} = 0.80 \cdot P_{pv}^{pred}$ and $P_{load}^{safe} = 1.15 \cdot P_{load}^{pred}$. |
| **Grid Metering Physics** | Usually idealized net-metering (single-phase assumption) | Realistic Czech 3-phase asymmetric metering (*Fázové řízení*), evaluating L1, L2, L3 individual imbalances. |
| **Battery Specifications** | Generic idealized 10 kWh blocks | Real **Fox-ESS HV2600 HS10.4** hardware specs: 10.24 kWh nominal, 20% SoC floor, >95% round-trip efficiency, 5.0 kW power. |
