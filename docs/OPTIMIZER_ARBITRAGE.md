# Optimizer Arbitrage (B2G Commercial EMS)

## 1. Overview & Purpose

`src/optimizer_arbitrage.py` implements a commercial Battery-to-Grid (**B2G**) Model Predictive Control (MPC) Energy Management System designed specifically for the Czech spot market (OTE):
- **Full Usable Battery Range:** 5% to 100% SoC (9.5 kWh usable on a 10 kWh battery).
- **Inelastic Household Load:** Real telemetry from `data/Maly.csv` is preserved strictly without synthetic shifting.
- **Commercial B2G Arbitrage:** Battery can discharge at full inverter continuous power (up to 3.6 kW) to export into the grid during high spot price spikes.
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
