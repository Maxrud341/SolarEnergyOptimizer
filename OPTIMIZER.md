# Optimizer documentation

## Purpose

This optimizer solves a rolling-horizon energy management problem for a microgrid with:

- rooftop PV
- battery storage
- grid import/export
- hourly market price

The objective is to decide when to:

- buy electricity from the grid
- sell electricity to the grid
- charge the battery
- discharge the battery

while minimizing cost and respecting physical constraints.

---

## Input data

The script reads the merged hourly dataset from:

- data/combined_market_data.csv

Required fields include:

- datetime
- consumption_kwh
- price_czk_per_kwh
- production_kwh
- pv_forecast_kwh
- temperature_c
- cloud_cover_pct
- humidity_pct
- wind_speed_ms
- battery_kwh

### Important semantic split

- production_kwh = actual / realized PV production signal
- pv_forecast_kwh = weather-based forecast from external sources used by the model for planning

This distinction is important:

- the model should use the forecast for future decisions
- the baseline rule can use the real production signal as a current-state approximation

---

## Forecast logic

The optimizer does not use fixed constants as the main PV assumption.

Instead it trains a lightweight regression model from historical data using:

- hour of day
- weekday / weekend pattern
- sine/cosine encoding for cyclic time effects
- temperature
- cloud cover
- humidity
- wind speed

The target variables are learned statistically for:

- price_czk_per_kwh
- consumption_kwh
- pv_forecast_kwh

In effect, the optimizer estimates how weather and calendar effects influence energy use and generation.

---

## Optimization model

The optimization is a rolling MPC / MILP problem.

At each time step it solves a horizon of 24 hours and chooses actions for the first step.

Decision variables include:

- buy_kwh
- sell_kwh
- charge_kwh
- discharge_kwh
- soc_kwh

Main constraints:

- energy balance across the hour
- battery SOC bounds
- charge/discharge power limits
- grid import/export limits
- efficiency losses

Objective:

- minimize purchase cost
- subtract revenue from selling
- include battery degradation cost

---

## Files involved

### src/optimizer.py
Main logic for:

- feature preparation
- statistical forecasting
- MILP solving
- backtest loop
- summary metrics

### prepare_data.py
Builds the final merged dataset and standardizes units to kWh/hour.

### data/combined_market_data.csv
Main input table used by optimizer.

### trading_log.csv
Output CSV produced by the backtest with hourly decisions.

---

## How to run

From the project root:

```bash
py src/optimizer.py --input data/combined_market_data.csv --max-steps 20 --output trading_log.csv
```

Useful flags:

```bash
--horizon 24
--max-steps 200
--battery-capacity 10
--soc-min 0.10
--soc-max 0.90
--battery-power-max 0.5
--grid-limit 100
```

---

## Output

The script prints summary metrics:

- total bought
- total sold
- total cost
- total revenue
- degradation cost
- net profit

and compares the MILP strategy against the naive baseline.

Example:

```text
=== MILP MPC Strategy ===
Total bought: 27.509 kWh
Total sold:   2.167 kWh
Total cost:   12.49 CZK
Total revenue:3.11 CZK
Battery deg.: 2.65 CZK
Net profit:   -12.03 CZK
```

---

## Practical interpretation

This is a decision engine for a home energy system:

- when price is low: buy and charge the battery
- when price is high: discharge battery or sell surplus
- when PV forecast is high: avoid buying and use self-generation
- when forecast is low: keep battery ready and reduce dependence on grid

The current version is a usable rolling forecast + optimization prototype, not a fully production-grade forecasting model.

---

## Important note for future improvements

Next useful upgrades:

- separate actual PV and forecast PV in reporting
- use longer historical windows for model learning
- compare forecast RMSE versus realized output
- test a stronger model such as ridge regression or gradient boosting
- add explicit uncertainty bands for the forecast

This project is already structured for that next step.
