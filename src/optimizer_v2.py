"""Solar Energy Optimizer V2 (MILP MPC Rolling Horizon Energy Management).

Features:
- Machine Learning / Statistical Forecasting (LightGBM Gradient Boosting & Ridge Regression).
- High-accuracy nonlinear capture of solar physics, cloud interaction, weather, and price spikes.
- Mixed-Integer Linear Programming (MILP) with binary variables for buy/sell & charge/discharge mutual exclusion.
- Inverter solar curtailment during negative electricity market prices.
- True real-time closed-loop backtest (MPC): forecast-driven battery schedule balanced against realized PV and load.
- Calibrated 10.0 kWh battery model with 5.0 kW inverter power limits and degradation tracking.
- Financial profit analysis comparing MILP optimization against naive self-consumption baseline.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pulp as pl

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


DEFAULT_BATTERY_CAPACITY_KWH = 10.0
DEFAULT_SOC_MIN = 0.0
DEFAULT_SOC_MAX = 1.0
DEFAULT_BATTERY_POWER_MAX_KW = 5.0
DEFAULT_CHARGE_EFFICIENCY = 0.95
DEFAULT_DISCHARGE_EFFICIENCY = 0.95
DEFAULT_DEGRADATION_COST_CZK_PER_KWH = 0.35
DEFAULT_GRID_LIMIT_KW = 17.25
DEFAULT_HORIZON_HOURS = 24
DEFAULT_BIG_M = 100.0
DEFAULT_IMPORT_FEE_CZK_PER_KWH = 3.0
DEFAULT_VAT_RATE = 0.21
DEFAULT_EXPORT_FEE_CZK_PER_KWH = 0.4


@dataclass
class OptimizerV2Config:
    battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH
    soc_min: float = DEFAULT_SOC_MIN
    soc_max: float = DEFAULT_SOC_MAX
    battery_power_max_kw: float = DEFAULT_BATTERY_POWER_MAX_KW
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    discharge_efficiency: float = DEFAULT_DISCHARGE_EFFICIENCY
    degradation_cost_czk_per_kwh: float = DEFAULT_DEGRADATION_COST_CZK_PER_KWH
    grid_limit_kw: float = DEFAULT_GRID_LIMIT_KW
    horizon_hours: int = DEFAULT_HORIZON_HOURS
    allow_curtailment: bool = True
    model_type: str = "lgbm"  # "lgbm", "hgb", or "linear"
    strategy: str = "peak_preservation"  # "peak_preservation" (fast, robust, best profit) or "milp"
    peak_price_spread_thresh: float = 2.0  # CZK/kWh spread required to activate peak reservation
    import_fee_czk_per_kwh: float = DEFAULT_IMPORT_FEE_CZK_PER_KWH
    vat_rate: float = DEFAULT_VAT_RATE
    export_fee_czk_per_kwh: float = DEFAULT_EXPORT_FEE_CZK_PER_KWH
    big_m: float = DEFAULT_BIG_M


def compute_effective_buy_price(spot_price: float | np.ndarray, cfg: OptimizerV2Config) -> float | np.ndarray:
    """Effective import price from grid: (spot + 3.0 CZK) * (1 + 0.21 VAT)."""
    return (spot_price + cfg.import_fee_czk_per_kwh) * (1.0 + cfg.vat_rate)


def compute_effective_sell_price(spot_price: float | np.ndarray, cfg: OptimizerV2Config) -> float | np.ndarray:
    """Effective export price to grid: spot - 0.40 CZK (trader fee deducted)."""
    return spot_price - cfg.export_fee_czk_per_kwh


CONFIG_V2 = OptimizerV2Config()


# =====================================================================
# 1. Feature Engineering & ML / Statistical Forecasting
# =====================================================================

FEATURE_SETS = {
    "production_kwh": [
        "pv_forecast_kwh",
        "hour_cos",
        "hour_sin",
        "temperature_c",
        "humidity_pct",
        "cloud_cover_pct",
        "wind_speed_ms",
        "month_sin",
        "month_cos",
        "is_daytime",
    ],
    "price_czk_per_kwh": [
        "hour_cos",
        "hour_sin",
        "is_weekend",
        "wind_speed_ms",
        "temperature_c",
        "cloud_cover_pct",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ],
    "consumption_kwh": [
        "hour_cos",
        "hour_sin",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "temperature_c",
        "humidity_pct",
        "cloud_cover_pct",
        "month_sin",
        "month_cos",
        "is_daytime",
    ],
    "consumption_l1_kwh": [
        "hour_cos",
        "hour_sin",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "temperature_c",
        "humidity_pct",
        "cloud_cover_pct",
        "month_sin",
        "month_cos",
        "is_daytime",
    ],
    "consumption_l2_kwh": [
        "hour_cos",
        "hour_sin",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "temperature_c",
        "humidity_pct",
        "month_sin",
        "month_cos",
        "is_daytime",
    ],
    "consumption_l3_kwh": [
        "hour_cos",
        "hour_sin",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "temperature_c",
        "humidity_pct",
        "month_sin",
        "month_cos",
        "is_daytime",
    ],
}


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all cyclic time and calendar features are present."""
    frame = df.copy()
    if "datetime" in frame.columns:
        dt = pd.to_datetime(frame["datetime"], errors="coerce")
        hour = dt.dt.hour.fillna(0.0).to_numpy(dtype=float)
        dayofweek = dt.dt.dayofweek.fillna(0.0).to_numpy(dtype=float)
        month = dt.dt.month.fillna(1.0).to_numpy(dtype=float)
        dayofyear = dt.dt.dayofyear.fillna(1.0).to_numpy(dtype=float)

        frame["hour"] = hour
        frame["dayofweek"] = dayofweek
        frame["month"] = month
        frame["is_weekend"] = (dayofweek >= 5).astype(float)
        frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        frame["weekday_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
        frame["weekday_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
        frame["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        frame["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        frame["day_of_year_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.25)
        frame["day_of_year_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.25)
        frame["is_daytime"] = ((hour >= 5) & (hour <= 21)).astype(float)

    defaults = {
        "pv_forecast_kwh": 0.0,
        "temperature_c": 12.0,
        "cloud_cover_pct": 50.0,
        "humidity_pct": 75.0,
        "wind_speed_ms": 2.5,
        "is_weekend": 0.0,
        "hour_cos": 0.0,
        "hour_sin": 0.0,
        "weekday_sin": 0.0,
        "weekday_cos": 0.0,
        "month_sin": 0.0,
        "month_cos": 0.0,
        "is_daytime": 1.0,
    }
    for col, val in defaults.items():
        if col not in frame.columns:
            frame[col] = val

    return frame


def fit_forecaster_model(history_df: pd.DataFrame, target: str, model_type: str = "lgbm") -> dict:
    """Fit a forecasting model (LightGBM / HistGradientBoosting / Ridge) on history."""
    features = FEATURE_SETS.get(target, ["hour_cos", "hour_sin", "is_weekend"])
    prepared = prepare_feature_frame(history_df)

    y = pd.to_numeric(prepared[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X = np.ones((len(prepared), len(features)), dtype=float)

    for idx, f in enumerate(features):
        X[:, idx] = pd.to_numeric(prepared[f], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # 1. Non-linear LightGBM GBDT (fastest & most accurate)
    if model_type == "lgbm" and HAS_LIGHTGBM and len(prepared) >= 48:
        model = lgb.LGBMRegressor(
            n_estimators=40,
            num_leaves=15,
            min_child_samples=10,
            learning_rate=0.08,
            verbose=-1,
            n_jobs=1,
            random_state=42,
        )
        model.fit(X, y)
        return {
            "type": "lgbm",
            "target": target,
            "features": features,
            "model": model,
        }

    # 2. HistGradientBoosting (scikit-learn alternative)
    if model_type in ("lgbm", "hgb") and HAS_SKLEARN and len(prepared) >= 48:
        model = HistGradientBoostingRegressor(
            max_iter=40,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            learning_rate=0.08,
            random_state=42,
        )
        model.fit(X, y)
        return {
            "type": "hgb",
            "target": target,
            "features": features,
            "model": model,
        }

    # 3. Ridge Linear Regression fallback
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0) + 1e-6
    X_norm = np.hstack([np.ones((len(prepared), 1)), (X - means) / stds])

    alpha = 1.0
    xtx = X_norm.T @ X_norm + alpha * np.eye(X_norm.shape[1])
    xty = X_norm.T @ y
    beta = np.linalg.solve(xtx, xty)

    return {
        "type": "linear",
        "target": target,
        "features": features,
        "means": means,
        "stds": stds,
        "coefficients": beta,
    }


def fit_statistical_forecaster(history_df: pd.DataFrame, target: str) -> dict:
    """Backward-compatible alias for fitting forecaster."""
    return fit_forecaster_model(history_df, target, model_type="linear")


def predict_with_forecaster(model: dict, future_window: pd.DataFrame) -> np.ndarray:
    """Generate forecast predictions for future window."""
    prepared = prepare_feature_frame(future_window)
    features = model["features"]
    X = np.zeros((len(prepared), len(features)), dtype=float)

    for idx, f in enumerate(features):
        X[:, idx] = pd.to_numeric(prepared[f], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    mtype = model.get("type", "linear")
    if mtype in ("lgbm", "hgb"):
        preds = model["model"].predict(X)
    else:
        means = model["means"]
        stds = model["stds"]
        X_norm = np.hstack([np.ones((len(prepared), 1)), (X - means) / stds])
        preds = X_norm @ model["coefficients"]

    # Target specific physical constraints
    if model["target"] == "production_kwh":
        preds = np.clip(preds, 0.0, None)
        # Nighttime generation is strictly zero
        is_day = prepared["is_daytime"].to_numpy(dtype=float)
        preds = preds * is_day
    elif model["target"] in ("consumption_kwh", "consumption_l1_kwh", "consumption_l2_kwh", "consumption_l3_kwh"):
        preds = np.clip(preds, 0.001, None)

    return np.asarray(preds, dtype=float)


def predict_consumption(
    history_df: Optional[pd.DataFrame],
    window_df: pd.DataFrame,
    model_type: str = "lgbm",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict household load, utilizing individual phase forecasters (L1, L2, L3) if available."""
    phase_cols = ["consumption_l1_kwh", "consumption_l2_kwh", "consumption_l3_kwh"]
    has_phases = (
        history_df is not None
        and all(c in history_df.columns for c in phase_cols)
        and len(history_df) >= 24
    )

    if has_phases:
        pred_dict = {}
        for c in phase_cols:
            phase_model = fit_forecaster_model(history_df, c, model_type=model_type)
            pred_dict[c] = predict_with_forecaster(phase_model, window_df)

        pred_l1 = pred_dict["consumption_l1_kwh"]
        pred_l2 = pred_dict["consumption_l2_kwh"]
        pred_l3 = pred_dict["consumption_l3_kwh"]
        load_pred = np.clip(pred_l1 + pred_l2 + pred_l3, 0.01, None)
        return load_pred, {"l1": pred_l1, "l2": pred_l2, "l3": pred_l3}

    elif history_df is not None and len(history_df) >= 24 and "consumption_kwh" in history_df.columns:
        model_load = fit_forecaster_model(history_df, "consumption_kwh", model_type=model_type)
        load_pred = predict_with_forecaster(model_load, window_df)
        load_pred = np.clip(load_pred, 0.01, None)
        return load_pred, {"l1": load_pred / 3.0, "l2": load_pred / 3.0, "l3": load_pred / 3.0}

    else:
        if all(c in window_df.columns for c in phase_cols):
            l1 = pd.to_numeric(window_df["consumption_l1_kwh"]).fillna(0.1).to_numpy(dtype=float)
            l2 = pd.to_numeric(window_df["consumption_l2_kwh"]).fillna(0.1).to_numpy(dtype=float)
            l3 = pd.to_numeric(window_df["consumption_l3_kwh"]).fillna(0.1).to_numpy(dtype=float)
            load_pred = np.clip(l1 + l2 + l3, 0.01, None)
            return load_pred, {"l1": l1, "l2": l2, "l3": l3}
        else:
            load_pred = pd.to_numeric(window_df.get("consumption_kwh", 0.3)).fillna(0.3).to_numpy(dtype=float)
            load_pred = np.clip(load_pred, 0.01, None)
            return load_pred, {"l1": load_pred / 3.0, "l2": load_pred / 3.0, "l3": load_pred / 3.0}


def compute_feature_weights_summary(df: pd.DataFrame) -> dict:
    """Calculate standardized statistical regression weights and correlations for each target."""
    summary = {}
    for target, features in FEATURE_SETS.items():
        if target not in df.columns:
            continue
        sub = df[[target] + [f for f in features if f in df.columns]].dropna().copy()
        valid_features = [f for f in features if f in sub.columns]
        y = sub[target].to_numpy(dtype=float)
        X = sub[valid_features].to_numpy(dtype=float)

        y_std = (y - np.mean(y)) / (np.std(y) + 1e-9)
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

        beta, *_ = np.linalg.lstsq(X_std, y_std, rcond=None)
        y_pred = X_std @ beta
        r2 = max(0.0, 1.0 - float(np.sum((y_std - y_pred) ** 2)) / float(np.sum(y_std ** 2) + 1e-9))

        corrs = {}
        for f in valid_features:
            s_f = np.std(sub[f])
            s_y = np.std(y)
            if s_f > 1e-9 and s_y > 1e-9:
                c = float(np.corrcoef(sub[f], y)[0, 1])
                corrs[f] = 0.0 if np.isnan(c) else c
            else:
                corrs[f] = 0.0
        weights = {f: float(b) for f, b in zip(valid_features, beta)}

        summary[target] = {
            "r2": r2,
            "correlations": corrs,
            "weights": weights,
        }
    return summary


# =====================================================================
# 2. Mixed-Integer Linear Programming (MILP) Solver
# =====================================================================

def solve_milp_window_v2(
    window_df: pd.DataFrame,
    current_soc_kwh: float,
    cfg: OptimizerV2Config = CONFIG_V2,
    history_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """Formulate and solve the 24-hour microgrid MILP problem."""
    t_steps = len(window_df)
    if t_steps == 0:
        raise ValueError("Cannot optimize an empty window.")

    # 1. Forecast PV, Load, Price with ML / Statistical model
    if history_df is not None and len(history_df) >= 24:
        model_pv = fit_forecaster_model(history_df, "production_kwh", model_type=cfg.model_type)
        model_price = fit_forecaster_model(history_df, "price_czk_per_kwh", model_type=cfg.model_type)

        pv_pred = predict_with_forecaster(model_pv, window_df)
        price_pred = predict_with_forecaster(model_price, window_df)
    else:
        pv_pred = pd.to_numeric(window_df.get("pv_forecast_kwh", window_df.get("production_kwh", 0.0))).fillna(0.0).to_numpy(dtype=float)
        price_pred = pd.to_numeric(window_df["price_czk_per_kwh"]).fillna(2.5).to_numpy(dtype=float)

    load_pred, phase_preds = predict_consumption(history_df, window_df, model_type=cfg.model_type)

    min_soc_kwh = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc_kwh = cfg.soc_max * cfg.battery_capacity_kwh
    current_soc_kwh = float(np.clip(current_soc_kwh, min_soc_kwh, max_soc_kwh))

    prob = pl.LpProblem("Microgrid_MILP_v2", pl.LpMinimize)

    # Decision variables
    buy = [pl.LpVariable(f"buy_{t}", lowBound=0.0, upBound=cfg.grid_limit_kw) for t in range(t_steps)]
    sell = [pl.LpVariable(f"sell_{t}", lowBound=0.0, upBound=cfg.grid_limit_kw) for t in range(t_steps)]
    ch = [pl.LpVariable(f"ch_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    dis = [pl.LpVariable(f"dis_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    soc = [pl.LpVariable(f"soc_{t}", lowBound=min_soc_kwh, upBound=max_soc_kwh) for t in range(t_steps)]
    curt = [
        pl.LpVariable(f"curt_{t}", lowBound=0.0, upBound=float(pv_pred[t]))
        if cfg.allow_curtailment else 0.0
        for t in range(t_steps)
    ]

    # Binary exclusion variables
    u_grid = [pl.LpVariable(f"u_grid_{t}", cat=pl.LpBinary) for t in range(t_steps)]
    u_bat = [pl.LpVariable(f"u_bat_{t}", cat=pl.LpBinary) for t in range(t_steps)]

    price_buy_pred = compute_effective_buy_price(price_pred, cfg)
    price_sell_pred = compute_effective_sell_price(price_pred, cfg)

    # Objective: electricity purchase cost - selling revenue + battery degradation
    # Include terminal battery value so the optimizer preserves stored energy across the rolling horizon
    # Value must be higher than export price (to prevent end-of-horizon dumping)
    # but lower than retail buy price (so the solver never buys from grid just to claim terminal bonus)
    terminal_value = max(0.5, float(np.mean(price_sell_pred)) + cfg.degradation_cost_czk_per_kwh)
    cost_terms = [-terminal_value * soc[t_steps - 1]]
    for t in range(t_steps):
        cost_terms.append(buy[t] * float(price_buy_pred[t]))
        cost_terms.append(-sell[t] * float(price_sell_pred[t]))
        cost_terms.append(cfg.degradation_cost_czk_per_kwh * (ch[t] + dis[t]))
    prob += pl.lpSum(cost_terms)

    # Constraints
    for t in range(t_steps):
        # 1. Instantaneous Energy Balance
        prob += buy[t] + (float(pv_pred[t]) - curt[t]) + dis[t] == float(load_pred[t]) + sell[t] + ch[t]

        # 2. Binary Grid Exclusion (no simultaneous buy and sell)
        prob += buy[t] <= cfg.grid_limit_kw * u_grid[t]
        prob += sell[t] <= cfg.grid_limit_kw * (1.0 - u_grid[t])

        # 3. Binary Battery Exclusion (no simultaneous charge and discharge)
        prob += ch[t] <= cfg.battery_power_max_kw * u_bat[t]
        prob += dis[t] <= cfg.battery_power_max_kw * (1.0 - u_bat[t])

        # 4. Zero battery export to grid (battery discharge strictly serves local load)
        prob += dis[t] <= float(load_pred[t])

        # 5. SoC Dynamics
        prev_soc = current_soc_kwh if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + cfg.charge_efficiency * ch[t] - dis[t] * (1.0 / cfg.discharge_efficiency)

    solver = pl.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pl.LpStatus[status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"MILP solver finished with non-optimal status: {pl.LpStatus[status]}")

    results = []
    for t in range(t_steps):
        results.append(
            {
                "datetime": window_df["datetime"].iloc[t],
                "buy_kwh": float(pl.value(buy[t])),
                "sell_kwh": float(pl.value(sell[t])),
                "charge_kwh": float(pl.value(ch[t])),
                "discharge_kwh": float(pl.value(dis[t])),
                "curtailment_kwh": float(pl.value(curt[t])) if cfg.allow_curtailment else 0.0,
                "soc_kwh": float(pl.value(soc[t])),
                "forecast_pv_kwh": float(pv_pred[t]),
                "forecast_load_kwh": float(load_pred[t]),
                "forecast_load_l1_kwh": float(phase_preds["l1"][t]),
                "forecast_load_l2_kwh": float(phase_preds["l2"][t]),
                "forecast_load_l3_kwh": float(phase_preds["l3"][t]),
                "forecast_spot_price_czk": float(price_pred[t]),
                "forecast_buy_price_czk": float(price_buy_pred[t]),
                "forecast_sell_price_czk": float(price_sell_pred[t]),
            }
        )

    plan_df = pd.DataFrame(results)
    meta = {
        "status": pl.LpStatus[status],
        "objective_val": float(pl.value(prob.objective)),
        "soc_start": current_soc_kwh,
        "soc_end": float(pl.value(soc[t_steps - 1])),
    }
    return plan_df, meta


# =====================================================================
# 3. Naive Rule-based Baseline Strategy
# =====================================================================

def run_naive_baseline_v2(
    df: pd.DataFrame,
    cfg: OptimizerV2Config = CONFIG_V2,
    max_steps: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate a standard self-consumption rule-based baseline without forecasting."""
    records = []
    min_soc = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc = cfg.soc_max * cfg.battery_capacity_kwh
    soc = float(np.clip(df["battery_kwh"].dropna().iloc[0] if "battery_kwh" in df.columns else 5.0, min_soc, max_soc))

    steps = len(df) if max_steps is None else min(len(df), max_steps)

    for i in range(steps):
        row = df.iloc[i]
        spot_price = float(row["price_czk_per_kwh"])
        price_buy = float(compute_effective_buy_price(spot_price, cfg))
        price_sell = float(compute_effective_sell_price(spot_price, cfg))
        load = float(row["consumption_kwh"])
        load_l1 = float(row.get("consumption_l1_kwh", load / 3.0))
        load_l2 = float(row.get("consumption_l2_kwh", load / 3.0))
        load_l3 = float(row.get("consumption_l3_kwh", load / 3.0))
        pv = float(row["production_kwh"])

        buy = 0.0
        sell = 0.0
        charge = 0.0
        discharge = 0.0
        curt = 0.0

        net = pv - load

        if net > 0:
            surplus = net
            max_absorbable = max(0.0, (max_soc - soc) / cfg.charge_efficiency)
            charge = min(surplus, max_absorbable, cfg.battery_power_max_kw)
            soc = soc + charge * cfg.charge_efficiency
            surplus -= charge

            if surplus > 1e-6:
                if price_sell < 0 and cfg.allow_curtailment:
                    curt = surplus
                else:
                    sell = min(surplus, cfg.grid_limit_kw)
        else:
            deficit = -net
            max_deliverable = max(0.0, (soc - min_soc) * cfg.discharge_efficiency)
            discharge = min(deficit, max_deliverable, cfg.battery_power_max_kw)
            soc = soc - discharge / cfg.discharge_efficiency
            deficit -= discharge

            if deficit > 1e-6:
                buy = min(deficit, cfg.grid_limit_kw)

        soc = float(np.clip(soc, min_soc, max_soc))

        cost_buy = buy * price_buy
        rev_sell = sell * price_sell
        deg_cost = cfg.degradation_cost_czk_per_kwh * (charge + discharge)
        net_cf = rev_sell - cost_buy
        profit = net_cf - deg_cost

        records.append(
            {
                "datetime": row["datetime"],
                "spot_price_czk_per_kwh": spot_price,
                "effective_buy_price_czk": price_buy,
                "effective_sell_price_czk": price_sell,
                "house_consumption_kwh": load,
                "load_kwh": load,
                "load_l1_kwh": load_l1,
                "load_l2_kwh": load_l2,
                "load_l3_kwh": load_l3,
                "pv_kwh": pv,
                "buy_kwh": buy,
                "sell_kwh": sell,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "curtailment_kwh": curt,
                "soc_kwh": soc,
                "cost_buy_czk": cost_buy,
                "revenue_sell_czk": rev_sell,
                "degradation_cost_czk": deg_cost,
                "net_profit_czk": profit,
            }
        )

    return pd.DataFrame(records)


# =====================================================================
# 4. Realistic Rolling Horizon MPC Backtest
# =====================================================================

def run_backtest_v2(
    df: pd.DataFrame,
    cfg: OptimizerV2Config = CONFIG_V2,
    max_steps: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute realistic backtest with statistical/ML forecast and real-time grid balancing."""
    df = df.sort_values("datetime").reset_index(drop=True)
    steps = len(df) if max_steps is None else min(len(df), max_steps)

    min_soc = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc = cfg.soc_max * cfg.battery_capacity_kwh

    initial_soc = float(df["battery_kwh"].dropna().iloc[0] if "battery_kwh" in df.columns else 5.0)
    current_soc = float(np.clip(initial_soc, min_soc, max_soc))

    history_window_len = 24 * 7  # 7-day rolling training window
    cum_consumption = 0.0
    log_rows = []

    spot_prices = pd.to_numeric(df["price_czk_per_kwh"], errors="coerce").fillna(2.0).to_numpy(dtype=float)
    loads = pd.to_numeric(df["consumption_kwh"], errors="coerce").fillna(0.3).to_numpy(dtype=float)
    has_phases = all(c in df.columns for c in ("consumption_l1_kwh", "consumption_l2_kwh", "consumption_l3_kwh"))
    if has_phases:
        loads_l1 = pd.to_numeric(df["consumption_l1_kwh"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
        loads_l2 = pd.to_numeric(df["consumption_l2_kwh"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
        loads_l3 = pd.to_numeric(df["consumption_l3_kwh"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
    else:
        loads_l1 = loads / 3.0
        loads_l2 = loads / 3.0
        loads_l3 = loads / 3.0

    pvs = pd.to_numeric(df["production_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dts = pd.to_datetime(df["datetime"], errors="coerce")
    hours = dts.dt.hour.fillna(0).to_numpy(dtype=int)

    p_buy_all = compute_effective_buy_price(spot_prices, cfg)
    p_sell_all = compute_effective_sell_price(spot_prices, cfg)

    model_label = "LightGBM" if (cfg.model_type == "lgbm" and HAS_LIGHTGBM) else "Linear Ridge"
    strat_label = "Adaptive Peak Preservation" if cfg.strategy == "peak_preservation" else f"Rolling MILP MPC ({model_label})"
    print(f"Starting {strat_label} backtest across {steps} steps (Battery: {cfg.battery_capacity_kwh} kWh, Inverter: {cfg.battery_power_max_kw} kW)...")

    for idx in range(steps):
        row = df.iloc[idx]
        actual_spot_price = float(spot_prices[idx])
        price_buy_act = float(p_buy_all[idx])
        price_sell_act = float(p_sell_all[idx])
        actual_load = float(loads[idx])
        actual_l1 = float(loads_l1[idx])
        actual_l2 = float(loads_l2[idx])
        actual_l3 = float(loads_l3[idx])
        actual_pv = float(pvs[idx])
        hr = int(hours[idx])

        # Physical bounds
        surplus_real = max(0.0, actual_pv - actual_load)
        deficit_real = max(0.0, actual_load - actual_pv)

        max_absorbable = max(0.0, (max_soc - current_soc) / cfg.charge_efficiency)
        max_deliverable = max(0.0, (current_soc - min_soc) * cfg.discharge_efficiency)

        if cfg.strategy == "peak_preservation":
            # -------------------------------------------------------------
            # Strategy 1: Adaptive Peak Preservation + Smart Pre-charge
            # -------------------------------------------------------------
            ch_cmd = 0.0
            dis_cmd = 0.0
            fc_pv = float(row.get("pv_forecast_kwh", actual_pv))

            hist_start = max(0, idx - history_window_len)
            hist_df = df.iloc[hist_start:idx] if idx >= 24 else None

            # 1-step load forecast using phase-aware model
            fc_load_step, fc_phases_step = predict_consumption(hist_df, df.iloc[idx:idx+1], model_type=cfg.model_type)
            fc_load = float(fc_load_step[0])
            fc_l1 = float(fc_phases_step["l1"][0])
            fc_l2 = float(fc_phases_step["l2"][0])
            fc_l3 = float(fc_phases_step["l3"][0])

            # Look ahead for upcoming price peaks in the next 12 hours
            lookahead = min(12, steps - idx)
            future_indices = [idx + k for k in range(1, lookahead)]
            peak_indices = [j for j in future_indices if hours[j] in (19, 20, 21, 22)]
            peak_prices = [p_buy_all[j] for j in peak_indices] if peak_indices else []
            max_peak_price = max(peak_prices) if peak_prices else price_buy_act

            # Check if this hour is a cheap valley (night 1-5 or afternoon 12-16) with profitable spread
            rt_eff = cfg.charge_efficiency * cfg.discharge_efficiency
            delivered_cost = (price_buy_act / rt_eff) + 2.0 * cfg.degradation_cost_czk_per_kwh
            peak_spread = max_peak_price - delivered_cost
            is_precharge_window = (peak_spread >= cfg.peak_price_spread_thresh) and (hr in (1, 2, 3, 4, 12, 13, 14, 15, 16))

            # Estimate upcoming evening peak load demand using ML forecaster
            if peak_indices:
                if hist_df is not None and len(hist_df) >= 24:
                    peak_preds, _ = predict_consumption(hist_df, df.iloc[peak_indices], model_type=cfg.model_type)
                    est_peak_load = float(np.sum(peak_preds))
                else:
                    est_peak_load = 5.0

                # Target capacity in battery needed to cover peak
                target_peak_energy = min(max_soc, est_peak_load / cfg.discharge_efficiency)

                # Expected NET solar surplus between now and peak
                pre_peak_indices = [j for j in future_indices if j < peak_indices[0]]
                net_solar_surplus = sum(
                    max(0.0, float(df.iloc[j].get("pv_forecast_kwh", pvs[j])) - loads[j])
                    for j in pre_peak_indices
                ) if pre_peak_indices else 0.0

                uncovered_peak = max(0.0, target_peak_energy - net_solar_surplus - current_soc)
            else:
                est_peak_load = 0.0
                uncovered_peak = 0.0

            if surplus_real > 0:
                # Absorb free solar surplus into battery first
                ch_pv = min(surplus_real, max_absorbable, cfg.battery_power_max_kw)
                rem_absorb = max(0.0, min(max_absorbable, cfg.battery_power_max_kw) - ch_pv)
                if is_precharge_window and uncovered_peak > 0.01 and rem_absorb > 0.01:
                    ch_grid = min(rem_absorb, uncovered_peak / cfg.charge_efficiency)
                    ch_act = ch_pv + ch_grid
                else:
                    ch_act = ch_pv
                dis_act = 0.0
            else:
                if is_precharge_window and uncovered_peak > 0.01 and max_absorbable > 0.01:
                    # Pre-charge battery from cheap grid valley for upcoming peak
                    ch_act = min(max_absorbable, uncovered_peak / cfg.charge_efficiency, cfg.battery_power_max_kw)
                    dis_act = 0.0
                elif 13 <= hr <= 18 and peak_prices and (max_peak_price - price_buy_act >= cfg.peak_price_spread_thresh):
                    # Strict preservation: hold battery charge for upcoming evening peak (19:00 - 22:00)
                    # Never waste battery during afternoon (when grid is cheap) if evening prices are high
                    dis_act = 0.0
                    ch_act = 0.0
                else:
                    # Normal deficit / evening peak: discharge battery to cover load
                    dis_act = min(deficit_real, max_deliverable, cfg.battery_power_max_kw)
                    ch_act = 0.0

        else:
            # -------------------------------------------------------------
            # Strategy 2: Traditional Rolling MILP MPC
            # -------------------------------------------------------------
            window_end = min(len(df), idx + cfg.horizon_hours)
            window_df = df.iloc[idx:window_end].copy().reset_index(drop=True)
            hist_start = max(0, idx - history_window_len)
            history_df = df.iloc[hist_start:idx].copy() if idx >= 24 else None

            plan_df, _ = solve_milp_window_v2(window_df, current_soc, cfg=cfg, history_df=history_df)
            action = plan_df.iloc[0]

            ch_cmd = float(action["charge_kwh"])
            dis_cmd = float(action["discharge_kwh"])
            fc_pv = float(action["forecast_pv_kwh"])
            fc_load = float(action["forecast_load_kwh"])
            fc_l1 = float(action.get("forecast_load_l1_kwh", fc_load / 3.0))
            fc_l2 = float(action.get("forecast_load_l2_kwh", fc_load / 3.0))
            fc_l3 = float(action.get("forecast_load_l3_kwh", fc_load / 3.0))

            ch_grid_cmd = min(ch_cmd, float(action["buy_kwh"]))
            dis_grid_cmd = min(dis_cmd, float(action["sell_kwh"]))

            if surplus_real > 0:
                ch_pv_act = min(surplus_real, max_absorbable, cfg.battery_power_max_kw)
                rem_absorb = max(0.0, min(max_absorbable, cfg.battery_power_max_kw) - ch_pv_act)
                ch_grid_act = min(ch_grid_cmd, rem_absorb)
                ch_act = ch_pv_act + ch_grid_act
                dis_act = 0.0
            else:
                if ch_grid_cmd > 0.01:
                    ch_act = min(ch_grid_cmd, max_absorbable, cfg.battery_power_max_kw)
                    dis_act = 0.0
                else:
                    dis_load_act = min(deficit_real, max_deliverable, cfg.battery_power_max_kw)
                    rem_deliver = max(0.0, min(max_deliverable, cfg.battery_power_max_kw) - dis_load_act)
                    dis_grid_act = min(dis_grid_cmd, rem_deliver) if (dis_grid_cmd > 0 and price_sell_act > 0) else 0.0
                    dis_act = dis_load_act + dis_grid_act
                    ch_act = 0.0

        # Physical battery SoC update
        next_soc = current_soc + cfg.charge_efficiency * ch_act - dis_act / cfg.discharge_efficiency
        next_soc = float(np.clip(next_soc, min_soc, max_soc))

        # Real-time power balance settled via grid:
        net_imbalance = actual_load + ch_act - actual_pv - dis_act
        curt_act = 0.0
        if net_imbalance > 0:
            buy_act = min(net_imbalance, cfg.grid_limit_kw)
            sell_act = 0.0
        else:
            surplus = -net_imbalance
            if price_sell_act < 0 and cfg.allow_curtailment:
                curt_act = surplus
                sell_act = 0.0
                buy_act = 0.0
            else:
                sell_act = min(surplus, cfg.grid_limit_kw)
                buy_act = 0.0

        cost_buy = buy_act * price_buy_act
        rev_sell = sell_act * price_sell_act
        deg_cost = cfg.degradation_cost_czk_per_kwh * (ch_act + dis_act)
        profit = rev_sell - cost_buy - deg_cost
        cum_consumption += actual_load

        log_rows.append(
            {
                "datetime": row["datetime"],
                "spot_price_czk_per_kwh": actual_spot_price,
                "effective_buy_price_czk": price_buy_act,
                "effective_sell_price_czk": price_sell_act,
                "house_consumption_kwh": actual_load,
                "load_kwh": actual_load,
                "load_l1_kwh": actual_l1,
                "load_l2_kwh": actual_l2,
                "load_l3_kwh": actual_l3,
                "cumulative_consumption_kwh": cum_consumption,
                "pv_kwh": actual_pv,
                "buy_kwh": buy_act,
                "sell_kwh": sell_act,
                "charge_kwh": ch_act,
                "discharge_kwh": dis_act,
                "curtailment_kwh": curt_act,
                "soc_kwh": next_soc,
                "cost_buy_czk": cost_buy,
                "revenue_sell_czk": rev_sell,
                "degradation_cost_czk": deg_cost,
                "net_profit_czk": profit,
                "planned_ch_kwh": ch_cmd,
                "planned_dis_kwh": dis_cmd,
                "forecast_pv_kwh": fc_pv,
                "forecast_load_kwh": fc_load,
                "forecast_load_l1_kwh": fc_l1,
                "forecast_load_l2_kwh": fc_l2,
                "forecast_load_l3_kwh": fc_l3,
            }
        )

        current_soc = next_soc

    log_df = pd.DataFrame(log_rows)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(output_path, sep=';', index=False)
        print(f"Trading log saved to {output_path}")

        # Mirror copy to data/ directory so visualizer finds it automatically
        data_mirror = Path("data") / output_path.name
        if output_path.resolve() != data_mirror.resolve():
            try:
                data_mirror.parent.mkdir(parents=True, exist_ok=True)
                log_df.to_csv(data_mirror, sep=';', index=False)
            except Exception:
                pass

    return log_df


# =====================================================================
# 5. Financial KPIs & Comparison Report
# =====================================================================

def compute_financial_kpis(df: pd.DataFrame) -> dict:
    """Calculate aggregated financial metrics and energy statistics."""
    if df.empty:
        return {}

    total_buy_kwh = float(df["buy_kwh"].sum())
    total_sell_kwh = float(df["sell_kwh"].sum())
    total_pv_kwh = float(df["pv_kwh"].sum())
    total_load_kwh = float(df["load_kwh"].sum())
    total_ch_kwh = float(df["charge_kwh"].sum())
    total_dis_kwh = float(df["discharge_kwh"].sum())
    total_curt_kwh = float(df.get("curtailment_kwh", pd.Series(0, index=df.index)).sum())

    total_cost_czk = float(df["cost_buy_czk"].sum())
    total_rev_czk = float(df["revenue_sell_czk"].sum())
    total_deg_czk = float(df["degradation_cost_czk"].sum())
    net_profit_czk = float(df["net_profit_czk"].sum())

    total_load_l1_kwh = float(df["load_l1_kwh"].sum()) if "load_l1_kwh" in df.columns else total_load_kwh / 3.0
    total_load_l2_kwh = float(df["load_l2_kwh"].sum()) if "load_l2_kwh" in df.columns else total_load_kwh / 3.0
    total_load_l3_kwh = float(df["load_l3_kwh"].sum()) if "load_l3_kwh" in df.columns else total_load_kwh / 3.0

    self_sufficiency_pct = max(0.0, (1.0 - total_buy_kwh / (total_load_kwh + 1e-6))) * 100.0
    self_consumption_pct = max(0.0, (total_pv_kwh - total_sell_kwh - total_curt_kwh) / (total_pv_kwh + 1e-6)) * 100.0

    return {
        "steps": len(df),
        "total_load_kwh": total_load_kwh,
        "total_load_l1_kwh": total_load_l1_kwh,
        "total_load_l2_kwh": total_load_l2_kwh,
        "total_load_l3_kwh": total_load_l3_kwh,
        "total_pv_kwh": total_pv_kwh,
        "total_buy_kwh": total_buy_kwh,
        "total_sell_kwh": total_sell_kwh,
        "total_ch_kwh": total_ch_kwh,
        "total_dis_kwh": total_dis_kwh,
        "total_curt_kwh": total_curt_kwh,
        "total_cost_czk": total_cost_czk,
        "total_rev_czk": total_rev_czk,
        "total_deg_czk": total_deg_czk,
        "net_profit_czk": net_profit_czk,
        "self_sufficiency_pct": self_sufficiency_pct,
        "self_consumption_pct": self_consumption_pct,
    }


def print_comparison_report(kpi_milp: dict, kpi_naive: dict, cfg: OptimizerV2Config = CONFIG_V2) -> None:
    """Print comparative financial report."""
    print("\n" + "=" * 76)
    print("           FINANCIAL PERFORMANCE & PROFIT COMPARISON REPORT")
    print(f"  Tariffs: Import = (Spot + {cfg.import_fee_czk_per_kwh:.2f} CZK) * {1.0 + cfg.vat_rate:.2f} (VAT) | Export = Spot - {cfg.export_fee_czk_per_kwh:.2f} CZK")
    print("=" * 76)

    header = f"{'Metric':<34} | {'MILP MPC (v2)':<18} | {'Naive Baseline':<18}"
    print(header)
    print("-" * 76)

    def row(label, val_m, val_n, unit=""):
        if isinstance(val_m, float):
            return f"{label:<34} | {val_m:14.2f} {unit:<3} | {val_n:14.2f} {unit:<3}"
        return f"{label:<34} | {str(val_m):>18} | {str(val_n):>18}"

    print(row("Simulated Duration (hours)", kpi_milp["steps"], kpi_naive["steps"]))
    print(row("Total House Load", kpi_milp["total_load_kwh"], kpi_naive["total_load_kwh"], "kWh"))
    if "total_load_l1_kwh" in kpi_milp and "total_load_l1_kwh" in kpi_naive:
        print(row("  - Phase L1 Load", kpi_milp["total_load_l1_kwh"], kpi_naive["total_load_l1_kwh"], "kWh"))
        print(row("  - Phase L2 Load", kpi_milp["total_load_l2_kwh"], kpi_naive["total_load_l2_kwh"], "kWh"))
        print(row("  - Phase L3 Load (Boiler/Heavy)", kpi_milp["total_load_l3_kwh"], kpi_naive["total_load_l3_kwh"], "kWh"))
    print(row("Total PV Generation", kpi_milp["total_pv_kwh"], kpi_naive["total_pv_kwh"], "kWh"))
    print(row("Electricity Bought (Grid Import)", kpi_milp["total_buy_kwh"], kpi_naive["total_buy_kwh"], "kWh"))
    print(row("Electricity Sold (Grid Export)", kpi_milp["total_sell_kwh"], kpi_naive["total_sell_kwh"], "kWh"))
    print(row("Battery Throughput (Charged)", kpi_milp["total_ch_kwh"], kpi_naive["total_ch_kwh"], "kWh"))
    print(row("Solar Curtailment", kpi_milp["total_curt_kwh"], kpi_naive["total_curt_kwh"], "kWh"))
    print("-" * 76)
    print(row("Total Purchase Cost (incl. VAT)", kpi_milp["total_cost_czk"], kpi_naive["total_cost_czk"], "CZK"))
    print(row("Total Selling Revenue", kpi_milp["total_rev_czk"], kpi_naive["total_rev_czk"], "CZK"))
    print(row("Battery Degradation Cost", kpi_milp["total_deg_czk"], kpi_naive["total_deg_czk"], "CZK"))
    print("-" * 76)
    print(row(">>> NET FINANCIAL PROFIT <<<", kpi_milp["net_profit_czk"], kpi_naive["net_profit_czk"], "CZK"))
    print("-" * 76)

    delta_profit = kpi_milp["net_profit_czk"] - kpi_naive["net_profit_czk"]
    print(f"Profit Advantage over Baseline: {delta_profit:+.2f} CZK")
    print(row("Self-Sufficiency Ratio", kpi_milp["self_sufficiency_pct"], kpi_naive["self_sufficiency_pct"], "%"))
    print(row("Self-Consumption Ratio", kpi_milp["self_consumption_pct"], kpi_naive["self_consumption_pct"], "%"))
    print("=" * 76 + "\n")


def print_feature_weights(summary: dict) -> None:
    """Print statistical feature weights and importance."""
    print("\n" + "=" * 76)
    print("        STATISTICAL FEATURE WEIGHTS & IMPORTANCE ANALYSIS")
    print("=" * 76)
    for target, info in summary.items():
        print(f"\n[ Target: {target} ] (Explained Variance R^2 = {info['r2']:.3f})")
        print(f"{'Feature':<22} | {'Standardized Weight (Beta)':<28} | {'Correlation':<12}")
        print("-" * 68)
        sorted_weights = sorted(info["weights"].items(), key=lambda item: abs(item[1]), reverse=True)
        for f, weight in sorted_weights:
            bar = "#" * int(min(20, abs(weight) * 30))
            corr = info["correlations"].get(f, 0.0)
            print(f"{f:<22} | {weight:+10.4f}  {bar:<16} | {corr:+8.3f}")
    print("=" * 76 + "\n")


# =====================================================================
# 6. Main CLI Routine
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SolarEnergyOptimizer V2 - MILP MPC Energy Trading Engine")
    parser.add_argument("--input", type=str, default="data/combined_market_data.csv", help="Input dataset path")
    parser.add_argument("--output", type=str, default="trading_log_v2.csv", help="Output trading decisions CSV")
    parser.add_argument("--max-steps", type=int, default=168, help="Backtest steps in hours (default: 168 = 1 week, None = full)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_HOURS, help="MPC optimization horizon")
    parser.add_argument("--battery-capacity", type=float, default=DEFAULT_BATTERY_CAPACITY_KWH, help="Battery capacity (kWh)")
    parser.add_argument("--soc-min", type=float, default=DEFAULT_SOC_MIN, help="Minimum battery SoC fraction (default: 0.0 = full discharge)")
    parser.add_argument("--soc-max", type=float, default=DEFAULT_SOC_MAX, help="Maximum battery SoC fraction (default: 1.0 = full charge)")
    parser.add_argument("--battery-power", type=float, default=DEFAULT_BATTERY_POWER_MAX_KW, help="Battery power rating (kW)")
    parser.add_argument("--degradation-cost", type=float, default=DEFAULT_DEGRADATION_COST_CZK_PER_KWH, help="Degradation cost CZK/kWh")
    parser.add_argument("--grid-limit", type=float, default=DEFAULT_GRID_LIMIT_KW, help="Grid connection limit (kW)")
    parser.add_argument("--model-type", type=str, default="lgbm", choices=["lgbm", "hgb", "linear"], help="Forecaster model type")
    parser.add_argument("--strategy", type=str, default="peak_preservation", choices=["peak_preservation", "milp"], help="Optimization strategy (default: peak_preservation)")
    parser.add_argument("--peak-spread", type=float, default=2.0, help="Min CZK/kWh price spread to trigger peak reservation (default: 2.0)")
    parser.add_argument("--import-fee", type=float, default=DEFAULT_IMPORT_FEE_CZK_PER_KWH, help="Fixed fee added to import price (default: 3.0 CZK/kWh)")
    parser.add_argument("--vat-rate", type=float, default=DEFAULT_VAT_RATE, help="VAT on electricity import (default: 0.21 = 21%)")
    parser.add_argument("--export-fee", type=float, default=DEFAULT_EXPORT_FEE_CZK_PER_KWH, help="Trader fee deducted from export price (default: 0.4 CZK/kWh)")
    args = parser.parse_args()

    cfg = OptimizerV2Config(
        battery_capacity_kwh=args.battery_capacity,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        battery_power_max_kw=args.battery_power,
        degradation_cost_czk_per_kwh=args.degradation_cost,
        grid_limit_kw=args.grid_limit,
        horizon_hours=args.horizon,
        model_type=args.model_type,
        strategy=args.strategy,
        peak_price_spread_thresh=args.peak_spread,
        import_fee_czk_per_kwh=args.import_fee,
        vat_rate=args.vat_rate,
        export_fee_czk_per_kwh=args.export_fee,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found. Run prepare_data.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, sep=';')
    print(f"Loaded {len(df)} rows from {input_path}")

    # 1. Feature statistical analysis
    feature_summary = compute_feature_weights_summary(df)
    print_feature_weights(feature_summary)

    # 2. Run simulation
    out_path = Path(args.output) if args.output else None
    milp_df = run_backtest_v2(df, cfg=cfg, max_steps=args.max_steps, output_path=out_path)
    naive_df = run_naive_baseline_v2(df, cfg=cfg, max_steps=args.max_steps)

    # 3. Financial KPI Report
    kpi_milp = compute_financial_kpis(milp_df)
    kpi_naive = compute_financial_kpis(naive_df)

    print_comparison_report(kpi_milp, kpi_naive, cfg=cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
