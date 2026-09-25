"""Energy management optimization for PV + battery + grid trading.

This module solves a 24-hour MPC-style MILP for a home microgrid that can trade
energy with the grid. The implementation is intentionally compact but production-
ready enough for backtesting on the project's hourly dataset.

The script supports two usage modes:
1. Solve a single 24-hour optimization window from a supplied forecast table.
2. Run a historical backtest across the merged dataset and produce a CSV trading log.

For the initial version, the "forecast" is approximated by the realized values in
rolling historical windows. This is sufficient for a backtest and can later be
replaced by an ML-based short-horizon forecast engine.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pulp as pl


BATTERY_CAPACITY_KWH_DEFAULT = 10
SOC_MIN_DEFAULT = 0.10
SOC_MAX_DEFAULT = 0.90
BATTERY_POWER_MAX_KWH_DEFAULT = 0.5
CHARGE_EFFICIENCY_DEFAULT = 0.93
DISCHARGE_EFFICIENCY_DEFAULT = 0.93
DEGRADATION_COST_DEFAULT = 0.5
GRID_LIMIT_DEFAULT = 100.0
HORIZON_HOURS_DEFAULT = 24
BIG_M_DEFAULT = 1000.0


@dataclass
class OptimizationConfig:
    battery_capacity_kwh: float = BATTERY_CAPACITY_KWH_DEFAULT
    soc_min: float = SOC_MIN_DEFAULT
    soc_max: float = SOC_MAX_DEFAULT
    battery_power_max_kwh: float = BATTERY_POWER_MAX_KWH_DEFAULT
    charge_efficiency: float = CHARGE_EFFICIENCY_DEFAULT
    discharge_efficiency: float = DISCHARGE_EFFICIENCY_DEFAULT
    degradation_cost_per_kwh: float = DEGRADATION_COST_DEFAULT
    grid_limit_kwh_per_hour: float = GRID_LIMIT_DEFAULT
    horizon_hours: int = HORIZON_HOURS_DEFAULT
    big_m: float = BIG_M_DEFAULT


CONFIG = OptimizationConfig()


def select_real_pv_column(df: pd.DataFrame) -> str:
    """Return the actual current PV generation column, used for realized production."""
    for candidate in ("production_kwh", "pv_generation_kwh", "solar_production", "pv_forecast_kwh"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No actual PV generation column found. Expected one of: "
        "production_kwh, pv_generation_kwh, solar_production, pv_forecast_kwh."
    )


def select_forecast_pv_column(df: pd.DataFrame) -> str:
    """Return the PV forecast column used for optimization decisions."""
    for candidate in ("pv_forecast_kwh", "pv_generation_kwh", "production_kwh", "solar_production"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No PV forecast column found. Expected one of: "
        "pv_forecast_kwh, pv_generation_kwh, production_kwh, solar_production."
    )


def select_pv_column(df: pd.DataFrame) -> str:
    """Backward-compatible alias for the forecast PV column."""
    return select_forecast_pv_column(df)


def prepare_forecast_window(df: pd.DataFrame, start_idx: int, horizon: int) -> pd.DataFrame:
    """Slice a rolling forecast window of length `horizon` from the dataset."""
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    end_idx = min(len(df), start_idx + horizon)
    window = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    if window.empty:
        raise ValueError(f"No data available from index {start_idx} for horizon {horizon}.")

    if "datetime" in window.columns:
        window = window.copy()
        timestamps = pd.to_datetime(window["datetime"], errors="coerce")
        hour = timestamps.dt.hour.fillna(0.0)
        weekday = timestamps.dt.dayofweek.fillna(0.0)
        window["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        window["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        window["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
        window["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
        window["is_weekend"] = (weekday >= 5).astype(float)
        window["weekday"] = weekday.astype(float)

    for col, default in {
        "temperature_c": 15.0,
        "cloud_cover_pct": 50.0,
        "humidity_pct": 50.0,
        "wind_speed_ms": 2.0,
    }.items():
        if col not in window.columns:
            window[col] = default

    return window


DEFAULT_FEATURE_POOL = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "temperature_c",
    "cloud_cover_pct",
    "humidity_pct",
    "wind_speed_ms",
)
FEATURE_CACHE_PATH = Path("data/feature_selection_cache.json")


def _ensure_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/weather features required by the statistical forecast model."""
    prepared = frame.copy()
    if "datetime" in prepared.columns:
        timestamps = pd.to_datetime(prepared["datetime"], errors="coerce")
        hour = timestamps.dt.hour.fillna(0.0).to_numpy(dtype=float)
        weekday = timestamps.dt.dayofweek.fillna(0.0).to_numpy(dtype=float)
        prepared["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        prepared["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        prepared["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
        prepared["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
        prepared["is_weekend"] = (weekday >= 5).astype(float)
    else:
        n = len(prepared)
        prepared["hour_sin"] = np.sin(2.0 * np.pi * np.arange(n) / 24.0)
        prepared["hour_cos"] = np.cos(2.0 * np.pi * np.arange(n) / 24.0)
        prepared["weekday_sin"] = 0.0
        prepared["weekday_cos"] = 0.0
        prepared["is_weekend"] = 0.0

    for col, default in {
        "temperature_c": 15.0,
        "cloud_cover_pct": 50.0,
        "humidity_pct": 50.0,
        "wind_speed_ms": 2.0,
    }.items():
        if col not in prepared.columns:
            prepared[col] = default

    return prepared


def add_regression_features(df: pd.DataFrame, feature_names: Optional[Iterable[str]] = None) -> tuple[np.ndarray, list[str]]:
    """Create a dense feature matrix with hourly and weekday cyclic effects.

    When a subset is supplied, the model only uses that subset of features. This lets
    the optimizer automatically choose the best attributes for yearly / rolling training runs.
    """
    frame = _ensure_temporal_features(df)
    if feature_names is None:
        selected = ["const", *DEFAULT_FEATURE_POOL]
    else:
        selected = ["const", *[name for name in feature_names if name != "const"]]

    matrix = np.ones((len(frame), len(selected)), dtype=float)
    for idx, name in enumerate(selected[1:], start=1):
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            values = np.zeros(len(frame), dtype=float)
        matrix[:, idx] = values
    return matrix, selected


def _make_yearly_or_rolling_folds(df: pd.DataFrame) -> list[list[int]]:
    """Split a temporal dataset into folds for systematic feature selection.

    If more than one year is present, a leave-one-year-out strategy is used.
    Otherwise the data is split into rolling blocks for the same yearly iteration logic.
    """
    if "datetime" in df.columns:
        timestamps = pd.to_datetime(df["datetime"], errors="coerce")
        years = sorted(timestamps.dropna().dt.year.unique())
        if len(years) > 1:
            return [df.index[timestamps.dt.year == year].tolist() for year in years]

    if len(df) < 24:
        return [list(range(len(df)))]

    fold_count = min(4, max(2, len(df) // 24))
    blocks = np.array_split(np.arange(len(df)), fold_count)
    return [list(block) for block in blocks if len(block) > 0]


def load_feature_cache(cache_path: str | Path = FEATURE_CACHE_PATH) -> dict:
    """Load cached selected features from disk, if available."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_feature_cache(cache: dict, cache_path: str | Path = FEATURE_CACHE_PATH) -> None:
    """Persist selected feature choices for future runs."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def _score_subset_for_target(subset: tuple[str, ...], frame: pd.DataFrame, target_col: str) -> tuple[tuple[str, ...], float]:
    """Evaluate a specific feature subset across temporal validation folds."""
    fold_scores: list[float] = []
    for fold in _make_yearly_or_rolling_folds(frame):
        train_idx = [idx for idx in range(len(frame)) if idx not in fold]
        test_idx = fold
        if len(train_idx) < 5 or len(test_idx) < 5:
            continue

        train_df = frame.iloc[train_idx].copy()
        test_df = frame.iloc[test_idx].copy()
        train_y = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        test_y = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        x_train, _ = add_regression_features(train_df, feature_names=["const", *list(subset)])
        x_test, _ = add_regression_features(test_df, feature_names=["const", *list(subset)])
        beta, *_ = np.linalg.lstsq(x_train, train_y, rcond=None)
        predictions = x_test @ beta
        fold_scores.append(float(np.sqrt(np.mean((test_y - predictions) ** 2))))

    if not fold_scores:
        return subset, float("inf")

    score = float(np.mean(fold_scores)) + 0.01 * len(subset)
    return subset, score


def select_optimal_attributes(
    df: pd.DataFrame,
    target_col: str,
    max_features: int = 5,
    cache_path: str | Path = FEATURE_CACHE_PATH,
    batch_size: int = 4,
    use_cache: bool = True,
) -> list[str]:
    """Select the most informative attributes using cross-validation across yearly/rolling folds.

    The first run evaluates candidate subsets in batches using parallel workers to keep the
    search fast. The result is cached to disk and reused on later runs.
    """
    frame = _ensure_temporal_features(df.copy())
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataframe.")

    cache_key = f"{target_col}:{max_features}"
    if use_cache:
        cached = load_feature_cache(cache_path)
        if cache_key in cached:
            return list(cached[cache_key])

    available = [name for name in DEFAULT_FEATURE_POOL if name in frame.columns or name in {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend"}]
    if not available:
        return []

    max_features = max(1, min(max_features, len(available)))
    candidate_subsets: list[tuple[str, ...]] = []
    for size in range(1, max_features + 1):
        candidate_subsets.extend(combinations(available, size))

    if not candidate_subsets:
        return available[:max_features]

    best_subset: list[str] = []
    best_score = np.inf
    max_workers = min(len(candidate_subsets), max(1, os.cpu_count() or 1))

    for offset in range(0, len(candidate_subsets), max(1, batch_size)):
        batch = candidate_subsets[offset:offset + max(1, batch_size)]
        if len(batch) == 1:
            subset, score = _score_subset_for_target(batch[0], frame, target_col)
            scored = [(subset, score)]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                scored = list(executor.map(lambda sub: _score_subset_for_target(sub, frame, target_col), batch))

        for subset, score in scored:
            if score < best_score or (abs(score - best_score) < 1e-9 and len(subset) < len(best_subset)):
                best_score = score
                best_subset = list(subset)

    if not best_subset:
        best_subset = available[:max_features]

    if use_cache:
        cached = load_feature_cache(cache_path)
        cached[cache_key] = best_subset
        save_feature_cache(cached, cache_path)

    return best_subset


def fit_forecast_model(train_df: pd.DataFrame, target_col: str, selected_features: Optional[Iterable[str]] = None) -> dict:
    """Fit a rolling OLS model that learns the effect of weekday and weather on one target."""
    if train_df.empty:
        raise ValueError(f"Cannot fit forecast model on empty data for target '{target_col}'.")

    if selected_features is None:
        selected_features = select_optimal_attributes(train_df, target_col, max_features=5)

    y = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x, feature_names = add_regression_features(train_df, feature_names=["const", *list(selected_features)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"target_col": target_col, "feature_names": feature_names, "coefficients": beta}


def summarize_feature_selection(
    df: pd.DataFrame,
    target_col: str,
    max_features: int = 5,
) -> dict:
    """Return the selected features and their learned weights in a console-friendly format."""
    selected = select_optimal_attributes(df, target_col, max_features=max_features)
    if not selected:
        return {"selected_features": [], "weights": {}}

    model = fit_forecast_model(df, target_col, selected_features=selected)
    weights = {
        name: float(value)
        for name, value in zip(model["feature_names"], model["coefficients"])
        if name in selected
    }
    return {"selected_features": selected, "weights": weights}


def predict_forecast_series(model: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict the target over a future frame using the learned regression coefficients."""
    x, _ = add_regression_features(frame, feature_names=model["feature_names"])
    return np.asarray(x @ model["coefficients"], dtype=float)


def estimate_target_forecast(window: pd.DataFrame, history_df: pd.DataFrame, target_col: str) -> np.ndarray:
    """Use a simple OLS forecast trained on historical data to predict the target on a future window."""
    if history_df is None or history_df.empty:
        series = pd.to_numeric(window[target_col], errors="coerce").fillna(0.0).astype(float)
        return series.to_numpy(dtype=float)

    history = history_df.copy()
    if len(history) < 24:
        series = pd.to_numeric(window[target_col], errors="coerce").fillna(0.0).astype(float)
        return series.to_numpy(dtype=float)

    model = fit_forecast_model(history, target_col)
    predictions = predict_forecast_series(model, window)
    if target_col in {"consumption_kwh", "pv_forecast_kwh", "pv_generation_kwh", "production_kwh", "solar_production"}:
        predictions = np.clip(predictions, 0.0, None)
    return predictions.astype(float)


def estimate_pv_forecast(window: pd.DataFrame, pv_col: str, history_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Return a weather- and weekday-aware PV forecast learned from historical data."""
    return estimate_target_forecast(window, history_df, pv_col)


def solve_milp_window(
    window: pd.DataFrame,
    soc_start_kwh: float,
    cfg: OptimizationConfig = CONFIG,
    history_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """Solve the MILP on a single time window and return the plan plus metadata.

    The optimization model follows the standard energy trading formulation:

    E_buy + PV + E_dis = Load + E_sell + E_ch
    SOC_t = SOC_{t-1} + E_ch * eta_ch - E_dis / eta_dis
    E_buy, E_sell, E_ch, E_dis >= 0
    E_buy <= M * u, E_sell <= M * (1-u)
    SoC_min <= SOC <= SoC_max

    Objective:
        minimize price * buy - price * sell + degradation_cost * (charge + discharge)
    """
    required_columns = {"datetime", "consumption_kwh", "price_czk_per_kwh"}
    forecast_pv_col = select_forecast_pv_column(window)
    required_columns.add(forecast_pv_col)
    missing = required_columns - set(window.columns)
    if missing:
        raise ValueError(f"Missing required columns for optimization: {sorted(missing)}")

    t_steps = len(window)
    load = estimate_target_forecast(window, history_df, "consumption_kwh") if history_df is not None else window["consumption_kwh"].to_numpy(dtype=float)
    pv = estimate_pv_forecast(window, forecast_pv_col, history_df)
    price = estimate_target_forecast(window, history_df, "price_czk_per_kwh") if history_df is not None else window["price_czk_per_kwh"].to_numpy(dtype=float)

    min_soc_kwh = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc_kwh = cfg.soc_max * cfg.battery_capacity_kwh
    soc_start_kwh = float(np.clip(soc_start_kwh, min_soc_kwh, max_soc_kwh))

    prob = pl.LpProblem("EMS_Trading_MILP", pl.LpMinimize)

    buy = [pl.LpVariable(f"buy_{i}", lowBound=0) for i in range(t_steps)]
    sell = [pl.LpVariable(f"sell_{i}", lowBound=0) for i in range(t_steps)]
    ch = [pl.LpVariable(f"charge_{i}", lowBound=0) for i in range(t_steps)]
    dis = [pl.LpVariable(f"discharge_{i}", lowBound=0) for i in range(t_steps)]
    soc = [
        pl.LpVariable(f"soc_{i}", lowBound=min_soc_kwh, upBound=max_soc_kwh)
        for i in range(t_steps)
    ]

    objective = []
    for i in range(t_steps):
        objective.append(
            buy[i] * price[i]
            - sell[i] * price[i]
            + cfg.degradation_cost_per_kwh * (ch[i] + dis[i])
        )
    prob += pl.lpSum(objective)

    for i in range(t_steps):
        prob += buy[i] + pv[i] + dis[i] == load[i] + sell[i] + ch[i]
        prob += buy[i] <= cfg.grid_limit_kwh_per_hour
        prob += sell[i] <= cfg.grid_limit_kwh_per_hour
        prob += ch[i] <= cfg.battery_power_max_kwh
        prob += dis[i] <= cfg.battery_power_max_kwh

        discharge_reciprocal = 1.0 / cfg.discharge_efficiency
        cumulative_charge = pl.lpSum(ch[: i + 1])
        cumulative_discharge = pl.lpSum(dis[: i + 1])
        prob += soc[i] == soc_start_kwh + cfg.charge_efficiency * cumulative_charge - cumulative_discharge * discharge_reciprocal

    for i in range(t_steps):
        prob += soc[i] >= min_soc_kwh
        prob += soc[i] <= max_soc_kwh

    solver = pl.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pl.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP solver failed for window with status: {pl.LpStatus[status]}")

    results = []
    for i in range(t_steps):
        results.append(
            {
                "datetime": window["datetime"].iloc[i],
                "buy_kwh": float(pl.value(buy[i])),
                "sell_kwh": float(pl.value(sell[i])),
                "charge_kwh": float(pl.value(ch[i])),
                "discharge_kwh": float(pl.value(dis[i])),
                "soc_kwh": float(pl.value(soc[i])),
                "price_czk_per_kwh": float(price[i]),
            }
        )

    plan_df = pd.DataFrame(results)
    metadata = {
        "status": pl.LpStatus[status],
        "objective_value": float(pl.value(prob.objective)),
        "soc_start_kwh": soc_start_kwh,
        "soc_end_kwh": float(pl.value(soc[t_steps - 1])) if t_steps else soc_start_kwh,
    }
    return plan_df, metadata


def naive_policy(df: pd.DataFrame, start_soc_kwh: float, cfg: OptimizationConfig = CONFIG) -> pd.DataFrame:
    """A simple rule-based baseline strategy used for comparison in backtesting.

    The logic is intentionally simple:
    - use local PV first
    - charge battery with surplus if below SoC max
    - discharge battery to cover deficit whenever possible
    - otherwise buy or sell on the grid
    """
    records = []
    soc = float(np.clip(start_soc_kwh, 0.0, cfg.battery_capacity_kwh))
    min_soc = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc = cfg.soc_max * cfg.battery_capacity_kwh
    pv_col = select_real_pv_column(df)

    for _, row in df.iterrows():
        price = float(row["price_czk_per_kwh"])
        load = float(row["consumption_kwh"])
        pv = float(row[pv_col])
        net = pv - load

        buy = 0.0
        sell = 0.0
        charge = 0.0
        discharge = 0.0

        if net > 0:
            surplus = net
            if soc < max_soc:
                charge = min(surplus, max_soc - soc, cfg.battery_power_max_kwh)
                soc = soc + charge * cfg.charge_efficiency
                surplus -= charge
            if surplus > 1e-9:
                sell = min(surplus, cfg.grid_limit_kwh_per_hour)
                soc = min(soc, max_soc)
        else:
            deficit = -net
            if soc > min_soc:
                discharge = min(deficit, soc - min_soc, cfg.battery_power_max_kwh)
                soc = soc - discharge / cfg.discharge_efficiency
                deficit -= discharge
            if deficit > 1e-9:
                buy = min(deficit, cfg.grid_limit_kwh_per_hour)

        soc = float(np.clip(soc, min_soc, max_soc))
        records.append(
            {
                "datetime": row["datetime"],
                "price_czk_per_kwh": price,
                "buy_kwh": buy,
                "sell_kwh": sell,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "soc_kwh": soc,
            }
        )

    return pd.DataFrame(records)


def compute_metrics(log_df: pd.DataFrame, cfg: OptimizationConfig = CONFIG) -> dict:
    """Compute financial KPIs from the hourly trading log."""
    if log_df.empty:
        return {
            "total_buy_kwh": 0.0,
            "total_sell_kwh": 0.0,
            "total_cost_czk": 0.0,
            "total_revenue_czk": 0.0,
            "total_degradation_cost_czk": 0.0,
            "net_profit_czk": 0.0,
        }

    buy = log_df["buy_kwh"].fillna(0.0).sum()
    sell = log_df["sell_kwh"].fillna(0.0).sum()
    cost = (log_df["buy_kwh"].fillna(0.0) * log_df["price_czk_per_kwh"].fillna(0.0)).sum()
    revenue = (log_df["sell_kwh"].fillna(0.0) * log_df["price_czk_per_kwh"].fillna(0.0)).sum()
    degradation = cfg.degradation_cost_per_kwh * (
        log_df["charge_kwh"].fillna(0.0).sum() + log_df["discharge_kwh"].fillna(0.0).sum()
    )
    net_profit = revenue - cost - degradation

    return {
        "total_buy_kwh": float(buy),
        "total_sell_kwh": float(sell),
        "total_cost_czk": float(cost),
        "total_revenue_czk": float(revenue),
        "total_degradation_cost_czk": float(degradation),
        "net_profit_czk": float(net_profit),
    }


def run_backtest(
    df: pd.DataFrame,
    cfg: OptimizationConfig = CONFIG,
    max_steps: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute the rolling MPC simulation over the historical dataset."""
    required = {"datetime", "consumption_kwh", "price_czk_per_kwh"}
    actual_pv_col = select_real_pv_column(df)
    forecast_pv_col = select_forecast_pv_column(df)
    required.update({actual_pv_col, forecast_pv_col})
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset missing required columns: {sorted(missing)}")

    df = df.sort_values("datetime").reset_index(drop=True)
    if max_steps is None:
        max_steps = len(df)

    initial_soc = float(df["battery_kwh"].dropna().iloc[0]) if "battery_kwh" in df.columns and not df["battery_kwh"].dropna().empty else 0.25
    initial_soc = float(np.clip(initial_soc, cfg.soc_min * cfg.battery_capacity_kwh, cfg.soc_max * cfg.battery_capacity_kwh))

    log_rows = []
    current_soc = initial_soc

    for idx in range(min(len(df), max_steps)):
        window = prepare_forecast_window(df, idx, cfg.horizon_hours)
        history = df.iloc[max(0, idx - cfg.horizon_hours * 7):idx].copy() if idx > 0 else df.iloc[:idx].copy()
        plan_df, _ = solve_milp_window(window, current_soc, cfg=cfg, history_df=history)
        if plan_df.empty:
            raise RuntimeError(f"Optimization returned no plan at step index {idx}.")

        action = plan_df.iloc[0]
        row = df.iloc[idx]
        price = float(row["price_czk_per_kwh"])
        buy = float(action["buy_kwh"])
        sell = float(action["sell_kwh"])
        charge = float(action["charge_kwh"])
        discharge = float(action["discharge_kwh"])

        # Apply the first-hour decision and update SoC.
        next_soc = current_soc + cfg.charge_efficiency * charge - discharge / cfg.discharge_efficiency
        next_soc = float(np.clip(next_soc, cfg.soc_min * cfg.battery_capacity_kwh, cfg.soc_max * cfg.battery_capacity_kwh))

        cost_buy = buy * price
        revenue_sell = sell * price
        net_cf = revenue_sell - cost_buy
        degradation_cost = cfg.degradation_cost_per_kwh * (charge + discharge)

        log_rows.append(
            {
                "datetime": row["datetime"],
                "price_czk_per_kwh": price,
                "buy_kwh": buy,
                "sell_kwh": sell,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "soc_kwh": next_soc,
                "cost_buy_czk": cost_buy,
                "revenue_sell_czk": revenue_sell,
                "net_cash_flow_czk": net_cf,
                "degradation_cost_czk": degradation_cost,
            }
        )

        current_soc = next_soc

    log_df = pd.DataFrame(log_rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(output_path, index=False, sep=';')

    return log_df


def run_naive_baseline(
    df: pd.DataFrame,
    cfg: OptimizationConfig = CONFIG,
    max_steps: Optional[int] = None,
) -> pd.DataFrame:
    """Run the baseline controller for comparison and return a log-like dataframe."""
    df = df.sort_values("datetime").reset_index(drop=True)
    if max_steps is not None:
        df = df.iloc[:max_steps].copy()

    initial_soc = float(df["battery_kwh"].dropna().iloc[0]) if "battery_kwh" in df.columns and not df["battery_kwh"].dropna().empty else 0.25
    baseline = naive_policy(df, initial_soc, cfg)
    baseline["cost_buy_czk"] = baseline["buy_kwh"] * baseline["price_czk_per_kwh"]
    baseline["revenue_sell_czk"] = baseline["sell_kwh"] * baseline["price_czk_per_kwh"]
    baseline["net_cash_flow_czk"] = baseline["revenue_sell_czk"] - baseline["cost_buy_czk"]
    baseline["degradation_cost_czk"] = cfg.degradation_cost_per_kwh * (baseline["charge_kwh"] + baseline["discharge_kwh"])
    return baseline


def print_summary(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"Total bought: {metrics['total_buy_kwh']:.3f} kWh")
    print(f"Total sold:   {metrics['total_sell_kwh']:.3f} kWh")
    print(f"Total cost:   {metrics['total_cost_czk']:.2f} CZK")
    print(f"Total revenue:{metrics['total_revenue_czk']:.2f} CZK")
    print(f"Battery deg.: {metrics['total_degradation_cost_czk']:.2f} CZK")
    print(f"Net profit:   {metrics['net_profit_czk']:.2f} CZK")


def print_feature_selection_summary(df: pd.DataFrame, max_features: int = 5) -> None:
    """Print the selected feature subset and learned regression weights for each target."""
    print("\n=== Feature selection summary ===")
    for target in ("price_czk_per_kwh", "consumption_kwh", "pv_forecast_kwh"):
        if target not in df.columns:
            continue
        info = summarize_feature_selection(df, target, max_features=max_features)
        if not info["selected_features"]:
            continue
        print(f"\nTarget: {target}")
        print("Selected features:", ", ".join(info["selected_features"]))
        sorted_weights = sorted(info["weights"].items(), key=lambda item: abs(item[1]), reverse=True)
        for name, weight in sorted_weights:
            print(f"  {name}: {weight:.6f}")
        print(f"  intercept: {float(list(fit_forecast_model(df, target, selected_features=info['selected_features'])['coefficients'])[0]):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PV + Battery + Grid energy trading optimizer")
    parser.add_argument("--input", type=str, default="data/combined_market_data.csv", help="Path to the hourly dataset CSV")
    parser.add_argument("--output", type=str, default="trading_log.csv", help="Destination CSV for trading decisions")
    parser.add_argument("--horizon", type=int, default=HORIZON_HOURS_DEFAULT, help="Optimization horizon in hours")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for historical backtest length")
    parser.add_argument("--battery-capacity", type=float, default=BATTERY_CAPACITY_KWH_DEFAULT)
    parser.add_argument("--soc-min", type=float, default=SOC_MIN_DEFAULT)
    parser.add_argument("--soc-max", type=float, default=SOC_MAX_DEFAULT)
    parser.add_argument("--battery-power-max", type=float, default=BATTERY_POWER_MAX_KWH_DEFAULT)
    parser.add_argument("--charge-eff", type=float, default=CHARGE_EFFICIENCY_DEFAULT)
    parser.add_argument("--discharge-eff", type=float, default=DISCHARGE_EFFICIENCY_DEFAULT)
    parser.add_argument("--degradation-cost", type=float, default=DEGRADATION_COST_DEFAULT)
    parser.add_argument("--grid-limit", type=float, default=GRID_LIMIT_DEFAULT)
    parser.add_argument("--big-m", type=float, default=BIG_M_DEFAULT)
    args = parser.parse_args()

    cfg = OptimizationConfig(
        battery_capacity_kwh=args.battery_capacity,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        battery_power_max_kwh=args.battery_power_max,
        charge_efficiency=args.charge_eff,
        discharge_efficiency=args.discharge_eff,
        degradation_cost_per_kwh=args.degradation_cost,
        grid_limit_kwh_per_hour=args.grid_limit,
        horizon_hours=args.horizon,
        big_m=args.big_m,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path, sep=';')
    if "datetime" not in df.columns:
        raise ValueError("Dataset must include a 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    print_feature_selection_summary(df, max_features=min(5, max(2, len(df) // 24)))

    log_df = run_backtest(df, cfg=cfg, max_steps=args.max_steps, output_path=Path(args.output))
    naive_df = run_naive_baseline(df, cfg=cfg, max_steps=args.max_steps)

    metrics_milp = compute_metrics(log_df, cfg=cfg)
    metrics_naive = compute_metrics(naive_df, cfg=cfg)

    print_summary("MILP MPC Strategy", metrics_milp)
    print_summary("Naive Baseline", metrics_naive)

    delta = metrics_milp["net_profit_czk"] - metrics_naive["net_profit_czk"]
    print(f"\nImprovement vs naive: {delta:.2f} CZK")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
