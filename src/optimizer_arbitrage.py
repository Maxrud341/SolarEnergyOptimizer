"""Optimizer Arbitrage - Commercial Battery-to-Grid (B2G) Microgrid EMS.

Implements true commercial spot market arbitrage:
- Uncontrollable / inelastic household consumption (zero synthetic data shifting).
- 24-hour Rolling Horizon Model Predictive Control (MPC) with hourly closed-loop updates.
- Aggressive battery utilization: SoC from 5% to 100% (95% usable capacity).
- Battery-to-Grid (B2G) export: Battery can discharge at full inverter power into the grid
  during high spot price peaks to capture maximum market spread.
- Dynamic curtailment: automatic Zero Export enforcement when export prices are negative.
- Czech per-phase metering (Fázové řízení) support.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pulp as pl

from src.optimizer_v3 import (
    ForecasterEngineV3,
    compute_effective_buy_price,
    compute_effective_sell_price,
    filter_dataset_by_date_or_month,
)


@dataclass(frozen=True)
class OptimizerArbitrageConfig:
    battery_capacity_kwh: float = 10.0
    soc_min: float = 0.05  # 5% aggressive limit
    soc_max: float = 1.00  # 100% full capacity
    battery_power_max_kw: float = 3.6  # Standard continuous inverter battery power limit
    charge_efficiency: float = 0.90
    discharge_efficiency: float = 0.90
    degradation_cost_czk_per_kwh: float = 0.25  # Battery cycle degradation
    grid_limit_kw: float = 10.0
    import_fee_czk_per_kwh: float = 1.80  # Czech distribution grid fee
    vat_rate: float = 0.21  # Czech VAT
    export_fee_czk_per_kwh: float = 0.45  # Trader export fee
    allow_curtailment: bool = True
    asymmetric_metering: bool = True
    mpc_horizon_hours: int = 24
    model_type: str = "lgbm"
    forecast_pv_safety_factor: float = 0.80  # Conservative discount on PV forecast (uncertainty protection)
    forecast_load_safety_factor: float = 1.15  # Safety buffer on household load forecast
    preflush_min_surplus_kwh: float = 3.0  # Minimum conservative surplus above headroom for extreme pre-flush
    preflush_min_sell_price_czk: float = 2.00  # Minimum profitable morning sell price for extreme pre-flush


CONFIG_ARBITRAGE = OptimizerArbitrageConfig()


def solve_mpc_step(
    current_soc_kwh: float,
    window_df: pd.DataFrame,
    cfg: OptimizerArbitrageConfig,
    engine: ForecasterEngineV3,
) -> Tuple[float, float, float, float, float]:
    """Solve 24h rolling horizon LP and return immediate dispatch and regime metadata."""
    t_steps = len(window_df)
    if t_steps == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Predictions for the rolling horizon
    spot_prices = pd.to_numeric(window_df["price_czk_per_kwh"], errors="coerce").fillna(2.0).to_numpy(dtype=float)
    p_buy_pred = compute_effective_buy_price(spot_prices, cfg)
    p_sell_pred = compute_effective_sell_price(spot_prices, cfg)

    # Use ML forecaster for PV and Loads across horizon
    pv_pred = engine.predict_target("production_kwh", window_df)
    load_pred, _ = engine.predict_load(window_df)

    if "datetime" in window_df.columns:
        hours = pd.to_datetime(window_df["datetime"]).dt.hour.to_numpy()
    elif isinstance(window_df.index, pd.DatetimeIndex):
        hours = window_df.index.hour.to_numpy()
    else:
        hours = np.zeros(t_steps, dtype=int)

    min_soc = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc = cfg.soc_max * cfg.battery_capacity_kwh
    cur_soc = float(np.clip(current_soc_kwh, min_soc, max_soc))

    # Conservative forecast with safety margin for weather/consumption uncertainty
    pv_safe = np.maximum(0.0, pv_pred) * cfg.forecast_pv_safety_factor
    load_safe = np.maximum(0.0, load_pred) * cfg.forecast_load_safety_factor
    solar_surplus_safe = float(np.sum(np.maximum(0.0, pv_safe - load_safe)))
    headroom = max_soc - cur_soc

    # Sunny day detection: conservative surplus is ample to fill battery headroom
    is_sunny_day = solar_surplus_safe >= max(1.5, headroom * 0.85)

    # Find the peak solar hour of the upcoming daylight period
    daylight_indices = [t for t in range(t_steps) if 6 <= hours[t] <= 18 and pv_pred[t] > 0.5]
    first_peak_idx = daylight_indices[int(np.argmax([pv_pred[t] for t in daylight_indices]))] if len(daylight_indices) > 0 else -1

    # Pre-flush curtailment risk: conservative surplus that exceeds battery headroom
    curtailment_risk = max(0.0, solar_surplus_safe - headroom)
    has_negative_prices = any(p_sell_pred[t] < 0 for t in range(t_steps))

    prob = pl.LpProblem("Microgrid_Arbitrage_MPC", pl.LpMinimize)

    # Decision variables
    buy = [pl.LpVariable(f"b_{t}", lowBound=0.0, upBound=cfg.grid_limit_kw) for t in range(t_steps)]
    sell = [pl.LpVariable(f"s_{t}", lowBound=0.0, upBound=cfg.grid_limit_kw) for t in range(t_steps)]
    ch = [pl.LpVariable(f"c_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    dis = [pl.LpVariable(f"d_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    soc = [pl.LpVariable(f"soc_{t}", lowBound=min_soc, upBound=max_soc) for t in range(t_steps)]
    curt = [pl.LpVariable(f"curt_{t}", lowBound=0.0, upBound=max(0.0, float(pv_pred[t]))) for t in range(t_steps)]

    # Cost terms
    # Terminal SoC valuation: Stored energy at the horizon end has future value because it avoids buying electricity tomorrow.
    # Displacing grid purchases tomorrow saves ~mean(p_buy_pred) * discharge_efficiency - degradation.
    # Setting an appropriate terminal valuation prevents premature dumping of stored energy to the grid.
    avoided_purchase_val = float(np.mean(p_buy_pred)) * cfg.discharge_efficiency - cfg.degradation_cost_czk_per_kwh
    export_val = float(np.mean(p_sell_pred)) + cfg.degradation_cost_czk_per_kwh
    terminal_val = max(0.5, min(avoided_purchase_val, export_val * 1.5))
    cost_terms = [-terminal_val * soc[t_steps - 1]]

    inv_dis_eff = 1.0 / max(1e-4, cfg.discharge_efficiency)

    for t in range(t_steps):
        cost_terms.append(buy[t] * float(p_buy_pred[t]))
        cost_terms.append(-sell[t] * float(p_sell_pred[t]))
        cost_terms.append(cfg.degradation_cost_czk_per_kwh * (ch[t] + dis[t]))
        cost_terms.append(0.005 * curt[t])

        # Instantaneous energy balance: buy + pv + dis == load + sell + ch + curt
        prob += buy[t] + float(pv_pred[t]) + dis[t] == float(load_pred[t]) + sell[t] + ch[t] + curt[t]

        # Battery SoC continuity equation
        prev = cur_soc if t == 0 else soc[t - 1]
        prob += soc[t] == prev + cfg.charge_efficiency * ch[t] - dis[t] * inv_dis_eff

        # Zero Export restriction when effective sell price is negative
        if p_sell_pred[t] < 0 and cfg.allow_curtailment:
            prob += sell[t] == 0.0

        # Daytime grid charging restriction: with positive spot prices, battery charges ONLY from solar surplus
        if (6 <= hours[t] <= 21) and (spot_prices[t] >= 0):
            surplus_pv = max(0.0, float(pv_pred[t]) - float(load_pred[t]))
            prob += ch[t] <= surplus_pv

        # Rule 1: Zero Grid Pre-Charge before a sunny day.
        # Only active in nighttime hours (00:00 - 05:00) before daytime solar starts.
        if is_sunny_day and (0 <= hours[t] <= 5) and (first_peak_idx > t):
            prob += ch[t] <= float(pv_pred[t])

        # Rule 2: Guard Solar Pre-Flush - only allowed in extreme surplus cases.
        # Only active in early morning hours (04:00 - 09:00) before peak solar irradiance.
        # Evening hours (17:00 - 23:00) commercial B2G export is 100% UNRESTRICTED.
        if (4 <= hours[t] <= 9) and (first_peak_idx > t):
            allow_preflush = (
                (curtailment_risk >= cfg.preflush_min_surplus_kwh)
                and has_negative_prices
                and (p_sell_pred[t] >= cfg.preflush_min_sell_price_czk)
            )
            if not allow_preflush:
                prob += sell[t] <= float(pv_pred[t])
            else:
                prob += sell[t] <= float(pv_pred[t]) + curtailment_risk

    prob += pl.lpSum(cost_terms)

    # Solve LP (CBC solver without message)
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    if pl.LpStatus[prob.status] not in ("Optimal", "Feasible"):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    first_ch = max(0.0, float(pl.value(ch[0])) if pl.value(ch[0]) is not None else 0.0)
    first_dis = max(0.0, float(pl.value(dis[0])) if pl.value(dis[0]) is not None else 0.0)
    first_sell = max(0.0, float(pl.value(sell[0])) if pl.value(sell[0]) is not None else 0.0)

    # Planned grid charge at t=0 (if LP planned charge beyond predicted PV surplus)
    pv_surplus_pred_0 = max(0.0, float(pv_pred[0]) - float(load_pred[0]))
    planned_grid_charge = max(0.0, first_ch - pv_surplus_pred_0)

    # Check future peak export reservation:
    # Does any future hour t > 0 have a commercial B2G export whose net profit exceeds buying power now?
    p_buy_now = float(p_buy_pred[0])
    reserved_soc = 0.0
    for t in range(1, t_steps):
        s_val = float(pl.value(sell[t])) if pl.value(sell[t]) is not None else 0.0
        pv_val = float(pv_pred[t])
        if s_val > pv_val + 0.1:
            net_peak_rev = float(p_sell_pred[t]) * cfg.discharge_efficiency - cfg.degradation_cost_czk_per_kwh
            if net_peak_rev > p_buy_now:
                export_from_bat = s_val - pv_val
                needed_soc = export_from_bat / max(1e-4, cfg.discharge_efficiency)
                reserved_soc = max(reserved_soc, needed_soc)

    return first_ch, first_dis, first_sell, planned_grid_charge, reserved_soc


def run_arbitrage_backtest(
    df: pd.DataFrame,
    cfg: OptimizerArbitrageConfig = CONFIG_ARBITRAGE,
    max_steps: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute closed-loop 24h rolling horizon MPC backtest."""
    steps = len(df) if max_steps is None else min(len(df), max_steps)
    if steps == 0:
        return pd.DataFrame()

    engine = ForecasterEngineV3(model_type=cfg.model_type)
    history_len = 168

    min_bat_soc = cfg.battery_capacity_kwh * cfg.soc_min
    max_bat_soc = cfg.battery_capacity_kwh * cfg.soc_max
    cur_bat_soc = cfg.battery_capacity_kwh * 0.5
    wacse = 2.5  # Weighted Average Cost of Stored Energy (CZK/kWh)

    spot_prices = pd.to_numeric(df["price_czk_per_kwh"], errors="coerce").fillna(2.0).to_numpy(dtype=float)
    p_buy_all = compute_effective_buy_price(spot_prices, cfg)
    p_sell_all = compute_effective_sell_price(spot_prices, cfg)

    loads_tot = pd.to_numeric(df["consumption_kwh"], errors="coerce").fillna(0.3).to_numpy(dtype=float)
    has_phases = all(c in df.columns for c in ("consumption_l1_kwh", "consumption_l2_kwh", "consumption_l3_kwh"))
    if has_phases:
        loads_l1 = pd.to_numeric(df["consumption_l1_kwh"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
        loads_l2 = pd.to_numeric(df["consumption_l2_kwh"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
        loads_l3 = pd.to_numeric(df["consumption_l3_kwh"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
    else:
        loads_l1 = loads_tot / 3.0
        loads_l2 = loads_tot / 3.0
        loads_l3 = loads_tot / 3.0

    pvs = pd.to_numeric(df["production_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    has_telemetry_series = df["has_telemetry"] if "has_telemetry" in df.columns else pd.Series(True, index=df.index)
    has_telemetry_all = has_telemetry_series.fillna(False).astype(bool).to_numpy()

    log_rows = []
    t_start = time.time()
    metering_str = "Asymmetric (3-phase)" if cfg.asymmetric_metering else "Symmetric"
    print(f"Starting Optimizer Arbitrage backtest (B2G Commercial MPC, {metering_str}, SoC {cfg.soc_min*100:.0f}%-{cfg.soc_max*100:.0f}%)...")

    for idx in range(steps):
        has_tel = bool(has_telemetry_all[idx]) and pd.notna(df["consumption_kwh"].iloc[idx] if "consumption_kwh" in df.columns else 0.0)
        actual_spot = float(spot_prices[idx])
        p_buy_act = float(p_buy_all[idx])
        p_sell_act = float(p_sell_all[idx])

        if not has_tel:
            # Missing telemetry hour: do not charge/discharge or trade for house, ignore in calculations
            log_rows.append(
                {
                    "datetime": df["datetime"].iloc[idx],
                    "has_telemetry": False,
                    "spot_price_czk": actual_spot,
                    "p_buy": p_buy_act,
                    "p_sell": p_sell_act,
                    "pv_kwh": np.nan,
                    "load_kwh": np.nan,
                    "load_l1_kwh": np.nan,
                    "load_l2_kwh": np.nan,
                    "load_l3_kwh": np.nan,
                    "charge_kwh": np.nan,
                    "discharge_kwh": np.nan,
                    "soc_kwh": cur_bat_soc,
                    "buy_kwh": np.nan,
                    "sell_kwh": np.nan,
                    "curtailment_kwh": np.nan,
                    "cost_buy_czk": np.nan,
                    "revenue_sell_czk": np.nan,
                    "degradation_cost_czk": np.nan,
                    "net_profit_czk": np.nan,
                }
            )
            continue

        actual_l1 = float(loads_l1[idx])
        actual_l2 = float(loads_l2[idx])
        actual_l3 = float(loads_l3[idx])
        actual_load = actual_l1 + actual_l2 + actual_l3
        actual_pv = float(pvs[idx])

        # 1. Update ML Forecaster models once daily
        hist_start = max(0, idx - history_len)
        hist_df = df.iloc[hist_start:idx] if idx >= 24 else None
        engine.maybe_retrain(hist_df, idx)

        # 2. Window for MPC (up to 24 hours ahead)
        h_len = min(cfg.mpc_horizon_hours, steps - idx)
        window_df = df.iloc[idx : idx + h_len].copy()

        # 3. Solve 24h Rolling Horizon LP
        planned_ch, planned_dis, planned_sell, planned_grid_charge, reserved_soc = solve_mpc_step(
            cur_bat_soc, window_df, cfg, engine
        )

        # 4. Closed-loop physical dispatch execution (Hierarchical Battery First with Arbitrage Overrides)
        max_possible_ch = max(0.0, (max_bat_soc - cur_bat_soc) / cfg.charge_efficiency)
        max_possible_dis = max(0.0, (cur_bat_soc - min_bat_soc) * cfg.discharge_efficiency)

        actual_solar_surplus = max(0.0, actual_pv - actual_load)
        net_needed = max(0.0, actual_load - actual_pv)

        if planned_grid_charge > 1e-4:
            # Regime 1: DELIBERATE GRID CHARGE (LP chose to buy cheap power into battery)
            actual_dis = 0.0
            allowed_ch = actual_solar_surplus + planned_grid_charge
            actual_ch = min(allowed_ch, max_possible_ch, cfg.battery_power_max_kw)

        elif planned_sell > 1e-4:
            # Regime 2: COMMERCIAL B2G EXPORT (LP chose to sell battery to grid)
            actual_ch = 0.0
            hurdle_rate = (wacse / max(1e-4, cfg.discharge_efficiency)) + cfg.degradation_cost_czk_per_kwh
            if p_sell_act < hurdle_rate:
                # Sell price does not cover cost of stored energy + degradation; clamp to self-consumption
                actual_dis = min(net_needed, max_possible_dis, cfg.battery_power_max_kw)
            else:
                actual_dis = min(planned_dis, max_possible_dis, cfg.battery_power_max_kw)

        else:
            # Regime 3: AUTONOMOUS BATTERY FIRST (Normal self-consumption)
            if actual_solar_surplus > 0:
                # PV covers house load, surplus charges battery
                actual_dis = 0.0
                actual_ch = min(actual_solar_surplus, max_possible_ch, cfg.battery_power_max_kw)
            else:
                # House has deficit: battery covers it, zero grid charge
                actual_ch = 0.0
                if reserved_soc > 1e-4:
                    safe_avail_soc = max(0.0, cur_bat_soc - (min_bat_soc + reserved_soc))
                    max_dis_allowed = safe_avail_soc * cfg.discharge_efficiency
                    actual_dis = min(net_needed, max_dis_allowed, cfg.battery_power_max_kw)
                else:
                    actual_dis = min(net_needed, max_possible_dis, cfg.battery_power_max_kw)

        # Update battery state of charge & WACSE
        delta_soc = actual_ch * cfg.charge_efficiency - actual_dis / max(1e-4, cfg.discharge_efficiency)
        new_soc = float(np.clip(cur_bat_soc + delta_soc, min_bat_soc, max_bat_soc))

        if actual_ch > 1e-4:
            pv_to_ch = min(actual_ch, max(0.0, actual_pv - actual_load))
            grid_to_ch = max(0.0, actual_ch - pv_to_ch)
            cost_of_new = pv_to_ch * 0.0 + grid_to_ch * p_buy_act
            wacse = (cur_bat_soc * wacse + cost_of_new) / max(0.1, cur_bat_soc + actual_ch * cfg.charge_efficiency)

        cur_bat_soc = new_soc

        # 5. Grid Exchange & Metering (Asymmetric vs Symmetric)
        curt_act = 0.0

        if cfg.asymmetric_metering:
            net_imbalance = actual_load + actual_ch - actual_pv - actual_dis
            buy_act = max(0.0, net_imbalance)
            sell_act = max(0.0, -net_imbalance)

            # Inverter grid limit curtailment
            if sell_act > cfg.grid_limit_kw and cfg.allow_curtailment:
                curt_act += (sell_act - cfg.grid_limit_kw)
                sell_act = cfg.grid_limit_kw

            # Zero Export on negative sell price
            if p_sell_act < 0 and cfg.allow_curtailment:
                curt_act += sell_act
                sell_act = 0.0
        else:
            p_inv_ac = actual_pv + actual_dis - actual_ch
            p_inv_phase = p_inv_ac / 3.0
            phase_loads = [actual_l1, actual_l2, actual_l3]

            phase_buys = [max(0.0, lp - p_inv_phase) for lp in phase_loads]
            phase_sells = [max(0.0, p_inv_phase - lp) for lp in phase_loads]

            sell_tot = sum(phase_sells)
            if sell_tot > cfg.grid_limit_kw and cfg.allow_curtailment:
                curt_act += (sell_tot - cfg.grid_limit_kw)
                sell_tot = cfg.grid_limit_kw

            if p_sell_act < 0 and cfg.allow_curtailment:
                curt_act += sell_tot
                sell_tot = 0.0

            buy_act = sum(phase_buys)
            sell_act = sell_tot

        cost_buy = buy_act * p_buy_act
        rev_sell = sell_act * p_sell_act
        deg_cost = cfg.degradation_cost_czk_per_kwh * (actual_ch + actual_dis)
        profit = rev_sell - cost_buy - deg_cost

        log_rows.append(
            {
                "datetime": df["datetime"].iloc[idx],
                "has_telemetry": True,
                "spot_price_czk": actual_spot,
                "p_buy": p_buy_act,
                "p_sell": p_sell_act,
                "pv_kwh": actual_pv,
                "load_kwh": actual_load,
                "load_l1_kwh": actual_l1,
                "load_l2_kwh": actual_l2,
                "load_l3_kwh": actual_l3,
                "charge_kwh": actual_ch,
                "discharge_kwh": actual_dis,
                "soc_kwh": cur_bat_soc,
                "buy_kwh": buy_act,
                "sell_kwh": sell_act,
                "curtailment_kwh": curt_act,
                "cost_buy_czk": cost_buy,
                "revenue_sell_czk": rev_sell,
                "degradation_cost_czk": deg_cost,
                "net_profit_czk": profit,
            }
        )

    duration = time.time() - t_start
    ms_per_step = (duration / max(1, steps)) * 1000.0
    print(f"Simulation completed in {duration:.2f} seconds ({ms_per_step:.1f} ms/step).")

    log_df = pd.DataFrame(log_rows)
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(out_p, sep=";", index=False)
        print(f"Trading log saved to {out_p}")

    return log_df


def compute_arbitrage_kpis(df: pd.DataFrame) -> dict:
    """Calculate detailed financial, grid, and battery performance metrics."""
    if df.empty:
        return {}

    valid_df = df[df["has_telemetry"] == True] if "has_telemetry" in df.columns else df

    total_buy_kwh = float(valid_df["buy_kwh"].dropna().sum())
    total_sell_kwh = float(valid_df["sell_kwh"].dropna().sum())
    total_pv_kwh = float(valid_df["pv_kwh"].dropna().sum())
    total_load_kwh = float(valid_df["load_kwh"].dropna().sum())
    total_ch_kwh = float(valid_df["charge_kwh"].dropna().sum())
    total_dis_kwh = float(valid_df["discharge_kwh"].dropna().sum())
    total_curt_kwh = float(valid_df["curtailment_kwh"].dropna().sum())

    total_cost_czk = float(valid_df["cost_buy_czk"].dropna().sum())
    total_rev_czk = float(valid_df["revenue_sell_czk"].dropna().sum())
    total_deg_czk = float(valid_df["degradation_cost_czk"].dropna().sum())
    net_profit_czk = float(valid_df["net_profit_czk"].dropna().sum())

    total_cycles = total_dis_kwh / 9.5

    return {
        "steps": len(valid_df),
        "total_steps": len(df),
        "missing_steps": len(df) - len(valid_df),
        "total_load_kwh": total_load_kwh,
        "total_pv_kwh": total_pv_kwh,
        "total_buy_kwh": total_buy_kwh,
        "total_sell_kwh": total_sell_kwh,
        "total_ch_kwh": total_ch_kwh,
        "total_dis_kwh": total_dis_kwh,
        "total_curt_kwh": total_curt_kwh,
        "total_cycles": total_cycles,
        "total_cost_czk": total_cost_czk,
        "total_rev_czk": total_rev_czk,
        "total_deg_czk": total_deg_czk,
        "net_profit_czk": net_profit_czk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SolarEnergyOptimizer Arbitrage - Commercial B2G MPC")
    parser.add_argument("--input", type=str, default="data/combined_market_data.csv", help="Input dataset path")
    parser.add_argument("--output", type=str, default="trading_log_arbitrage.csv", help="Output trading log")
    parser.add_argument("--month", type=str, default=None, help="Filter by month (e.g. '1', 'january', '2026-01')")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-steps", type=int, default=None, help="Backtest steps in hours")
    parser.add_argument("--battery-power", type=float, default=3.6, help="Max battery power in kW")
    parser.add_argument("--model-type", type=str, default="lgbm", choices=["lgbm", "hgb", "linear"])
    args = parser.parse_args()

    cfg = OptimizerArbitrageConfig(
        battery_power_max_kw=args.battery_power,
        model_type=args.model_type,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, sep=";")
    print(f"Loaded {len(df)} rows from {input_path}")

    if args.month or args.start_date:
        df = filter_dataset_by_date_or_month(df, month=args.month, start_date=args.start_date, end_date=args.end_date)
        print(f"Filtered to {len(df)} rows: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    out_p = Path(args.output) if args.output else None
    log_df = run_arbitrage_backtest(df, cfg=cfg, max_steps=args.max_steps, output_path=out_p)
    kpis = compute_arbitrage_kpis(log_df)

    print("\n=============================================================")
    print("        OPTIMIZER ARBITRAGE (B2G) BENCHMARK RESULTS          ")
    print("=============================================================")
    print(f"  Simulation Steps:        {kpis['steps']} hours")
    print(f"  Total Home Consumption:  {kpis['total_load_kwh']:.2f} kWh")
    print(f"  Total Solar Generation:  {kpis['total_pv_kwh']:.2f} kWh")
    print(f"  Curtailed Solar (Curtail):{kpis['total_curt_kwh']:.2f} kWh")
    print(f"  Grid Purchase:           {kpis['total_buy_kwh']:.2f} kWh  ({kpis['total_cost_czk']:.2f} CZK)")
    print(f"  Grid Export (Sold):      {kpis['total_sell_kwh']:.2f} kWh  ({kpis['total_rev_czk']:.2f} CZK)")
    print(f"  Battery Cycles:          {kpis['total_cycles']:.1f} cycles")
    print(f"  Battery Degradation Cost:{kpis['total_deg_czk']:.2f} CZK")
    print(f"  -----------------------------------------------------------")
    print(f"  NET FINANCIAL RESULT:    {kpis['net_profit_czk']:.2f} CZK")
    print("=============================================================\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
