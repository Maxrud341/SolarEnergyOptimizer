"""Optimizer Arbitrage V2 - Adaptive Risk-Aware B2G Microgrid EMS.

Implements true commercial spot market arbitrage with an Adaptive Stochastic Risk Architecture:
1. Dynamic Market Risk Engine (Newsvendor Critical Fractile q*):
   - Derives willingness to pre-charge from economic asymmetry between peak avoidance
     (underage cost) and displaced solar opportunity loss (overage cost).
   - Eliminates all hardcoded clock-hour restrictions (e.g. 6-21, 0-5, 4-9).
2. Atmospheric & Solar Uncertainty Engine (Cloud Entropy σ_pv):
   - Computes non-linear meteorological forecast variance from cloud cover percentage:
     cloud_entropy = 4 * cc * (1 - cc).
3. Dynamic Quantile Solar Headroom Reservation (R_solar):
   - Reserves battery capacity for free solar based on adaptive z-score quantile:
     R_solar = max(0, E[surplus] + z(q*) * σ_surplus).
4. Decomposed Charge Modeling (ch_pv vs ch_grid):
   - Zero-cost unconstrained charging from local solar surplus up to 100% SoC.
   - Grid charging allowed only when q* > 0, bounded by (max_soc - R_solar).
5. Closed-loop 24h Rolling Horizon MPC with Czech per-phase (Fázové řízení) metering.
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
class OptimizerArbitrageV2Config:
    battery_capacity_kwh: float = 10.24  # Fox-ESS HV2600 HS10.4 nominal capacity (10.24 kWh)
    soc_min: float = 0.20  # 20% minimum discharge limit
    soc_max: float = 1.00  # 100% full capacity
    battery_power_max_kw: float = 5.0  # Inverter max power in kW
    charge_efficiency: float = 0.975  # Fox-ESS round-trip efficiency > 95% (sqrt(0.95) ≈ 0.975)
    discharge_efficiency: float = 0.975
    degradation_cost_czk_per_kwh: float = 0.20  # LFP cell warranty >=6000 cycles base cost
    grid_limit_kw: float = 10.0
    import_fee_czk_per_kwh: float = 1.80  # Czech distribution grid fee
    vat_rate: float = 0.21  # Czech VAT
    export_fee_czk_per_kwh: float = 0.45  # Trader export fee
    allow_curtailment: bool = True
    asymmetric_metering: bool = True
    mpc_horizon_hours: int = 24
    model_type: str = "lgbm"


CONFIG_ARBITRAGE_V2 = OptimizerArbitrageV2Config()


def compute_adaptive_risk_metrics_v2(
    t: int,
    t_steps: int,
    spot_prices: np.ndarray,
    p_buy_pred: np.ndarray,
    p_sell_pred: np.ndarray,
    pv_pred: np.ndarray,
    load_pred: np.ndarray,
    cloud_cover: np.ndarray,
    cfg: OptimizerArbitrageV2Config,
) -> Tuple[float, float]:
    """Compute (q_star, reserved_solar_kwh) using Newsvendor quantile optimization and cloud entropy."""
    p_buy_cur = float(p_buy_pred[t])
    p_sell_cur = float(p_sell_pred[t])

    # 1. Economic Risk: Future Peak Opportunity vs Displaced Solar Cost (Newsvendor Fractile)
    future_steps = range(t + 1, t_steps)
    if not future_steps or spot_prices[t] < 0:
        q_star = 1.0 if spot_prices[t] < 0 else 0.0
    else:
        peak_benefit = 0.0
        for f_t in future_steps:
            benefit_f = max(float(p_buy_pred[f_t]), float(p_sell_pred[f_t]))
            if benefit_f > peak_benefit:
                peak_benefit = benefit_f

        # Underage cost: financial loss of having an empty battery during peak
        net_spread = (peak_benefit * cfg.discharge_efficiency) - (p_buy_cur / cfg.charge_efficiency) - (2.0 * cfg.degradation_cost_czk_per_kwh)
        c_u = max(0.0, net_spread)

        # Overage cost: loss of buying grid energy that gets displaced by solar to low-tariff export
        c_o = max(0.5, p_buy_cur - p_sell_cur + (2.0 * cfg.degradation_cost_czk_per_kwh))

        if c_u <= 1e-4:
            q_star = 0.0
        else:
            q_star = float(c_u / (c_u + c_o))
            q_star = min(1.0, max(0.0, q_star))

    # 2. Weather Uncertainty & Cloud Entropy
    future_surplus_mean = 0.0
    future_surplus_var = 0.0
    for k in range(t, t_steps):
        cc = float(cloud_cover[k]) / 100.0 if k < len(cloud_cover) else 0.5
        cc = min(1.0, max(0.0, cc))
        # Cloud entropy is maximal at 50% broken clouds, minimal at 0% or 100%
        cloud_entropy = 4.0 * cc * (1.0 - cc)

        pv_k = max(0.0, float(pv_pred[k]))
        load_k = max(0.0, float(load_pred[k]))
        surplus_k = max(0.0, pv_k - load_k)

        # Standard deviation: 10% base + up to 40% from cloud entropy
        sigma_k = pv_k * (0.10 + 0.40 * cloud_entropy)
        future_surplus_mean += surplus_k
        future_surplus_var += (sigma_k ** 2)

    sigma_total = np.sqrt(future_surplus_var)

    # 3. Dynamic Quantile Headroom Reservation
    # If q_star is high (huge spread), we take risk -> lower z-score
    # If q_star is low (tight spread), we protect solar -> higher z-score
    z_score = 1.65 * (1.0 - 1.8 * q_star)
    reserved_solar_kwh = max(0.0, future_surplus_mean + z_score * sigma_total)

    return q_star, reserved_solar_kwh


def solve_mpc_step_v2(
    current_soc_kwh: float,
    window_df: pd.DataFrame,
    cfg: OptimizerArbitrageV2Config,
    engine: ForecasterEngineV3,
) -> Tuple[float, float, float, float, float]:
    """Solve 24h rolling horizon LP with adaptive risk constraints and decomposed charging."""
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

    if "cloud_cover_pct" in window_df.columns:
        cloud_cover = pd.to_numeric(window_df["cloud_cover_pct"], errors="coerce").fillna(50.0).to_numpy(dtype=float)
    else:
        cloud_cover = np.full(t_steps, 50.0)

    min_soc = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc = cfg.soc_max * cfg.battery_capacity_kwh
    cur_soc = float(np.clip(current_soc_kwh, min_soc, max_soc))

    prob = pl.LpProblem("Microgrid_Arbitrage_MPC_V2", pl.LpMinimize)

    # Decision variables
    buy = [pl.LpVariable(f"b_{t}", lowBound=0.0, upBound=cfg.grid_limit_kw) for t in range(t_steps)]
    sell = [pl.LpVariable(f"s_{t}", lowBound=0.0, upBound=cfg.grid_limit_kw) for t in range(t_steps)]
    ch_pv = [pl.LpVariable(f"cpv_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    ch_grid = [pl.LpVariable(f"cgrid_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    ch = [pl.LpVariable(f"c_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    dis = [pl.LpVariable(f"d_{t}", lowBound=0.0, upBound=cfg.battery_power_max_kw) for t in range(t_steps)]
    soc = [pl.LpVariable(f"soc_{t}", lowBound=min_soc, upBound=max_soc) for t in range(t_steps)]
    curt = [pl.LpVariable(f"curt_{t}", lowBound=0.0, upBound=max(0.0, float(pv_pred[t]))) for t in range(t_steps)]

    # Terminal SoC valuation
    avoided_purchase_val = float(np.mean(p_buy_pred)) * cfg.discharge_efficiency - cfg.degradation_cost_czk_per_kwh
    export_val = float(np.mean(p_sell_pred)) + cfg.degradation_cost_czk_per_kwh
    terminal_val = max(0.5, min(avoided_purchase_val, export_val * 1.5))
    cost_terms = [-terminal_val * soc[t_steps - 1]]

    inv_dis_eff = 1.0 / max(1e-4, cfg.discharge_efficiency)
    cum_grid_charge_expr = []

    for t in range(t_steps):
        cost_terms.append(buy[t] * float(p_buy_pred[t]))
        cost_terms.append(-sell[t] * float(p_sell_pred[t]))
        cost_terms.append(cfg.degradation_cost_czk_per_kwh * (ch[t] + dis[t]))
        cost_terms.append(0.005 * curt[t])

        # Instantaneous energy balance: buy + pv + dis == load + sell + ch + curt
        prob += buy[t] + float(pv_pred[t]) + dis[t] == float(load_pred[t]) + sell[t] + ch[t] + curt[t]

        # Charging composition
        prob += ch[t] == ch_pv[t] + ch_grid[t]
        pv_surplus_t = max(0.0, float(pv_pred[t]) - float(load_pred[t]))
        prob += ch_pv[t] <= pv_surplus_t

        # Battery SoC continuity equation
        prev = cur_soc if t == 0 else soc[t - 1]
        prob += soc[t] == prev + cfg.charge_efficiency * ch[t] - dis[t] * inv_dis_eff

        # Zero Export restriction when effective sell price is negative
        if p_sell_pred[t] < 0 and cfg.allow_curtailment:
            prob += sell[t] == 0.0

        # ADAPTIVE RISK CONSTRAINT
        q_star, reserved_solar = compute_adaptive_risk_metrics_v2(
            t, t_steps, spot_prices, p_buy_pred, p_sell_pred, pv_pred, load_pred, cloud_cover, cfg
        )

        cum_grid_charge_expr.append(ch_grid[t])

        if spot_prices[t] < 0:
            # Negative spot price: unlimited grid charging
            pass
        elif q_star <= 0.0:
            # No profitable arbitrage: strictly zero grid charging
            prob += ch_grid[t] == 0.0
        else:
            # Profitable arbitrage exists: cumulative grid pre-charge must respect dynamic solar reserve
            max_grid_headroom = max(0.0, (max_soc - reserved_solar - cur_soc) / cfg.charge_efficiency)
            prob += pl.lpSum(cum_grid_charge_expr) <= max_grid_headroom

    prob += pl.lpSum(cost_terms)

    # Solve LP (CBC solver without message)
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    if pl.LpStatus[prob.status] not in ("Optimal", "Feasible"):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    first_ch = max(0.0, float(pl.value(ch[0])) if pl.value(ch[0]) is not None else 0.0)
    first_ch_grid = max(0.0, float(pl.value(ch_grid[0])) if pl.value(ch_grid[0]) is not None else 0.0)
    first_dis = max(0.0, float(pl.value(dis[0])) if pl.value(dis[0]) is not None else 0.0)
    first_sell = max(0.0, float(pl.value(sell[0])) if pl.value(sell[0]) is not None else 0.0)

    planned_grid_charge = first_ch_grid

    # Check future peak export reservation
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


def run_arbitrage_backtest_v2(
    df: pd.DataFrame,
    cfg: OptimizerArbitrageV2Config = CONFIG_ARBITRAGE_V2,
    max_steps: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute closed-loop 24h rolling horizon MPC backtest with V2 Adaptive Risk architecture."""
    steps = len(df) if max_steps is None else min(len(df), max_steps)
    if steps == 0:
        return pd.DataFrame()

    engine = ForecasterEngineV3(model_type=cfg.model_type)
    history_len = 168

    min_bat_soc = cfg.battery_capacity_kwh * cfg.soc_min
    max_bat_soc = cfg.battery_capacity_kwh * cfg.soc_max
    cur_bat_soc = cfg.battery_capacity_kwh * 0.5
    wacse = 2.5

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
    print(f"Starting Optimizer Arbitrage V2 backtest (Adaptive Risk MPC, {metering_str}, SoC {cfg.soc_min*100:.0f}%-{cfg.soc_max*100:.0f}%)...")

    for idx in range(steps):
        has_tel = bool(has_telemetry_all[idx]) and pd.notna(df["consumption_kwh"].iloc[idx] if "consumption_kwh" in df.columns else 0.0)
        actual_spot = float(spot_prices[idx])
        p_buy_act = float(p_buy_all[idx])
        p_sell_act = float(p_sell_all[idx])

        if not has_tel:
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

        # 3. Solve 24h Rolling Horizon LP with V2 Adaptive Risk
        planned_ch, planned_dis, planned_sell, planned_grid_charge, reserved_soc = solve_mpc_step_v2(
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
                actual_dis = min(net_needed, max_possible_dis, cfg.battery_power_max_kw)
            else:
                actual_dis = min(planned_dis, max_possible_dis, cfg.battery_power_max_kw)

        else:
            # Regime 3: AUTONOMOUS BATTERY FIRST (Normal self-consumption)
            if actual_solar_surplus > 0:
                actual_dis = 0.0
                actual_ch = min(actual_solar_surplus, max_possible_ch, cfg.battery_power_max_kw)
            else:
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


def compute_arbitrage_kpis_v2(df: pd.DataFrame) -> dict:
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

    usable_capacity_kwh = 10.24 * (1.00 - 0.20)
    total_cycles = total_dis_kwh / usable_capacity_kwh

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
    parser = argparse.ArgumentParser(description="SolarEnergyOptimizer Arbitrage V2 - Adaptive Risk-Aware MPC")
    parser.add_argument("--input", type=str, default="data/combined_market_data.csv", help="Input dataset path")
    parser.add_argument("--output", type=str, default="trading_log_arbitrage_v2.csv", help="Output trading log")
    parser.add_argument("--month", type=str, default=None, help="Filter by month (e.g. '1', 'january', '2026-01')")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-steps", type=int, default=None, help="Backtest steps in hours")
    parser.add_argument("--battery-capacity", type=float, default=10.24, help="Battery capacity in kWh")
    parser.add_argument("--battery-power", type=float, default=5.0, help="Max battery power in kW")
    parser.add_argument("--soc-min", type=float, default=0.20, help="Min SoC discharge floor")
    parser.add_argument("--model-type", type=str, default="lgbm", choices=["lgbm", "hgb", "linear"])
    args = parser.parse_args()

    cfg = OptimizerArbitrageV2Config(
        battery_capacity_kwh=args.battery_capacity,
        soc_min=args.soc_min,
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
    log_df = run_arbitrage_backtest_v2(df, cfg=cfg, max_steps=args.max_steps, output_path=out_p)
    kpis = compute_arbitrage_kpis_v2(log_df)

    print("\n=============================================================")
    print("      OPTIMIZER ARBITRAGE V2 (ADAPTIVE RISK) RESULTS         ")
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
