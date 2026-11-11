"""Real-Time Linear Programming (LP) Microgrid Arbitrage Optimizer.

Ports the exact mathematical optimization formulation, constraints, and tariff models
from OptimizerArbitrage into the real-time operational pipeline:
- PuLP Linear Programming with CBC solver
- Czech distribution fees (+1.80 CZK/kWh) and 21% VAT on grid imports
- Trader export fee (-0.45 CZK/kWh) on grid exports
- True Battery-to-Grid (B2G) commercial arbitrage
- Terminal SoC valuation to protect stored energy
- Zero Grid Pre-Charge rule before sunny days
- Guard Solar Pre-Flush rule to prevent premature morning dumping
- Zero-Export curtailment enforcement on negative export prices
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
import pulp as pl

try:
    from real_time.config import (
        ALLOW_CURTAILMENT,
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        CzechCommand,
        CzechReason,
        DEGRADATION_COST_CZK_PER_KWH,
        EFFICIENCY_CHARGE,
        EFFICIENCY_DISCHARGE,
        EXPORT_FEE_CZK_PER_KWH,
        FORECAST_LOAD_SAFETY_FACTOR,
        FORECAST_PV_SAFETY_FACTOR,
        GRID_LIMIT_KW,
        IMPORT_FEE_CZK_PER_KWH,
        MPC_HORIZON_HOURS,
        PREFLUSH_MIN_SELL_PRICE_CZK,
        PREFLUSH_MIN_SURPLUS_KWH,
        SOC_MAX,
        SOC_MIN,
        VAT_RATE,
    )
except ImportError:
    from config import (
        ALLOW_CURTAILMENT,
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        CzechCommand,
        CzechReason,
        DEGRADATION_COST_CZK_PER_KWH,
        EFFICIENCY_CHARGE,
        EFFICIENCY_DISCHARGE,
        EXPORT_FEE_CZK_PER_KWH,
        FORECAST_LOAD_SAFETY_FACTOR,
        FORECAST_PV_SAFETY_FACTOR,
        GRID_LIMIT_KW,
        IMPORT_FEE_CZK_PER_KWH,
        MPC_HORIZON_HOURS,
        PREFLUSH_MIN_SELL_PRICE_CZK,
        PREFLUSH_MIN_SURPLUS_KWH,
        SOC_MAX,
        SOC_MIN,
        VAT_RATE,
    )

logger = logging.getLogger("LpOptimizerRT")


def compute_effective_buy_price(spot_price: float | np.ndarray) -> float | np.ndarray:
    """Effective import price from grid: (spot + 1.80 CZK) * (1 + 0.21 VAT)."""
    return (spot_price + IMPORT_FEE_CZK_PER_KWH) * (1.0 + VAT_RATE)


def compute_effective_sell_price(spot_price: float | np.ndarray) -> float | np.ndarray:
    """Effective export price to grid: spot - 0.45 CZK (trader fee deducted)."""
    return spot_price - EXPORT_FEE_CZK_PER_KWH


class ScheduleDict(dict):
    """Dictionary supporting lookup by (hour, minute_slot) tuple or by integer hour."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return super().__getitem__((key, 0))
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, int):
            return super().get((key, 0), default)
        return super().get(key, default)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, int):
            return super().__contains__((key, 0))
        return super().__contains__(key)


@dataclass
class PlannedHourAction:
    """Optimal directive for a specific slot of the MPC planning horizon."""

    hour: int
    command: str
    target_power_w: float
    reason: str
    planned_buy_kw: float
    planned_sell_kw: float
    planned_ch_kw: float
    planned_dis_kw: float
    planned_end_soc_kwh: float
    planned_end_soc_pct: float
    effective_buy_czk: float
    effective_sell_czk: float
    minute_slot: int = 0
    time_str: str = ""


def compute_adaptive_risk_metrics_rt(
    t: int,
    t_steps: int,
    spot_prices: np.ndarray,
    p_buy_pred: np.ndarray,
    p_sell_pred: np.ndarray,
    pv_pred: np.ndarray,
    load_pred: np.ndarray,
    cloud_cover: np.ndarray,
    discharge_eff: float,
    charge_eff: float,
    degradation_cost: float,
    dt_hours: float = 1.0,
) -> tuple[float, float]:
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
        net_spread = (peak_benefit * discharge_eff) - (p_buy_cur / charge_eff) - (2.0 * degradation_cost)
        c_u = max(0.0, net_spread)

        # Overage cost: loss of buying grid energy that gets displaced by solar to low-tariff export
        c_o = max(0.5, p_buy_cur - p_sell_cur + (2.0 * degradation_cost))

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
        cloud_entropy = 4.0 * cc * (1.0 - cc)

        pv_k = max(0.0, float(pv_pred[k]))
        load_k = max(0.0, float(load_pred[k]))
        surplus_k = max(0.0, pv_k - load_k) * dt_hours

        sigma_k = pv_k * (0.10 + 0.40 * cloud_entropy) * dt_hours
        future_surplus_mean += surplus_k
        future_surplus_var += (sigma_k ** 2)

    sigma_total = np.sqrt(future_surplus_var)

    # 3. Dynamic Quantile Headroom Reservation
    z_score = 1.65 * (1.0 - 1.8 * q_star)
    reserved_solar_kwh = max(0.0, future_surplus_mean + z_score * sigma_total)

    return q_star, reserved_solar_kwh


class LpArbitrageOptimizerRT:
    """Real-Time Linear Programming Arbitrage Optimizer with Adaptive Stochastic Risk."""

    def __init__(self) -> None:
        self.capacity_kwh = BATTERY_CAPACITY_KWH
        self.soc_min_kwh = SOC_MIN * BATTERY_CAPACITY_KWH
        self.soc_max_kwh = SOC_MAX * BATTERY_CAPACITY_KWH
        self.power_max_kw = BATTERY_POWER_MAX_KW
        self.charge_eff = EFFICIENCY_CHARGE
        self.discharge_eff = EFFICIENCY_DISCHARGE
        self.inv_discharge_eff = 1.0 / max(1e-4, EFFICIENCY_DISCHARGE)
        self.degradation_cost = DEGRADATION_COST_CZK_PER_KWH
        self.grid_limit = GRID_LIMIT_KW

    def solve_horizon(
        self,
        current_soc_kwh: float,
        spot_prices: list[float] | np.ndarray,
        pv_forecast_kw: list[float] | np.ndarray,
        load_forecast_kw: list[float] | np.ndarray,
        hours_of_day: list[int] | np.ndarray,
        cloud_cover_pct: list[float] | np.ndarray | None = None,
        dt_hours: float = 1.0,
        minute_slots: list[int] | None = None,
    ) -> dict[Any, PlannedHourAction]:
        """Solve rolling horizon LP with adaptive risk and return optimal schedule (15-min or 1-hour)."""
        max_steps = int(MPC_HORIZON_HOURS / dt_hours) if dt_hours > 0 else 24
        t_steps = min(len(spot_prices), len(pv_forecast_kw), len(load_forecast_kw), max_steps)
        if t_steps == 0:
            return {}

        spots = np.asarray(spot_prices[:t_steps], dtype=float)
        pvs = np.maximum(0.0, np.asarray(pv_forecast_kw[:t_steps], dtype=float))
        loads = np.maximum(0.0, np.asarray(load_forecast_kw[:t_steps], dtype=float))
        hours = np.asarray(hours_of_day[:t_steps], dtype=int)
        m_slots = np.asarray(minute_slots[:t_steps], dtype=int) if minute_slots is not None else np.zeros(t_steps, dtype=int)

        if cloud_cover_pct is not None and len(cloud_cover_pct) > 0:
            cloud_cover = np.asarray(cloud_cover_pct[:t_steps], dtype=float)
        else:
            cloud_cover = np.full(t_steps, 50.0)

        p_buy_pred = compute_effective_buy_price(spots)
        p_sell_pred = compute_effective_sell_price(spots)

        cur_soc = float(np.clip(current_soc_kwh, self.soc_min_kwh, self.soc_max_kwh))

        prob = pl.LpProblem("Microgrid_Arbitrage_RT_MPC_Adaptive", pl.LpMinimize)

        # Decision variables
        buy = [pl.LpVariable(f"b_{t}", lowBound=0.0, upBound=self.grid_limit) for t in range(t_steps)]
        sell = [pl.LpVariable(f"s_{t}", lowBound=0.0, upBound=self.grid_limit) for t in range(t_steps)]
        ch_pv = [pl.LpVariable(f"cpv_{t}", lowBound=0.0, upBound=self.power_max_kw) for t in range(t_steps)]
        ch_grid = [pl.LpVariable(f"cgrid_{t}", lowBound=0.0, upBound=self.power_max_kw) for t in range(t_steps)]
        ch = [pl.LpVariable(f"c_{t}", lowBound=0.0, upBound=self.power_max_kw) for t in range(t_steps)]
        dis = [pl.LpVariable(f"d_{t}", lowBound=0.0, upBound=self.power_max_kw) for t in range(t_steps)]
        soc = [pl.LpVariable(f"soc_{t}", lowBound=self.soc_min_kwh, upBound=self.soc_max_kwh) for t in range(t_steps)]
        curt = [pl.LpVariable(f"curt_{t}", lowBound=0.0, upBound=max(0.0, float(pvs[t]))) for t in range(t_steps)]

        # Terminal SoC valuation
        avoided_purchase_val = float(np.mean(p_buy_pred)) * self.discharge_eff - self.degradation_cost
        export_val = float(np.mean(p_sell_pred)) + self.degradation_cost
        terminal_val = max(0.5, min(avoided_purchase_val, export_val * 1.5))
        cost_terms = [-terminal_val * soc[t_steps - 1]]

        cum_grid_charge_expr = []

        for t in range(t_steps):
            cost_terms.append(buy[t] * float(p_buy_pred[t]) * dt_hours)
            cost_terms.append(-sell[t] * float(p_sell_pred[t]) * dt_hours)
            cost_terms.append(self.degradation_cost * (ch[t] + dis[t]) * dt_hours)
            cost_terms.append(0.005 * curt[t] * dt_hours)

            # Instantaneous balance: buy + pv + dis == load + sell + ch + curt
            prob += buy[t] + float(pvs[t]) + dis[t] == float(loads[t]) + sell[t] + ch[t] + curt[t]

            # Charging composition: solar surplus is free, grid charging is for arbitrage
            prob += ch[t] == ch_pv[t] + ch_grid[t]
            pv_surplus_t = max(0.0, float(pvs[t]) - float(loads[t]))
            prob += ch_pv[t] <= pv_surplus_t

            # Battery SoC continuity equation with exact dt_hours scaling
            prev = cur_soc if t == 0 else soc[t - 1]
            prob += soc[t] == prev + (self.charge_eff * ch[t] - dis[t] * self.inv_discharge_eff) * dt_hours

            # Zero Export when effective sell price is negative
            if p_sell_pred[t] < 0 and ALLOW_CURTAILMENT:
                prob += sell[t] == 0.0

            # ADAPTIVE RISK CONSTRAINT (Replaces all hardcoded clock-hour heuristics)
            q_star, reserved_solar = compute_adaptive_risk_metrics_rt(
                t, t_steps, spots, p_buy_pred, p_sell_pred, pvs, loads, cloud_cover,
                self.discharge_eff, self.charge_eff, self.degradation_cost, dt_hours=dt_hours
            )

            cum_grid_charge_expr.append(ch_grid[t] * dt_hours)

            if spots[t] < 0:
                # Negative price: unlimited grid charge (grid pays us)
                pass
            elif q_star <= 0.0:
                # No profitable arbitrage: strictly zero grid charging
                prob += ch_grid[t] == 0.0
            else:
                # Profitable arbitrage exists: cumulative grid pre-charge must leave space for solar
                max_grid_headroom = max(0.0, (self.soc_max_kwh - reserved_solar - cur_soc) / self.charge_eff)
                prob += pl.lpSum(cum_grid_charge_expr) <= max_grid_headroom

        prob += pl.lpSum(cost_terms)

        # Solve LP using CBC solver without console spam
        prob.solve(pl.PULP_CBC_CMD(msg=False))

        schedule = ScheduleDict()
        if pl.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            logger.warning("PuLP LP solver did not find optimal solution (status %s). Using fallback.", pl.LpStatus[prob.status])
            return schedule

        for t in range(t_steps):
            h = int(hours[t])
            m = int(m_slots[t]) if minute_slots is not None else 0

            val_ch = max(0.0, float(pl.value(ch[t])) if pl.value(ch[t]) is not None else 0.0)
            val_ch_grid = max(0.0, float(pl.value(ch_grid[t])) if pl.value(ch_grid[t]) is not None else 0.0)
            val_dis = max(0.0, float(pl.value(dis[t])) if pl.value(dis[t]) is not None else 0.0)
            val_buy = max(0.0, float(pl.value(buy[t])) if pl.value(buy[t]) is not None else 0.0)
            val_sell = max(0.0, float(pl.value(sell[t])) if pl.value(sell[t]) is not None else 0.0)
            val_soc = max(self.soc_min_kwh, float(pl.value(soc[t])) if pl.value(soc[t]) is not None else cur_soc)
            soc_pct = (val_soc / self.capacity_kwh) * 100.0

            # Determine Czech inverter action
            cmd = CzechCommand.STANDBY
            reason = CzechReason.STANDBY_SCHEDULE
            target_power_w = 0.0

            if val_ch > 0.05:
                target_power_w = round(val_ch * 1000.0, 0)
                # Did LP plan grid import charging?
                if val_ch_grid > 0.05:
                    cmd = CzechCommand.CHARGE_GRID
                    if spots[t] < 0:
                        reason = CzechReason.LOW_PRICE
                    else:
                        reason = "Arbitrážní přednabití (výhodný nákup)"
                else:
                    cmd = CzechCommand.CHARGE_PV
                    reason = CzechReason.PV_SURPLUS

            elif val_dis > 0.05:
                target_power_w = round(val_dis * 1000.0, 0)
                # Did LP plan commercial B2G export?
                if val_sell > float(pvs[t]) + 0.1:
                    cmd = CzechCommand.DISCHARGE_GRID
                    reason = CzechReason.HIGH_PRICE
                else:
                    cmd = CzechCommand.COVER_HOME
                    reason = CzechReason.HOME_SAVING

            start_min = m * 15 if dt_hours == 0.25 else 0
            end_min = start_min + 15 if dt_hours == 0.25 else 0
            slot_str = f"{h:02d}:{start_min:02d} - {h:02d}:{end_min:02d}" if dt_hours == 0.25 else f"{h:02d}:00"

            action = PlannedHourAction(
                hour=h,
                command=cmd,
                target_power_w=target_power_w,
                reason=reason,
                planned_buy_kw=round(val_buy, 2),
                planned_sell_kw=round(val_sell, 2),
                planned_ch_kw=round(val_ch, 2),
                planned_dis_kw=round(val_dis, 2),
                planned_end_soc_kwh=round(val_soc, 3),
                planned_end_soc_pct=round(soc_pct, 1),
                effective_buy_czk=round(float(p_buy_pred[t]), 3),
                effective_sell_czk=round(float(p_sell_pred[t]), 3),
                minute_slot=m,
                time_str=slot_str,
            )

            if dt_hours < 0.5:
                # 15-minute resolution: key is (hour, minute_slot)
                if (h, m) not in schedule:
                    schedule[(h, m)] = action
            else:
                # Hourly resolution: key is hour
                if h not in schedule:
                    schedule[h] = action

        return schedule
