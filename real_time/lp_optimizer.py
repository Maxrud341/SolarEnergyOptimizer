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


@dataclass
class PlannedHourAction:
    """Optimal directive for a specific hour of the MPC planning horizon."""

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


class LpArbitrageOptimizerRT:
    """Real-Time Linear Programming Arbitrage Optimizer."""

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
    ) -> dict[int, PlannedHourAction]:
        """Solve 24h rolling horizon LP and return optimal hourly schedule."""
        t_steps = min(len(spot_prices), len(pv_forecast_kw), len(load_forecast_kw), MPC_HORIZON_HOURS)
        if t_steps == 0:
            return {}

        spots = np.asarray(spot_prices[:t_steps], dtype=float)
        pvs = np.maximum(0.0, np.asarray(pv_forecast_kw[:t_steps], dtype=float))
        loads = np.maximum(0.0, np.asarray(load_forecast_kw[:t_steps], dtype=float))
        hours = np.asarray(hours_of_day[:t_steps], dtype=int)

        p_buy_pred = compute_effective_buy_price(spots)
        p_sell_pred = compute_effective_sell_price(spots)

        cur_soc = float(np.clip(current_soc_kwh, self.soc_min_kwh, self.soc_max_kwh))
        headroom = self.soc_max_kwh - cur_soc

        # Conservative forecasts with safety buffers
        pv_safe = pvs * FORECAST_PV_SAFETY_FACTOR
        load_safe = loads * FORECAST_LOAD_SAFETY_FACTOR
        solar_surplus_safe = float(np.sum(np.maximum(0.0, pv_safe - load_safe)))

        # Sunny day detection: conservative surplus ample to fill battery headroom
        is_sunny_day = solar_surplus_safe >= max(1.5, headroom * 0.85)

        # Peak solar hour in upcoming daylight period
        daylight_indices = [t for t in range(t_steps) if 6 <= hours[t] <= 18 and pvs[t] > 0.5]
        first_peak_idx = daylight_indices[int(np.argmax([pvs[t] for t in daylight_indices]))] if daylight_indices else -1

        # Curtailment risk & negative prices
        curtailment_risk = max(0.0, solar_surplus_safe - headroom)
        has_negative_prices = any(p_sell_pred[t] < 0 for t in range(t_steps))

        prob = pl.LpProblem("Microgrid_Arbitrage_RT_MPC", pl.LpMinimize)

        # Decision variables
        buy = [pl.LpVariable(f"b_{t}", lowBound=0.0, upBound=self.grid_limit) for t in range(t_steps)]
        sell = [pl.LpVariable(f"s_{t}", lowBound=0.0, upBound=self.grid_limit) for t in range(t_steps)]
        ch = [pl.LpVariable(f"c_{t}", lowBound=0.0, upBound=self.power_max_kw) for t in range(t_steps)]
        dis = [pl.LpVariable(f"d_{t}", lowBound=0.0, upBound=self.power_max_kw) for t in range(t_steps)]
        soc = [pl.LpVariable(f"soc_{t}", lowBound=self.soc_min_kwh, upBound=self.soc_max_kwh) for t in range(t_steps)]
        curt = [pl.LpVariable(f"curt_{t}", lowBound=0.0, upBound=max(0.0, float(pvs[t]))) for t in range(t_steps)]

        # Terminal SoC valuation
        avoided_purchase_val = float(np.mean(p_buy_pred)) * self.discharge_eff - self.degradation_cost
        export_val = float(np.mean(p_sell_pred)) + self.degradation_cost
        terminal_val = max(0.5, min(avoided_purchase_val, export_val * 1.5))
        cost_terms = [-terminal_val * soc[t_steps - 1]]

        for t in range(t_steps):
            cost_terms.append(buy[t] * float(p_buy_pred[t]))
            cost_terms.append(-sell[t] * float(p_sell_pred[t]))
            cost_terms.append(self.degradation_cost * (ch[t] + dis[t]))
            cost_terms.append(0.005 * curt[t])

            # Instantaneous balance: buy + pv + dis == load + sell + ch + curt
            prob += buy[t] + float(pvs[t]) + dis[t] == float(loads[t]) + sell[t] + ch[t] + curt[t]

            # Battery SoC continuity equation
            prev = cur_soc if t == 0 else soc[t - 1]
            prob += soc[t] == prev + self.charge_eff * ch[t] - dis[t] * self.inv_discharge_eff

            # Zero Export when effective sell price is negative
            if p_sell_pred[t] < 0 and ALLOW_CURTAILMENT:
                prob += sell[t] == 0.0

            # Daytime grid charging restriction: with positive spot prices, charge only from PV surplus
            if (6 <= hours[t] <= 21) and (spots[t] >= 0):
                surplus_pv = max(0.0, float(pvs[t]) - float(loads[t]))
                prob += ch[t] <= surplus_pv

            # Rule 1: Zero Grid Pre-Charge before a sunny day (00:00 - 05:00)
            if is_sunny_day and (0 <= hours[t] <= 5) and (first_peak_idx > t):
                prob += ch[t] <= float(pvs[t])

            # Rule 2: Guard Solar Pre-Flush - early morning (04:00 - 09:00)
            if (4 <= hours[t] <= 9) and (first_peak_idx > t):
                allow_preflush = (
                    (curtailment_risk >= PREFLUSH_MIN_SURPLUS_KWH)
                    and has_negative_prices
                    and (p_sell_pred[t] >= PREFLUSH_MIN_SELL_PRICE_CZK)
                )
                if not allow_preflush:
                    prob += sell[t] <= float(pvs[t])
                else:
                    prob += sell[t] <= float(pvs[t]) + curtailment_risk

        prob += pl.lpSum(cost_terms)

        # Solve LP using CBC solver without console spam
        prob.solve(pl.PULP_CBC_CMD(msg=False))

        schedule: dict[int, PlannedHourAction] = {}
        if pl.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            logger.warning("PuLP LP solver did not find optimal solution (status %s). Using fallback.", pl.LpStatus[prob.status])
            return schedule

        for t in range(t_steps):
            h = int(hours[t])
            val_ch = max(0.0, float(pl.value(ch[t])) if pl.value(ch[t]) is not None else 0.0)
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
                if val_buy > 0.1:
                    cmd = CzechCommand.CHARGE_GRID
                    reason = CzechReason.LOW_PRICE
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

            schedule[h] = PlannedHourAction(
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
            )

        return schedule
