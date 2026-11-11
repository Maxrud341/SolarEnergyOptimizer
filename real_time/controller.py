"""Real-Time Microgrid Arbitrage Controller and Dispatcher.

Combines Day-Ahead spot price forecasts, solar radiation forecasts,
and real-time 2-minute telemetry to determine optimal inverter dispatch
using the mathematical Linear Programming (LP) optimization model
(identical to OptimizerArbitrage) solved via PuLP and CBC.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

try:
    from real_time.battery_sim import BatteryStepResult, FoxEssBatterySimulator
    from real_time.config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_W,
        CzechCommand,
        CzechReason,
        SOC_MAX,
        SOC_MIN,
    )
    from real_time.lp_optimizer import LpArbitrageOptimizerRT, PlannedHourAction
    from real_time.price_fetcher import SpotPriceFetcher
    from real_time.weather_fetcher import SolarWeatherFetcher
except ImportError:
    from battery_sim import BatteryStepResult, FoxEssBatterySimulator
    from config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_W,
        CzechCommand,
        CzechReason,
        SOC_MAX,
        SOC_MIN,
    )
    from lp_optimizer import LpArbitrageOptimizerRT, PlannedHourAction
    from price_fetcher import SpotPriceFetcher
    from weather_fetcher import SolarWeatherFetcher

logger = logging.getLogger("RealTimeController")


class RealTimeController:
    """Core decision engine executing real-time LP arbitrage dispatch."""

    def __init__(self, initial_soc: float = 50.0) -> None:
        self.price_fetcher = SpotPriceFetcher()
        self.weather_fetcher = SolarWeatherFetcher()
        self.lp_optimizer = LpArbitrageOptimizerRT()
        self.battery_sim = FoxEssBatterySimulator(initial_soc_pct=initial_soc)

        self._lp_schedule: dict[int, PlannedHourAction] = {}
        self._last_schedule_key: str = ""

    def sync_sim_soc_with_hardware(self, real_soc: float) -> None:
        """Align simulated SoC with physical hardware (used upon cold start or after long outage)."""
        self.battery_sim.set_soc(real_soc)

    def update_daily_schedule_if_needed(self, current_dt: dt.datetime | None = None) -> None:
        """Execute PuLP 24-hour Linear Programming MPC optimization."""
        ref_dt = current_dt or dt.datetime.now()
        hour_start = ref_dt.replace(minute=0, second=0, microsecond=0)
        schedule_key = hour_start.strftime("%Y-%m-%d-%H")

        if schedule_key == self._last_schedule_key and self._lp_schedule:
            return

        prices_df = self.price_fetcher.get_price_vector_15min(start_dt=hour_start, hours=24)
        if prices_df.empty:
            return

        pv_df = self.weather_fetcher.get_pv_forecast_vector_15min(start_dt=hour_start, hours=24)
        pv_forecast = pv_df["pv_forecast_kw"].tolist() if not pv_df.empty else [0.0] * len(prices_df)

        # Baseline household load profile (kW) based on typical residential patterns
        # Morning/day ~0.40 kW, evening peak ~0.80 kW, night ~0.25 kW
        load_forecast: list[float] = []
        for idx, row in prices_df.iterrows():
            h = row["start"].hour
            if 17 <= h <= 22:
                load_forecast.append(0.80)
            elif 7 <= h <= 16:
                load_forecast.append(0.45)
            else:
                load_forecast.append(0.25)

        cur_soc_kwh = (self.battery_sim.soc / 100.0) * BATTERY_CAPACITY_KWH
        hours_of_day = [row["start"].hour for _, row in prices_df.iterrows()]
        minute_slots = [row["start"].minute // 15 for _, row in prices_df.iterrows()]
        spot_prices = prices_df["price_czk_kwh"].tolist()
        cloud_forecast = pv_df["cloud_pct"].tolist() if "cloud_pct" in pv_df.columns else None

        # Solve Linear Programming Horizon (Adaptive Stochastic Risk MPC) with 15-min intervals
        schedule = self.lp_optimizer.solve_horizon(
            current_soc_kwh=cur_soc_kwh,
            spot_prices=spot_prices,
            pv_forecast_kw=pv_forecast,
            load_forecast_kw=load_forecast,
            hours_of_day=hours_of_day,
            cloud_cover_pct=cloud_forecast,
            dt_hours=0.25,
            minute_slots=minute_slots,
        )

        if schedule:
            self._lp_schedule = schedule
            self._last_schedule_key = schedule_key
            logger.info("LP Arbitrage 15-min schedule refreshed for %d slots ahead via PuLP.", len(schedule))

    def decide_tick_action(
        self,
        current_time_dt: dt.datetime,
        price_czk: float,
        p_home_w: float,
        p_pv_w: float,
        real_soc: float,
        dt_seconds: float,
    ) -> tuple[dict[str, Any], BatteryStepResult]:
        """Compute the optimal action for the current 2-minute tick using the LP schedule.

        Returns:
            (command_json_payload, step_result)
        """
        self.update_daily_schedule_if_needed(current_time_dt)

        current_hour = current_time_dt.hour
        current_slot = current_time_dt.minute // 15
        planned_action = self._lp_schedule.get((current_hour, current_slot)) or self._lp_schedule.get(current_hour)

        if planned_action is not None and planned_action.command != CzechCommand.STANDBY:
            command = planned_action.command
            target_power_w = planned_action.target_power_w
            reason = planned_action.reason
        else:
            # Default / Standby behavior
            command = CzechCommand.COVER_HOME
            target_power_w = BATTERY_POWER_MAX_W
            reason = CzechReason.HOME_SAVING

        # Physical solar activity detection (based on measured inverter generation, not clock hours)
        has_active_solar = (p_pv_w > 50.0)

        # Real-Time Closed-Loop Control & Disturbance Rejection:
        # 1. Closed-loop Solar Surplus Priority (Physics First):
        # If measured solar generation exceeds home consumption, store surplus in battery or export.
        # Exception: Do not override an explicit commercial B2G discharge planned by LP for an extreme peak.
        if p_pv_w > p_home_w and command != CzechCommand.DISCHARGE_GRID:
            if self.battery_sim.soc < (SOC_MAX * 100.0 - 0.5):
                command = CzechCommand.CHARGE_PV
                target_power_w = p_pv_w - p_home_w
                reason = CzechReason.PV_SURPLUS
            elif price_czk >= 0:
                command = CzechCommand.SELL_PV
                target_power_w = p_pv_w - p_home_w
                reason = CzechReason.SELL_SURPLUS
            else:
                command = CzechCommand.STANDBY
                target_power_w = 0.0
                reason = CzechReason.BATTERY_PROTECT_MAX

        # 2. Deficit Condition (Home consumption exceeds solar generation):
        elif p_home_w > p_pv_w:
            # Case A: LP planned grid charge (CHARGE_GRID)
            if command == CzechCommand.CHARGE_GRID:
                # With negative spot prices, always allow grid charging (grid pays us to consume)!
                if price_czk < 0:
                    pass  # Keep planned CHARGE_GRID
                # With positive spot prices during daylight solar: block grid charge, prioritize covering home
                elif has_active_solar:
                    if self.battery_sim.soc > (SOC_MIN * 100.0 + 0.1):
                        command = CzechCommand.COVER_HOME
                        target_power_w = min(BATTERY_POWER_MAX_W, p_home_w - p_pv_w)
                        reason = CzechReason.HOME_SAVING
                    else:
                        command = CzechCommand.STANDBY
                        target_power_w = 0.0
                        reason = CzechReason.BATTERY_PROTECT_MIN
                else:
                    # Nighttime pre-charge (Hurdle Rate verified by LP): allow CHARGE_GRID to proceed!
                    pass

            # Case B: Commercial export planned by LP (DISCHARGE_GRID)
            elif command == CzechCommand.DISCHARGE_GRID:
                pass  # Keep planned B2G export at target_power_w

            # Case C: Dynamic Home Load Tracking (COVER_HOME)
            elif command == CzechCommand.COVER_HOME:
                target_power_w = min(BATTERY_POWER_MAX_W, p_home_w - p_pv_w)

            # Case D: LP planned STANDBY (preserving battery capacity for upcoming expensive peak)
            elif command == CzechCommand.STANDBY:
                target_power_w = 0.0

        # 3. Hardware Protection Floor: if SoC <= 20.0%, strictly block any battery discharge
        if command in (CzechCommand.DISCHARGE_GRID, CzechCommand.COVER_HOME):
            if self.battery_sim.soc <= (SOC_MIN * 100.0 + 0.1):
                command = CzechCommand.STANDBY
                target_power_w = 0.0
                reason = CzechReason.BATTERY_PROTECT_MIN

        # 4. Hardware Protection Ceiling: if SoC >= 99.8%, strictly block any further charging
        if command in (CzechCommand.CHARGE_GRID, CzechCommand.CHARGE_PV):
            if self.battery_sim.soc >= (SOC_MAX * 100.0 - 0.2):
                if p_pv_w > p_home_w and price_czk >= 0:
                    command = CzechCommand.SELL_PV
                    target_power_w = p_pv_w - p_home_w
                    reason = CzechReason.SELL_SURPLUS
                else:
                    command = CzechCommand.STANDBY
                    target_power_w = 0.0
                    reason = CzechReason.BATTERY_PROTECT_MAX

        # Execute simulation step with dynamic dt
        step_result = self.battery_sim.step(
            command=command,
            target_power_w=target_power_w,
            p_home_w=p_home_w,
            p_pv_w=p_pv_w,
            dt_seconds=dt_seconds,
            price_czk_kwh=price_czk,
            reason=reason,
        )

        time_str = current_time_dt.strftime("%d.%m.%Y %H:%M:%S")
        if step_result.effective_command == CzechCommand.SELL_PV:
            cmd_power_w = int(max(0.0, step_result.p_grid_w))
        else:
            cmd_power_w = abs(int(step_result.p_bat_w))

        command_payload = {
            "prikaz": step_result.effective_command,
            "vykon_w": cmd_power_w,
            "duvod": step_result.effective_reason or reason,
            "sim_soc": step_result.new_sim_soc,
            "cas": time_str,
        }

        return command_payload, step_result

    def build_failsafe_command(self, current_time_dt: dt.datetime) -> dict[str, Any]:
        """Generate safe fallback command when telemetry stream is lost."""
        time_str = current_time_dt.strftime("%d.%m.%Y %H:%M:%S")
        return {
            "prikaz": CzechCommand.COVER_HOME,
            "vykon_w": 0,
            "duvod": CzechReason.FAILSAFE_TIMEOUT,
            "sim_soc": self.battery_sim.soc,
            "cas": time_str,
        }
