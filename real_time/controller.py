"""Real-Time Microgrid Arbitrage Controller and Dispatcher.

Combines Day-Ahead spot price forecasts, solar radiation forecasts,
and real-time 2-minute telemetry to determine optimal inverter dispatch
and construct Czech JSON command payloads for FVE/Maly_RX.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from real_time.battery_sim import BatteryStepResult, FoxEssBatterySimulator
from real_time.config import (
    BATTERY_POWER_MAX_W,
    CzechCommand,
    CzechReason,
    SOC_MAX,
    SOC_MIN,
)
from real_time.price_fetcher import SpotPriceFetcher
from real_time.weather_fetcher import SolarWeatherFetcher

logger = logging.getLogger("RealTimeController")


class RealTimeController:
    """Core decision engine for the real-time battery arbitrator."""

    def __init__(self, initial_soc: float = 50.0) -> None:
        self.price_fetcher = SpotPriceFetcher()
        self.weather_fetcher = SolarWeatherFetcher()
        self.battery_sim = FoxEssBatterySimulator(initial_soc_pct=initial_soc)

        self._hourly_schedule: dict[int, str] = {}
        self._last_schedule_date: str = ""

    def sync_sim_soc_with_hardware(self, real_soc: float) -> None:
        """Align simulated SoC with physical hardware (used upon cold start or after long outage)."""
        self.battery_sim.set_soc(real_soc)

    def update_daily_schedule_if_needed(self) -> None:
        """Refresh price & weather curves and construct hourly arbitrage guidelines."""
        now = dt.datetime.now()
        today_key = now.strftime("%Y-%m-%d-%H")

        if today_key == self._last_schedule_date:
            return

        prices_df = self.price_fetcher.get_hourly_price_vector(start_dt=now, hours=24)
        if prices_df.empty:
            return

        # Find price distribution over the next 24 hours
        price_vals = prices_df["price_czk_kwh"].tolist()
        if not price_vals:
            return

        min_price = min(price_vals)
        max_price = max(price_vals)
        median_price = float(sorted(price_vals)[len(price_vals) // 2])

        schedule: dict[int, str] = {}
        for idx, row in prices_df.iterrows():
            hour_of_day = row["start"].hour
            p = row["price_czk_kwh"]

            # Arbitrage logic:
            # 1. Ultra-cheap or negative hours -> Force charge from grid
            if p <= min_price + 0.30 or p < 1.0:
                schedule[hour_of_day] = CzechCommand.CHARGE_GRID
            # 2. Peak top prices -> Force discharge into grid
            elif p >= max_price - 0.40 and p > 4.50:
                schedule[hour_of_day] = CzechCommand.DISCHARGE_GRID
            # 3. Intermediate prices -> Cover home load (self-consumption)
            else:
                schedule[hour_of_day] = CzechCommand.COVER_HOME

        self._hourly_schedule = schedule
        self._last_schedule_date = today_key

    def decide_tick_action(
        self,
        current_time_dt: dt.datetime,
        price_czk: float,
        p_home_w: float,
        p_pv_w: float,
        real_soc: float,
        dt_seconds: float,
    ) -> tuple[dict[str, Any], BatteryStepResult]:
        """Compute the optimal action for the current 2-minute tick.

        Returns:
            (command_json_payload, step_result)
        """
        self.update_daily_schedule_if_needed()

        current_hour = current_time_dt.hour
        planned_cmd = self._hourly_schedule.get(current_hour, CzechCommand.COVER_HOME)
        target_power_w = BATTERY_POWER_MAX_W

        # Dynamic real-time overrides:
        # If spot price is negative or dirt cheap (< 1.0 CZK), aggressively charge from grid
        if price_czk < 1.0 and self.battery_sim.soc < (SOC_MAX * 100.0 - 2.0):
            planned_cmd = CzechCommand.CHARGE_GRID
            target_power_w = BATTERY_POWER_MAX_W

        # If solar generation exceeds home consumption, charge battery from PV
        elif p_pv_w > p_home_w and planned_cmd != CzechCommand.DISCHARGE_GRID:
            if self.battery_sim.soc < (SOC_MAX * 100.0 - 1.0):
                planned_cmd = CzechCommand.CHARGE_PV
                target_power_w = p_pv_w - p_home_w

        # If battery is close to empty (<=20%), avoid discharging
        elif planned_cmd in (CzechCommand.DISCHARGE_GRID, CzechCommand.COVER_HOME):
            if self.battery_sim.soc <= (SOC_MIN * 100.0 + 0.5):
                planned_cmd = CzechCommand.STANDBY

        # Execute simulation step with dynamic dt
        step_result = self.battery_sim.step(
            command=planned_cmd,
            target_power_w=target_power_w,
            p_home_w=p_home_w,
            p_pv_w=p_pv_w,
            dt_seconds=dt_seconds,
            price_czk_kwh=price_czk,
        )

        # Build final outgoing Czech JSON command
        time_str = current_time_dt.strftime("%d.%m.%Y %H:%M:%S")
        command_payload = {
            "prikaz": step_result.effective_command,
            "vykon_w": abs(int(step_result.p_bat_w)),
            "duvod": step_result.effective_reason,
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
