"""Simulated Battery State Machine for Fox-ESS HV2600 HS 10.4.

Maintains internal energy state, executes 2-minute dynamic steps,
enforces depth-of-discharge (SoC 20%-100%) and continuous power limits (5.0 kW),
and calculates physical energy conservation and financial throughput.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from real_time.config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_W,
        CzechCommand,
        CzechReason,
        DEGRADATION_COST_CZK_PER_KWH,
        EFFICIENCY_CHARGE,
        EFFICIENCY_DISCHARGE,
        SOC_MAX,
        SOC_MIN,
    )
except ImportError:
    from config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_W,
        CzechCommand,
        CzechReason,
        DEGRADATION_COST_CZK_PER_KWH,
        EFFICIENCY_CHARGE,
        EFFICIENCY_DISCHARGE,
        SOC_MAX,
        SOC_MIN,
    )


@dataclass
class BatteryStepResult:
    """Outcome of a single dynamic simulation step."""

    new_sim_soc: float         # In percent [20.0, 100.0]
    p_bat_w: float             # Actual battery power (positive = discharging, negative = charging)
    p_grid_w: float            # Actual grid exchange (positive = export/sell, negative = import/buy)
    effective_command: str     # Command actually executed (or downgraded by protection)
    effective_reason: str      # Explanation
    profit_step_czk: float     # Step net profit or financial balance (CZK)


class FoxEssBatterySimulator:
    """State machine modeling the Fox-ESS HV2600 HS 10.4 (10.24 kWh) system."""

    def __init__(self, initial_soc_pct: float = 50.0) -> None:
        self.capacity_kwh: float = BATTERY_CAPACITY_KWH
        self.soc_min: float = SOC_MIN * 100.0  # 20.0%
        self.soc_max: float = SOC_MAX * 100.0  # 100.0%
        self.max_power_w: float = BATTERY_POWER_MAX_W
        self.soc: float = max(self.soc_min, min(self.soc_max, initial_soc_pct))

    def set_soc(self, soc_pct: float) -> None:
        """Manually align or reset the simulated State of Charge."""
        self.soc = max(self.soc_min, min(self.soc_max, soc_pct))

    def step(
        self,
        command: str,
        target_power_w: float,
        p_home_w: float,
        p_pv_w: float,
        dt_seconds: float,
        price_czk_kwh: float,
    ) -> BatteryStepResult:
        """Execute one dynamic step of duration dt_seconds.

        Conventions:
        - p_home_w: positive electrical load of house (W)
        - p_pv_w: positive solar generation (W)
        - p_bat_w: positive = discharging (power out of battery), negative = charging (power into battery)
        - p_grid_w: positive = export to grid (selling), negative = import from grid (buying)
        """
        dt_hours = max(1.0, dt_seconds) / 3600.0
        # Energy headroom
        current_energy_kwh = (self.soc / 100.0) * self.capacity_kwh
        max_energy_kwh = (self.soc_max / 100.0) * self.capacity_kwh
        min_energy_kwh = (self.soc_min / 100.0) * self.capacity_kwh

        headroom_charge_kwh = max(0.0, max_energy_kwh - current_energy_kwh)
        headroom_discharge_kwh = max(0.0, current_energy_kwh - min_energy_kwh)

        effective_command = command
        effective_reason = ""
        p_bat_w = 0.0

        # -------------------------------------------------------------
        # 1. Evaluate Inverter Commands
        # -------------------------------------------------------------
        if command == CzechCommand.CHARGE_GRID:
            # Force grid charge
            requested_w = min(self.max_power_w, target_power_w if target_power_w > 0 else self.max_power_w)
            # Max electrical power that can be absorbed before hitting 100% SoC
            max_absorbable_w = (headroom_charge_kwh / (dt_hours * EFFICIENCY_CHARGE)) * 1000.0
            actual_w = min(requested_w, max_absorbable_w)

            if actual_w <= 10.0:
                effective_command = CzechCommand.STANDBY
                effective_reason = CzechReason.BATTERY_PROTECT_MAX
                p_bat_w = 0.0
            else:
                p_bat_w = -actual_w  # Negative = charging
                effective_reason = CzechReason.LOW_PRICE

        elif command == CzechCommand.DISCHARGE_GRID:
            # Arbitrage export to grid
            requested_w = min(self.max_power_w, target_power_w if target_power_w > 0 else self.max_power_w)
            # Max electrical power that can be delivered before hitting 20% SoC
            max_deliverable_w = (headroom_discharge_kwh * EFFICIENCY_DISCHARGE / dt_hours) * 1000.0
            actual_w = min(requested_w, max_deliverable_w)

            if actual_w <= 10.0:
                effective_command = CzechCommand.STANDBY
                effective_reason = CzechReason.BATTERY_PROTECT_MIN
                p_bat_w = 0.0
            else:
                p_bat_w = actual_w  # Positive = discharging
                effective_reason = CzechReason.HIGH_PRICE

        elif command == CzechCommand.COVER_HOME:
            # Self-consumption: cover house load
            net_home_demand = max(0.0, p_home_w - p_pv_w)
            requested_w = min(self.max_power_w, net_home_demand)
            max_deliverable_w = (headroom_discharge_kwh * EFFICIENCY_DISCHARGE / dt_hours) * 1000.0
            actual_w = min(requested_w, max_deliverable_w)

            if actual_w > 10.0:
                p_bat_w = actual_w
                effective_reason = CzechReason.HOME_SAVING
            elif p_pv_w > p_home_w and headroom_charge_kwh > 0.01:
                # Surplus solar automatically goes to battery
                surplus_w = p_pv_w - p_home_w
                max_absorbable_w = (headroom_charge_kwh / (dt_hours * EFFICIENCY_CHARGE)) * 1000.0
                charge_w = min(self.max_power_w, surplus_w, max_absorbable_w)
                p_bat_w = -charge_w
                effective_command = CzechCommand.CHARGE_PV
                effective_reason = CzechReason.PV_SURPLUS
            else:
                effective_command = CzechCommand.STANDBY
                effective_reason = (
                    CzechReason.BATTERY_PROTECT_MIN
                    if headroom_discharge_kwh <= 0.01
                    else "Nulová čistá spotřeba"
                )
                p_bat_w = 0.0

        elif command == CzechCommand.CHARGE_PV:
            # Solar charging only
            surplus_w = max(0.0, p_pv_w - p_home_w)
            max_absorbable_w = (headroom_charge_kwh / (dt_hours * EFFICIENCY_CHARGE)) * 1000.0
            actual_w = min(self.max_power_w, surplus_w, max_absorbable_w)

            if actual_w > 10.0:
                p_bat_w = -actual_w
                effective_reason = CzechReason.PV_SURPLUS
            else:
                effective_command = CzechCommand.STANDBY
                effective_reason = (
                    CzechReason.BATTERY_PROTECT_MAX
                    if headroom_charge_kwh <= 0.01
                    else "Nedostatek sluneční výroby"
                )
                p_bat_w = 0.0

        else:  # STANDBY or idle
            effective_command = CzechCommand.STANDBY
            effective_reason = CzechReason.STANDBY_SCHEDULE
            p_bat_w = 0.0

        # -------------------------------------------------------------
        # 2. Update Battery SoC
        # -------------------------------------------------------------
        if p_bat_w < 0:
            # Charging
            delta_kwh = abs(p_bat_w) / 1000.0 * dt_hours * EFFICIENCY_CHARGE
            current_energy_kwh += delta_kwh
            throughput_kwh = abs(p_bat_w) / 1000.0 * dt_hours
        elif p_bat_w > 0:
            # Discharging
            delta_kwh = (p_bat_w / 1000.0 * dt_hours) / EFFICIENCY_DISCHARGE
            current_energy_kwh -= delta_kwh
            throughput_kwh = p_bat_w / 1000.0 * dt_hours
        else:
            throughput_kwh = 0.0

        self.soc = round((current_energy_kwh / self.capacity_kwh) * 100.0, 2)
        self.soc = max(self.soc_min, min(self.soc_max, self.soc))

        # -------------------------------------------------------------
        # 3. Electrical Power Balance at Grid Meter
        # P_grid = P_pv + P_bat - P_home (positive = export/sell, negative = import/buy)
        # -------------------------------------------------------------
        p_grid_w = p_pv_w + p_bat_w - p_home_w

        # -------------------------------------------------------------
        # 4. Financial Step Calculation
        # -------------------------------------------------------------
        grid_kwh = (p_grid_w / 1000.0) * dt_hours
        degradation_cost = throughput_kwh * DEGRADATION_COST_CZK_PER_KWH

        if grid_kwh > 0:
            # Selling to grid at spot price
            financial_gain = grid_kwh * price_czk_kwh
        else:
            # Buying from grid at spot price (cost is negative)
            financial_gain = grid_kwh * price_czk_kwh

        step_net_profit = financial_gain - degradation_cost

        return BatteryStepResult(
            new_sim_soc=self.soc,
            p_bat_w=round(p_bat_w, 1),
            p_grid_w=round(p_grid_w, 1),
            effective_command=effective_command,
            effective_reason=effective_reason,
            profit_step_czk=round(step_net_profit, 4),
        )
