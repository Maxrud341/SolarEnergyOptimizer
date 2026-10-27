"""Unit Tests for Real-Time MQTT, Battery Simulation, and Storage Subsystem."""

from __future__ import annotations

import datetime as dt
import pytest

from real_time.battery_sim import FoxEssBatterySimulator
from real_time.config import (
    BATTERY_CAPACITY_KWH,
    BATTERY_POWER_MAX_W,
    CzechCommand,
    CzechReason,
    SOC_MAX,
    SOC_MIN,
)
from real_time.controller import RealTimeController
from real_time.database import (
    get_recent_ticks,
    init_db,
    load_sim_soc,
    save_sim_soc,
    save_telemetry_tick,
)


def test_battery_simulator_discharge_protection():
    """Verify that Fox-ESS simulator protects against discharging below 20% SoC."""
    # Start at 20.5% (very close to 20.0% floor)
    sim = FoxEssBatterySimulator(initial_soc_pct=20.5)

    # Request discharge to grid of 5000 W for 2 minutes
    res = sim.step(
        command=CzechCommand.DISCHARGE_GRID,
        target_power_w=5000.0,
        p_home_w=500.0,
        p_pv_w=0.0,
        dt_seconds=120.0,
        price_czk_kwh=6.50,
    )

    # Must not drop below 20.0% SoC
    assert res.new_sim_soc >= 20.0
    assert sim.soc >= 20.0

    # If already at 20.0%, requesting discharge must trigger STANDBY & protection reason
    sim.set_soc(20.0)
    res_empty = sim.step(
        command=CzechCommand.DISCHARGE_GRID,
        target_power_w=5000.0,
        p_home_w=500.0,
        p_pv_w=0.0,
        dt_seconds=120.0,
        price_czk_kwh=6.50,
    )
    assert res_empty.effective_command == CzechCommand.STANDBY
    assert res_empty.effective_reason == CzechReason.BATTERY_PROTECT_MIN
    assert res_empty.p_bat_w == 0.0


def test_battery_simulator_charge_protection():
    """Verify that Fox-ESS simulator protects against charging above 100% SoC."""
    sim = FoxEssBatterySimulator(initial_soc_pct=99.8)

    res = sim.step(
        command=CzechCommand.CHARGE_GRID,
        target_power_w=5000.0,
        p_home_w=200.0,
        p_pv_w=0.0,
        dt_seconds=120.0,
        price_czk_kwh=1.20,
    )

    # Must clamp at 100.0%
    assert res.new_sim_soc <= 100.0
    assert sim.soc <= 100.0

    # Full battery should switch command to STANDBY
    sim.set_soc(100.0)
    res_full = sim.step(
        command=CzechCommand.CHARGE_GRID,
        target_power_w=5000.0,
        p_home_w=200.0,
        p_pv_w=0.0,
        dt_seconds=120.0,
        price_czk_kwh=1.20,
    )
    assert res_full.effective_command == CzechCommand.STANDBY
    assert res_full.effective_reason == CzechReason.BATTERY_PROTECT_MAX
    assert res_full.p_bat_w == 0.0


def test_database_persistence_and_ticks():
    """Test SQLite initialization, tick insertion, and state persistence."""
    init_db()

    # Save and reload custom simulated SoC
    save_sim_soc(88.4)
    loaded_soc = load_sim_soc()
    assert abs(loaded_soc - 88.4) < 1e-4

    # Save a telemetry tick
    now_iso = dt.datetime.now().isoformat()
    save_telemetry_tick(
        timestamp=now_iso,
        date_str="26.10.2026",
        time_str="18:30:00",
        dt_seconds=120.0,
        price_czk=4.25,
        p_home_w=650.0,
        p_pv_w=1200.0,
        p_grid_w=550.0,
        p_bat_w=0.0,
        real_soc=75.0,
        sim_soc=88.4,
        command=CzechCommand.CHARGE_PV,
        power_w=550.0,
        reason=CzechReason.PV_SURPLUS,
        profit_step_czk=0.078,
    )

    df = get_recent_ticks(limit=10)
    assert not df.empty
    last_row = df.iloc[-1]
    assert last_row["command"] == CzechCommand.CHARGE_PV
    assert last_row["price_czk"] == 4.25


def test_realtime_controller_decision():
    """Verify controller decision logic and JSON payload output."""
    controller = RealTimeController(initial_soc=60.0)

    now_dt = dt.datetime(2026, 10, 26, 18, 0, 0)
    cmd_dict, step_res = controller.decide_tick_action(
        current_time_dt=now_dt,
        price_czk=6.80,  # Peak evening price
        p_home_w=800.0,
        p_pv_w=0.0,
        real_soc=60.0,
        dt_seconds=120.0,
    )

    # Validate Czech JSON payload fields
    assert "prikaz" in cmd_dict
    assert "vykon_w" in cmd_dict
    assert "duvod" in cmd_dict
    assert "sim_soc" in cmd_dict
    assert "cas" in cmd_dict
    assert isinstance(cmd_dict["vykon_w"], int)
    assert 20.0 <= cmd_dict["sim_soc"] <= 100.0


def test_battery_first_closed_loop_when_home_exceeds_pv():
    """Verify that if house load suddenly spikes above solar PV, the battery immediately discharges

    to cover the deficit (Battery First / Zero Grid Buy) even if the static plan had planned charging.
    """
    controller = RealTimeController(initial_soc=100.0)

    # 14:53 scenario: Home spikes to 5200W, Solar is 4500W (deficit 700W), battery is 100%
    now_dt = dt.datetime(2026, 10, 27, 14, 53, 0)
    cmd_dict, step_res = controller.decide_tick_action(
        current_time_dt=now_dt,
        price_czk=3.68,
        p_home_w=5200.0,
        p_pv_w=4500.0,
        real_soc=100.0,
        dt_seconds=30.0,
    )

    # Must prioritize COVER_HOME to protect against grid buy
    assert cmd_dict["prikaz"] == CzechCommand.COVER_HOME
    assert cmd_dict["duvod"] == CzechReason.HOME_SAVING
    assert abs(cmd_dict["vykon_w"] - 700) <= 2  # Discharges exactly the 700W deficit
    assert step_res.p_bat_w > 0.0  # Positive = battery discharge
    assert abs(step_res.p_grid_w) < 1.0  # Zero grid purchase!

