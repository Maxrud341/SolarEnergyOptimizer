import numpy as np
import pandas as pd
import pytest

from src.optimizer_arbitrage import (
    OptimizerArbitrageConfig,
    run_arbitrage_backtest,
    compute_arbitrage_kpis,
)


@pytest.fixture
def synthetic_arbitrage_data():
    dates = pd.date_range("2026-01-01 00:00:00", periods=72, freq="h")
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["consumption_kwh"] = 0.3 + 0.2 * np.sin(np.pi * df["hour"] / 12.0).clip(0, None)
    df["consumption_l1_kwh"] = df["consumption_kwh"] * 0.4
    df["consumption_l2_kwh"] = df["consumption_kwh"] * 0.3
    df["consumption_l3_kwh"] = df["consumption_kwh"] * 0.3
    df["production_kwh"] = 0.2 * np.sin(np.pi * (df["hour"] - 8) / 8.0).clip(0, None)  # Winter low PV
    
    # High volatility prices: night low (0.5 CZK), evening spike (8.0 CZK)
    base_price = 2.0 + 3.0 * np.sin(np.pi * (df["hour"] - 14) / 10.0)
    df["price_czk_per_kwh"] = base_price.clip(-1.0, 9.0)
    df["battery_kwh"] = 5.0
    df["temperature_c"] = -2.0
    df["humidity_pct"] = 80.0
    df["cloud_cover_pct"] = 90.0
    df["wind_speed_ms"] = 4.0
    return df


def test_arbitrage_energy_conservation(synthetic_arbitrage_data):
    cfg = OptimizerArbitrageConfig(battery_capacity_kwh=10.0, soc_min=0.05, soc_max=1.0)
    log = run_arbitrage_backtest(synthetic_arbitrage_data, cfg=cfg, max_steps=48)
    assert len(log) == 48

    # Energy balance per step: (buy + pv + dis) must match (load + sell + ch + curt)
    sources = log["buy_kwh"] + log["pv_kwh"] + log["discharge_kwh"]
    sinks = log["load_kwh"] + log["sell_kwh"] + log["charge_kwh"] + log["curtailment_kwh"]
    np.testing.assert_allclose(sources, sinks, atol=1e-4)


def test_battery_soc_stays_within_bounds(synthetic_arbitrage_data):
    cfg = OptimizerArbitrageConfig(battery_capacity_kwh=10.0, soc_min=0.05, soc_max=1.0)
    log = run_arbitrage_backtest(synthetic_arbitrage_data, cfg=cfg, max_steps=48)
    
    min_soc = cfg.battery_capacity_kwh * cfg.soc_min
    max_soc = cfg.battery_capacity_kwh * cfg.soc_max
    assert (log["soc_kwh"] >= min_soc - 1e-4).all()
    assert (log["soc_kwh"] <= max_soc + 1e-4).all()


def test_battery_to_grid_export_can_exceed_load(synthetic_arbitrage_data):
    # During high evening peaks, discharge can exceed local load to export commercially
    cfg = OptimizerArbitrageConfig(battery_capacity_kwh=10.0, battery_power_max_kw=3.6)
    log = run_arbitrage_backtest(synthetic_arbitrage_data, cfg=cfg, max_steps=48)
    
    # Check if there is any hour where discharge > load (which was forbidden in V2)
    b2g_hours = log[log["discharge_kwh"] > log["load_kwh"] + 0.1]
    assert len(b2g_hours) > 0, "B2G should export power beyond local load during high price peaks"


def test_zero_export_on_negative_sell_prices(synthetic_arbitrage_data):
    cfg = OptimizerArbitrageConfig(allow_curtailment=True)
    log = run_arbitrage_backtest(synthetic_arbitrage_data, cfg=cfg, max_steps=48)
    
    neg_sell_hours = log[log["p_sell"] < 0]
    if len(neg_sell_hours) > 0:
        assert (neg_sell_hours["sell_kwh"] == 0.0).all(), "Export should be 0 when sell price is negative"


def test_compute_arbitrage_kpis_validity(synthetic_arbitrage_data):
    log = run_arbitrage_backtest(synthetic_arbitrage_data, max_steps=24)
    kpis = compute_arbitrage_kpis(log)
    assert kpis["steps"] == 24
    assert kpis["total_load_kwh"] > 0
    assert "net_profit_czk" in kpis
    assert "total_cycles" in kpis
    assert "total_curt_kwh" in kpis


def test_zero_grid_precharge_before_sunny_day():
    """Verify that before a sunny day, the optimizer does NOT charge the battery from the grid at night."""
    dates = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h")
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["consumption_kwh"] = 0.2
    # Sunny day: 4 kW solar peak midday
    df["production_kwh"] = 4.0 * np.sin(np.pi * (df["hour"] - 6) / 14.0).clip(0, None)
    # Night is cheap (0.5 CZK), midday has negative prices (-0.2 CZK), evening has peak (6.0 CZK)
    prices = np.where(df["hour"] < 6, 0.5, np.where((df["hour"] >= 10) & (df["hour"] <= 14), -0.2, 3.0))
    df["price_czk_per_kwh"] = prices
    df["temperature_c"] = 25.0
    df["humidity_pct"] = 50.0
    df["cloud_cover_pct"] = 10.0
    df["wind_speed_ms"] = 2.0

    cfg = OptimizerArbitrageConfig()
    log = run_arbitrage_backtest(df, cfg=cfg, max_steps=24)

    # Night hours (00:00 - 05:00) before sun: charge_kwh must be 0 (no grid pre-charging)
    night_log = log[log["hour"] < 6] if "hour" in log.columns else log.iloc[:6]
    assert (night_log["charge_kwh"] == 0.0).all(), "Grid pre-charging at night before sunny day must be 0"


def test_curtailment_variable_prevents_infeasibility_on_negative_prices():
    """Verify that LP solves to Optimal even with huge PV surplus and negative export prices."""
    dates = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h")
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["consumption_kwh"] = 0.1
    # Massive solar generation: 8 kW
    df["production_kwh"] = 8.0 * np.sin(np.pi * (df["hour"] - 6) / 14.0).clip(0, None)
    # Negative spot prices all day
    df["price_czk_per_kwh"] = -0.5
    df["temperature_c"] = 25.0
    df["humidity_pct"] = 50.0
    df["cloud_cover_pct"] = 10.0
    df["wind_speed_ms"] = 2.0

    cfg = OptimizerArbitrageConfig(allow_curtailment=True)
    log = run_arbitrage_backtest(df, cfg=cfg, max_steps=24)
    assert len(log) == 24
    # Zero export must be respected
    assert (log["sell_kwh"] == 0.0).all()
    # Curtailment must absorb the excess solar
    assert log["curtailment_kwh"].sum() > 0.0
    # Energy balance must hold exactly
    sources = log["buy_kwh"] + log["pv_kwh"] + log["discharge_kwh"]
    sinks = log["load_kwh"] + log["sell_kwh"] + log["charge_kwh"] + log["curtailment_kwh"]
    np.testing.assert_allclose(sources, sinks, atol=1e-4)
