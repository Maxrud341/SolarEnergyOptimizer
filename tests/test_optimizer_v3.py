import numpy as np
import pandas as pd
import pytest

from src.optimizer_v3 import (
    OptimizerV3Config,
    ForecasterEngineV3,
    compute_financial_kpis_v3,
    run_backtest_v3,
    run_naive_baseline_v3,
    run_v2_baseline_v3,
)


@pytest.fixture
def synthetic_microgrid_data_v3():
    dates = pd.date_range("2025-06-01 00:00:00", periods=72, freq="h")
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["consumption_kwh"] = 0.5 + 0.4 * np.sin(np.pi * df["hour"] / 12.0).clip(0, None)
    df["consumption_l1_kwh"] = df["consumption_kwh"] * 0.3
    df["consumption_l2_kwh"] = df["consumption_kwh"] * 0.2
    df["consumption_l3_kwh"] = df["consumption_kwh"] * 0.5  # Boiler load on L3
    df["pv_forecast_kwh"] = 4.0 * np.sin(np.pi * (df["hour"] - 6) / 14.0).clip(0, None)
    df["production_kwh"] = df["pv_forecast_kwh"] * 0.95
    df["price_czk_per_kwh"] = 2.0 + 1.5 * np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["battery_kwh"] = 5.0
    df["temperature_c"] = 20.0
    df["humidity_pct"] = 60.0
    df["cloud_cover_pct"] = 20.0
    df["wind_speed_ms"] = 3.0
    return df


def test_forecaster_engine_v3_caches_models(synthetic_microgrid_data_v3):
    engine = ForecasterEngineV3(model_type="linear", retrain_cadence_hours=24)
    engine.maybe_retrain(synthetic_microgrid_data_v3.iloc[:48], current_step=0)
    assert engine.last_train_step == 0
    assert len(engine.models) >= 4

    # Calling at step 10 should use cache (not retrain)
    engine.maybe_retrain(synthetic_microgrid_data_v3.iloc[:48], current_step=10)
    assert engine.last_train_step == 0

    # Calling at step 25 should trigger retrain
    engine.maybe_retrain(synthetic_microgrid_data_v3.iloc[:48], current_step=25)
    assert engine.last_train_step == 25


def test_active_sector_coupling_diverts_surplus_to_heat(synthetic_microgrid_data_v3):
    cfg = OptimizerV3Config(
        battery_capacity_kwh=10.0,
        boiler_capacity_kwh=8.0,
        boiler_enabled=True,
        asymmetric_metering=True,
    )
    log = run_backtest_v3(synthetic_microgrid_data_v3, cfg=cfg, max_steps=48)
    assert len(log) == 48
    assert "heat_from_boiler_kwh" in log.columns
    assert "electric_to_boiler_kwh" in log.columns
    assert "boiler_soc_kwh" in log.columns

    kpi = compute_financial_kpis_v3(log)
    assert kpi["steps"] == 48
    assert kpi["solar_to_heat_kwh"] >= 0.0
    assert kpi["heat_supplied_kwh"] >= 0.0


def test_asymmetric_metering_reduces_costs_compared_to_symmetric(synthetic_microgrid_data_v3):
    cfg_asym = OptimizerV3Config(boiler_enabled=False, asymmetric_metering=True)
    cfg_sym = OptimizerV3Config(boiler_enabled=False, asymmetric_metering=False)

    log_asym = run_backtest_v3(synthetic_microgrid_data_v3, cfg=cfg_asym, max_steps=48)
    log_sym = run_backtest_v3(synthetic_microgrid_data_v3, cfg=cfg_sym, max_steps=48)

    kpi_asym = compute_financial_kpis_v3(log_asym)
    kpi_sym = compute_financial_kpis_v3(log_sym)

    # Asymmetric metering should be strictly more profitable or equal
    assert kpi_asym["net_profit_czk"] >= kpi_sym["net_profit_czk"] - 1e-4


def test_3_way_baseline_benchmark_execution(synthetic_microgrid_data_v3):
    cfg = OptimizerV3Config(battery_capacity_kwh=10.0, boiler_capacity_kwh=8.0)
    log_v3 = run_backtest_v3(synthetic_microgrid_data_v3, cfg=cfg, max_steps=24)
    log_v2 = run_v2_baseline_v3(synthetic_microgrid_data_v3, cfg=cfg, max_steps=24)
    log_n = run_naive_baseline_v3(synthetic_microgrid_data_v3, cfg=cfg, max_steps=24)

    assert len(log_v3) == 24
    assert len(log_v2) == 24
    assert len(log_n) == 24
