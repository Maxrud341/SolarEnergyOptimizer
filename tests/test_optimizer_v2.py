import numpy as np
import pandas as pd
import pytest

from src.optimizer_v2 import (
    OptimizerV2Config,
    compute_feature_weights_summary,
    compute_financial_kpis,
    fit_statistical_forecaster,
    predict_consumption,
    predict_with_forecaster,
    prepare_feature_frame,
    run_backtest_v2,
    run_naive_baseline_v2,
    solve_milp_window_v2,
)


@pytest.fixture
def synthetic_microgrid_data():
    dates = pd.date_range("2025-06-01 00:00:00", periods=72, freq="h")
    df = pd.DataFrame({"datetime": dates})
    df["hour"] = df["datetime"].dt.hour
    df["consumption_kwh"] = 0.3 + 0.4 * np.sin(np.pi * df["hour"] / 12.0).clip(0, None)
    df["pv_forecast_kwh"] = 4.0 * np.sin(np.pi * (df["hour"] - 6) / 14.0).clip(0, None)
    df["production_kwh"] = df["pv_forecast_kwh"] * 0.95
    df["price_czk_per_kwh"] = 2.0 + 1.5 * np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["battery_kwh"] = 5.0
    df["temperature_c"] = 20.0
    df["humidity_pct"] = 60.0
    df["cloud_cover_pct"] = 20.0
    df["wind_speed_ms"] = 3.0
    df["grid_import_kwh"] = 0.1
    df["grid_export_kwh"] = 0.0
    return df


def test_prepare_feature_frame_generates_cyclic_and_calendar_features(synthetic_microgrid_data):
    frame = prepare_feature_frame(synthetic_microgrid_data)
    for col in ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "is_daytime", "is_weekend"):
        assert col in frame.columns
        assert not frame[col].isna().any()


def test_forecaster_fits_and_respects_nighttime_zero_solar(synthetic_microgrid_data):
    model = fit_statistical_forecaster(synthetic_microgrid_data, "production_kwh")
    preds = predict_with_forecaster(model, synthetic_microgrid_data)

    assert len(preds) == len(synthetic_microgrid_data)
    assert np.all(preds >= 0.0)

    # Nighttime hours (e.g. midnight, 2 AM) must be strictly 0
    night_idx = synthetic_microgrid_data[synthetic_microgrid_data["hour"].isin([0, 1, 2, 3, 23])].index
    assert np.all(preds[night_idx] == 0.0)


def test_feature_weights_summary_returns_valid_kpis(synthetic_microgrid_data):
    summary = compute_feature_weights_summary(synthetic_microgrid_data)
    assert "production_kwh" in summary
    assert "price_czk_per_kwh" in summary
    assert "consumption_kwh" in summary

    prod_info = summary["production_kwh"]
    assert 0.0 <= prod_info["r2"] <= 1.0
    assert "pv_forecast_kwh" in prod_info["weights"]
    assert prod_info["weights"]["pv_forecast_kwh"] > 0.0


def test_milp_solver_enforces_binary_mutual_exclusions(synthetic_microgrid_data):
    cfg = OptimizerV2Config(battery_capacity_kwh=10.0, battery_power_max_kw=5.0)
    window = synthetic_microgrid_data.iloc[:24].copy().reset_index(drop=True)
    plan_df, meta = solve_milp_window_v2(window, current_soc_kwh=5.0, cfg=cfg)

    assert meta["status"] in ("Optimal", "Feasible")
    assert len(plan_df) == 24

    for _, row in plan_df.iterrows():
        # Mutual exclusion between buy and sell
        assert not (row["buy_kwh"] > 1e-4 and row["sell_kwh"] > 1e-4)
        # Mutual exclusion between charge and discharge
        assert not (row["charge_kwh"] > 1e-4 and row["discharge_kwh"] > 1e-4)
        # SoC bounds
        assert cfg.soc_min * cfg.battery_capacity_kwh - 1e-4 <= row["soc_kwh"] <= cfg.soc_max * cfg.battery_capacity_kwh + 1e-4


def test_naive_baseline_conserves_energy_bounds(synthetic_microgrid_data):
    cfg = OptimizerV2Config(battery_capacity_kwh=10.0)
    base_df = run_naive_baseline_v2(synthetic_microgrid_data, cfg=cfg, max_steps=24)

    min_soc = cfg.soc_min * cfg.battery_capacity_kwh
    max_soc = cfg.soc_max * cfg.battery_capacity_kwh

    assert np.all(base_df["soc_kwh"] >= min_soc - 1e-6)
    assert np.all(base_df["soc_kwh"] <= max_soc + 1e-6)
    assert not (base_df["buy_kwh"] < 0).any()
    assert not (base_df["sell_kwh"] < 0).any()


def test_backtest_closed_loop_executes_and_computes_kpis(synthetic_microgrid_data):
    cfg = OptimizerV2Config(battery_capacity_kwh=10.0, horizon_hours=12)
    milp_log = run_backtest_v2(synthetic_microgrid_data, cfg=cfg, max_steps=24)

    assert len(milp_log) == 24
    assert "cost_buy_czk" in milp_log.columns
    assert "revenue_sell_czk" in milp_log.columns
    assert "net_profit_czk" in milp_log.columns

    kpi = compute_financial_kpis(milp_log)
    assert kpi["steps"] == 24
    assert kpi["total_load_kwh"] > 0
    assert "net_profit_czk" in kpi


def test_milp_solver_enforces_zero_battery_export_to_grid(synthetic_microgrid_data):
    cfg = OptimizerV2Config(strategy="milp", horizon_hours=12)
    plan, _ = solve_milp_window_v2(synthetic_microgrid_data.iloc[:12], current_soc_kwh=8.0, cfg=cfg)
    for idx, row in plan.iterrows():
        assert row["discharge_kwh"] <= row["forecast_load_kwh"] + 1e-4


def test_phase_aware_consumption_prediction(synthetic_microgrid_data):
    data = synthetic_microgrid_data.copy()
    data["consumption_l1_kwh"] = data["consumption_kwh"] * 0.3
    data["consumption_l2_kwh"] = data["consumption_kwh"] * 0.2
    data["consumption_l3_kwh"] = data["consumption_kwh"] * 0.5

    # 1. Test predict_consumption sums phases accurately
    load_pred, phase_dict = predict_consumption(data.iloc[:48], data.iloc[48:72], model_type="linear")
    assert len(load_pred) == 24
    assert np.allclose(load_pred, phase_dict["l1"] + phase_dict["l2"] + phase_dict["l3"], atol=1e-5)

    # 2. Test backtest logs all phase metrics
    cfg = OptimizerV2Config(battery_capacity_kwh=10.0, horizon_hours=12)
    log = run_backtest_v2(data, cfg=cfg, max_steps=24)
    for phase_metric in ("load_l1_kwh", "load_l2_kwh", "load_l3_kwh", "forecast_load_l1_kwh", "forecast_load_l2_kwh", "forecast_load_l3_kwh"):
        assert phase_metric in log.columns

    kpi = compute_financial_kpis(log)
    assert "total_load_l1_kwh" in kpi
    assert "total_load_l2_kwh" in kpi
    assert "total_load_l3_kwh" in kpi
    assert np.isclose(kpi["total_load_kwh"], kpi["total_load_l1_kwh"] + kpi["total_load_l2_kwh"] + kpi["total_load_l3_kwh"], atol=1e-4)


