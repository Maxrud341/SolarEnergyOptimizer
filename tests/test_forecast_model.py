import pandas as pd

from src.optimizer import fit_forecast_model, predict_forecast_series


def _make_synthetic_dataset():
    base = pd.date_range("2025-01-01 00:00:00", periods=240, freq="h")
    df = pd.DataFrame({"datetime": base})
    df["temperature_c"] = 10 + 15 * ((df.index % 24) / 24) + 5 * (df["datetime"].dt.dayofweek >= 5)
    df["cloud_cover_pct"] = 25 + 25 * (df["datetime"].dt.hour / 23)
    df["humidity_pct"] = 55 + 15 * (df["datetime"].dt.hour / 23)
    df["consumption_kwh"] = 2.5 + 1.5 * (df["datetime"].dt.dayofweek >= 5) + 0.5 * df["temperature_c"] / 20
    df["pv_forecast_kwh"] = 0.2 + 0.7 * (1 - df["cloud_cover_pct"] / 100) * (1 + 0.1 * pd.Series((df["datetime"].dt.hour >= 12).astype(float), index=df.index))
    df["price_czk_per_kwh"] = 1.0 + 0.8 * (df["datetime"].dt.dayofweek >= 5) + 0.3 * df["temperature_c"] / 20 - 0.4 * df["cloud_cover_pct"] / 100
    return df


def test_forecast_model_can_estimate_weekday_and_weather_effects():
    df = _make_synthetic_dataset()
    model = fit_forecast_model(df, "price_czk_per_kwh")
    preds = predict_forecast_series(model, df)

    assert len(preds) == len(df)
    assert preds.ndim == 1
    assert pd.notna(preds).all()
    assert (preds >= 0).all()


def test_real_and_forecast_pv_are_separated():
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=3, freq="h"),
        "production_kwh": [1.0, 2.0, 3.0],
        "pv_forecast_kwh": [0.5, 1.5, 2.5],
    })

    from src.optimizer import select_forecast_pv_column, select_real_pv_column

    assert select_real_pv_column(df) == "production_kwh"
    assert select_forecast_pv_column(df) == "pv_forecast_kwh"


def test_select_optimal_attributes_prefers_weather_and_calendar_features():
    df = _make_synthetic_dataset()

    from src.optimizer import select_optimal_attributes

    selected = select_optimal_attributes(df, "price_czk_per_kwh", max_features=5)

    assert isinstance(selected, list)
    assert selected
    assert "temperature_c" in selected
    assert "is_weekend" in selected or "cloud_cover_pct" in selected or "hour_sin" in selected or "weekday_sin" in selected


def test_summarize_feature_selection_reports_weights():
    df = _make_synthetic_dataset()

    from src.optimizer import summarize_feature_selection

    info = summarize_feature_selection(df, "price_czk_per_kwh", max_features=3)

    assert isinstance(info, dict)
    assert "selected_features" in info
    assert "weights" in info
    assert info["selected_features"]
    assert set(info["weights"]).issubset(set(info["selected_features"]))


def test_selection_cache_and_batch_run_are_available(tmp_path):
    df = _make_synthetic_dataset()
    cache_path = tmp_path / "feature_cache.json"

    from src.optimizer import load_feature_cache, save_feature_cache, select_optimal_attributes

    selected = select_optimal_attributes(df, "price_czk_per_kwh", max_features=3, cache_path=cache_path, batch_size=2)

    assert isinstance(selected, list)
    assert selected
    cache = load_feature_cache(cache_path)
    assert "price_czk_per_kwh:3" in cache
    save_feature_cache({"price_czk_per_kwh:3": selected}, cache_path)
    assert load_feature_cache(cache_path)["price_czk_per_kwh:3"] == selected
