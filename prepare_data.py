from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ID_ZDROJE = 1
BATTERY_CAPACITY_KWH = 10.0  # Empirically measured capacity (~9.6-10.0 kWh usable)
MAX_HOME_POWER_W = 17500.0   # Standard 3x25A breaker limit (~17.25 kW)
MAX_PHASE_POWER_W = 10000.0  # Max physical power per phase breaker (~25-32A limit)
MAX_PV_POWER_W = 10000.0     # 8.7 kW PV array limit
MAX_BAT_POWER_W = 6000.0     # Inverter max charge/discharge power (~5.5 kW)
MAX_GRID_POWER_W = 17500.0   # 3x25A connection limit


def load_and_prepare_maly(source_id: int = DEFAULT_ID_ZDROJE) -> pd.DataFrame:
    """Load raw home energy telemetry, clean outliers, and aggregate to hourly kWh per phase."""
    df = pd.read_csv(DATA_DIR / "Maly.csv", sep=';')
    if "ID_ZDROJE" in df.columns:
        df = df[df["ID_ZDROJE"] == source_id].copy()

    df["datetime"] = pd.to_datetime(df["DatumCas"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Telemetry outlier filtering: clean sensor communication glitch spikes (> physical limits)
    # Using linear interpolation preserves continuity without injecting false 16-bit overflow loads
    for phase_col in ("P_HOME_L1", "P_HOME_L2", "P_HOME_L3"):
        if phase_col in df.columns:
            df.loc[df[phase_col] < -MAX_PHASE_POWER_W, phase_col] = np.nan
            df[phase_col] = df[phase_col].interpolate(method="linear").bfill().ffill().clip(lower=-MAX_PHASE_POWER_W, upper=0.0)

    df.loc[df["P_HOME"] < -MAX_HOME_POWER_W, "P_HOME"] = np.nan
    df["P_HOME"] = df["P_HOME"].interpolate(method="linear").bfill().ffill().clip(lower=-MAX_HOME_POWER_W, upper=0.0)

    # If phase telemetry is available, ensure total matches sum of phases
    if {"P_HOME_L1", "P_HOME_L2", "P_HOME_L3"}.issubset(df.columns):
        df["P_HOME"] = df["P_HOME_L1"] + df["P_HOME_L2"] + df["P_HOME_L3"]

    df.loc[df["P_PV"] > MAX_PV_POWER_W, "P_PV"] = np.nan
    df["P_PV"] = df["P_PV"].interpolate(method="linear").bfill().ffill().clip(lower=0.0, upper=MAX_PV_POWER_W)

    df.loc[df["P_GRID"].abs() > MAX_GRID_POWER_W, "P_GRID"] = np.nan
    df["P_GRID"] = df["P_GRID"].interpolate(method="linear").bfill().ffill().clip(lower=-MAX_GRID_POWER_W, upper=MAX_GRID_POWER_W)

    df.loc[df["P_BAT"].abs() > MAX_BAT_POWER_W, "P_BAT"] = np.nan
    df["P_BAT"] = df["P_BAT"].interpolate(method="linear").bfill().ffill().clip(lower=-MAX_BAT_POWER_W, upper=MAX_BAT_POWER_W)

    df["hour"] = df["datetime"].dt.floor("h")

    # 5-minute power (W) to energy (kWh): E = P * (5/60) / 1000
    # In Maly.csv:
    # P_HOME / P_HOME_LX <= 0 is household consumption (load)
    if "P_HOME_L1" in df.columns:
        df["consumption_l1_kwh"] = df["P_HOME_L1"].abs() * 5.0 / 60.0 / 1000.0
        df["consumption_l2_kwh"] = df["P_HOME_L2"].abs() * 5.0 / 60.0 / 1000.0
        df["consumption_l3_kwh"] = df["P_HOME_L3"].abs() * 5.0 / 60.0 / 1000.0
        df["consumption_kwh"] = df["consumption_l1_kwh"] + df["consumption_l2_kwh"] + df["consumption_l3_kwh"]
    else:
        df["consumption_kwh"] = df["P_HOME"].abs() * 5.0 / 60.0 / 1000.0
        df["consumption_l1_kwh"] = df["consumption_kwh"] / 3.0
        df["consumption_l2_kwh"] = df["consumption_kwh"] / 3.0
        df["consumption_l3_kwh"] = df["consumption_kwh"] / 3.0

    # P_PV >= 0 is solar production
    df["production_kwh"] = df["P_PV"] * 5.0 / 60.0 / 1000.0
    # Physical sign convention confirmed with BUY_T / SELL_T counters:
    # P_GRID > 0 -> energy exported to grid (SELL)
    # P_GRID < 0 -> energy imported from grid (BUY)
    df["grid_export_kwh"] = df["P_GRID"].clip(lower=0.0) * 5.0 / 60.0 / 1000.0
    df["grid_import_kwh"] = df["P_GRID"].clip(upper=0.0).abs() * 5.0 / 60.0 / 1000.0

    # Battery state of charge (SoC %) to stored kWh
    df["soc_pct"] = df["SoC"].fillna(0.0).clip(0.0, 100.0)
    df["battery_kwh"] = df["soc_pct"] / 100.0 * BATTERY_CAPACITY_KWH

    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            datetime=("hour", "first"),
            consumption_kwh=("consumption_kwh", "sum"),
            consumption_l1_kwh=("consumption_l1_kwh", "sum"),
            consumption_l2_kwh=("consumption_l2_kwh", "sum"),
            consumption_l3_kwh=("consumption_l3_kwh", "sum"),
            production_kwh=("production_kwh", "sum"),
            grid_import_kwh=("grid_import_kwh", "sum"),
            grid_export_kwh=("grid_export_kwh", "sum"),
            battery_kwh=("battery_kwh", "last"),
            soc_pct=("soc_pct", "last"),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    hourly["has_telemetry"] = True
    return hourly


def load_and_prepare_weather(source_id: int = DEFAULT_ID_ZDROJE) -> pd.DataFrame:
    """Load weather forecast for target installation and standardize columns."""
    df = pd.read_csv(DATA_DIR / "predpoved.csv", sep=';')
    if "ID_ZDROJE" in df.columns:
        df = df[df["ID_ZDROJE"] == source_id].copy()

    df["datetime"] = pd.to_datetime(df["DatumCas"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.floor("h")

    df = df.rename(
        columns={
            "Teplota": "temperature_c",
            "Oblacnost": "cloud_cover_pct",
            "Vitr": "wind_speed_ms",
            "Smer": "wind_dir_deg",
            "Vlhkost": "humidity_pct",
        }
    )

    cols = ["datetime", "temperature_c", "cloud_cover_pct", "wind_speed_ms", "wind_dir_deg", "humidity_pct"]
    return df[cols].drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def load_and_prepare_prices() -> pd.DataFrame:
    """Load market prices and keep CZK/kWh and EUR/MWh for optimization."""
    df = pd.read_csv(DATA_DIR / "ceny.csv", sep=';')
    df["datetime"] = pd.to_datetime(df["DatumCasOTE"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.floor("h")

    df["price_czk_per_kwh"] = df["CenaCZK"].astype(float)
    df["price_eur_per_mwh"] = df["CenaEUR"].astype(float)

    cols = ["datetime", "price_czk_per_kwh", "price_eur_per_mwh"]
    return df[cols].drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def load_and_prepare_solar(source_id: int = DEFAULT_ID_ZDROJE) -> pd.DataFrame:
    """Load solar production forecast for target installation (Wh -> kWh/hour)."""
    df = pd.read_csv(DATA_DIR / "sun.csv", sep=';')
    if "ID_ZDROJE" in df.columns:
        df = df[df["ID_ZDROJE"] == source_id].copy()

    df["datetime"] = pd.to_datetime(df["DatumCas"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.floor("h")

    # Vyroba is in Wh per hour; convert to kWh
    df["pv_forecast_kwh"] = df["Vyroba"].fillna(0.0).clip(lower=0.0).astype(float) / 1000.0

    solar = (
        df.groupby("datetime", as_index=False)
        .agg(pv_forecast_kwh=("pv_forecast_kwh", "mean"))
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return solar


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclical time, and meteorological features for statistical modeling."""
    frame = df.copy()
    timestamps = pd.to_datetime(frame["datetime"])

    hour = timestamps.dt.hour.to_numpy(dtype=float)
    dayofweek = timestamps.dt.dayofweek.to_numpy(dtype=float)
    month = timestamps.dt.month.to_numpy(dtype=float)
    dayofyear = timestamps.dt.dayofyear.to_numpy(dtype=float)

    frame["hour"] = hour.astype(int)
    frame["dayofweek"] = dayofweek.astype(int)
    frame["month"] = month.astype(int)
    frame["is_weekend"] = (dayofweek >= 5).astype(float)

    # Cyclical sin/cos encodings
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["weekday_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    frame["weekday_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    frame["day_of_year_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.25)
    frame["day_of_year_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.25)

    # Sun elevation proxy: daytime flag (rough estimate 6:00 to 20:00)
    frame["is_daytime"] = ((hour >= 5) & (hour <= 21)).astype(float)

    return frame


def merge_all_data(source_id: int = DEFAULT_ID_ZDROJE) -> pd.DataFrame:
    """Merge all relevant data to a unified hourly feature set for trading optimization."""
    maly = load_and_prepare_maly(source_id)
    weather = load_and_prepare_weather(source_id)
    prices = load_and_prepare_prices()
    solar = load_and_prepare_solar(source_id)

    # Find common overlapping date range
    common_start = max(maly["datetime"].min(), weather["datetime"].min(), prices["datetime"].min(), solar["datetime"].min())
    common_end = min(maly["datetime"].max(), weather["datetime"].max(), prices["datetime"].max(), solar["datetime"].max())
    full_index = pd.date_range(start=common_start, end=common_end, freq="h")

    # Reindex maly WITHOUT forward-filling missing telemetry into fake constant values
    maly_reindexed = maly.set_index("datetime").reindex(full_index)
    maly_reindexed["has_telemetry"] = maly_reindexed["has_telemetry"].fillna(False).astype(bool)
    maly = maly_reindexed.reset_index().rename(columns={"index": "datetime"})

    # Weather, prices, and solar forecasts can be aligned
    weather = weather.set_index("datetime").reindex(full_index).ffill().bfill().reset_index().rename(columns={"index": "datetime"})
    prices = prices.set_index("datetime").reindex(full_index).ffill().bfill().reset_index().rename(columns={"index": "datetime"})
    solar = solar.set_index("datetime").reindex(full_index).fillna(0.0).reset_index().rename(columns={"index": "datetime"})

    merged = maly.merge(prices, on="datetime", how="left")
    merged = merged.merge(weather, on="datetime", how="left")
    merged = merged.merge(solar, on="datetime", how="left")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["datetime"]).copy()

    # Drop any remaining unaligned price rows
    merged = merged.dropna(subset=["price_czk_per_kwh"]).reset_index(drop=True)

    # Add engineered features (cyclical time, calendar)
    merged = add_engineered_features(merged)

    return merged


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=';', index=False)
    print(f"Saved merged dataset: {output_path}")


def main() -> None:
    print("Preparing cleaned merged dataset for energy trading optimization...")
    print(f"Selected installation ID_ZDROJE: {DEFAULT_ID_ZDROJE}")
    print(f"Calibrated battery capacity: {BATTERY_CAPACITY_KWH} kWh")

    combined = merge_all_data()

    print(f"\nTotal rows: {len(combined)}")
    print(f"Date span: {combined['datetime'].min()} -> {combined['datetime'].max()}")
    print(f"Columns ({len(combined.columns)}): {list(combined.columns)}")

    output_path = DATA_DIR / "combined_market_data.csv"
    save_dataset(combined, output_path)

    print("\nSummary statistics of key metrics:")
    key_cols = [
        "consumption_kwh",
        "consumption_l1_kwh",
        "consumption_l2_kwh",
        "consumption_l3_kwh",
        "production_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "battery_kwh",
        "price_czk_per_kwh",
        "pv_forecast_kwh",
    ]
    print(combined[key_cols].describe().to_string())


if __name__ == "__main__":
    main()
