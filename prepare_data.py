from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
BATTERY_CAPACITY_KWH = 0.5


def load_and_prepare_maly() -> pd.DataFrame:
    """Load raw home energy data and convert it to hourly kWh values."""
    df = pd.read_csv(DATA_DIR / "Maly.csv", sep=';')
    df["datetime"] = pd.to_datetime(df["DatumCas"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    df["hour"] = df["datetime"].dt.floor("h")

    # Power is in W, convert to kWh for each 5-minute interval.
    df["consumption_kwh"] = df["P_HOME"].fillna(0).abs() * 5 / 60 / 1000
    df["production_kwh"] = df["P_PV"].fillna(0).clip(lower=0) * 5 / 60 / 1000
    df["grid_import_kwh"] = df["P_GRID"].fillna(0).clip(lower=0) * 5 / 60 / 1000
    df["grid_export_kwh"] = df["P_GRID"].fillna(0).clip(upper=0).abs() * 5 / 60 / 1000
    df["battery_kwh"] = df["SoC"].fillna(0).clip(0, 100) / 100 * BATTERY_CAPACITY_KWH

    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            datetime=("hour", "first"),
            consumption_kwh=("consumption_kwh", "sum"),
            production_kwh=("production_kwh", "sum"),
            grid_import_kwh=("grid_import_kwh", "sum"),
            grid_export_kwh=("grid_export_kwh", "sum"),
            battery_kwh=("battery_kwh", "last"),
            soc_pct=("SoC", "last"),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return hourly


def load_and_prepare_weather() -> pd.DataFrame:
    """Load weather forecast and keep only useful meteorological features."""
    df = pd.read_csv(DATA_DIR / "predpoved.csv", sep=';')
    df["datetime"] = pd.to_datetime(df["DatumCas"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
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

    return (
        df[["datetime", "temperature_c", "cloud_cover_pct", "wind_speed_ms", "wind_dir_deg", "humidity_pct"]]
        .drop_duplicates()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_and_prepare_prices() -> pd.DataFrame:
    """Load market prices and keep only CZK/kWh for the optimization model."""
    df = pd.read_csv(DATA_DIR / "ceny.csv", sep=';')
    df["datetime"] = pd.to_datetime(df["DatumCasOTE"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.floor("h")

    df["price_czk_per_kwh"] = df["CenaCZK"].astype(float)

    return (
        df[["datetime", "price_czk_per_kwh"]]
        .drop_duplicates()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_and_prepare_solar() -> pd.DataFrame:
    """Load solar production forecast and convert to kWh per hour."""
    df = pd.read_csv(DATA_DIR / "sun.csv", sep=';')
    df["datetime"] = pd.to_datetime(df["DatumCas"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.floor("h")

    # `Vyroba` in raw file looks like Wh for the hour, so divide by 1000 to get kWh.
    df["pv_generation_kwh"] = df["Vyroba"].fillna(0).astype(float) / 1000.0

    solar = (
        df.groupby("datetime", as_index=False)
        .agg(pv_generation_kwh=("pv_generation_kwh", "sum"))
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return solar


def merge_all_data() -> pd.DataFrame:
    """Merge all relevant data to a single hourly feature set for trading optimization."""
    maly = load_and_prepare_maly()
    weather = load_and_prepare_weather()
    prices = load_and_prepare_prices()
    solar = load_and_prepare_solar()

    # Use only the common time overlap across all data sources.
    maly = maly.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    weather = weather.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    prices = prices.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    solar = solar.drop_duplicates(subset=["datetime"]).sort_values("datetime")

    common_start = max(maly["datetime"].min(), weather["datetime"].min(), prices["datetime"].min(), solar["datetime"].min())
    common_end = min(maly["datetime"].max(), weather["datetime"].max(), prices["datetime"].max(), solar["datetime"].max())
    full_index = pd.date_range(start=common_start, end=common_end, freq="h")

    maly = maly.set_index("datetime").reindex(full_index).ffill().bfill().reset_index().rename(columns={"index": "datetime"})
    weather = weather.set_index("datetime").reindex(full_index).ffill().bfill().reset_index().rename(columns={"index": "datetime"})
    prices = prices.set_index("datetime").reindex(full_index).ffill().bfill().reset_index().rename(columns={"index": "datetime"})
    solar = solar.set_index("datetime").reindex(full_index).ffill().bfill().reset_index().rename(columns={"index": "datetime"})

    merged = maly.merge(prices, on="datetime", how="left")
    merged = merged.merge(weather, on="datetime", how="left")
    merged = merged.merge(solar, on="datetime", how="left")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["datetime"]).copy()

    # For optimization tasks, keep only the key variables and remove purely technical columns.
    keep_columns = [
        "datetime",
        "consumption_kwh",
        "production_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "battery_kwh",
        "soc_pct",
        "price_czk_per_kwh",
        "temperature_c",
        "cloud_cover_pct",
        "wind_speed_ms",
        "wind_dir_deg",
        "humidity_pct",
        "pv_generation_kwh",
    ]

    merged = merged[[col for col in keep_columns if col in merged.columns]].copy()

    # Remove starting rows with no price or weather, if this dataset is incomplete.
    merged = merged.dropna(subset=["price_czk_per_kwh"]).copy()

    return merged


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, sep=';', index=False)
    print(f"Saved merged dataset: {output_path}")


def main() -> None:
    print("Preparing merged dataset for energy trading optimization...\n")

    combined = merge_all_data()

    print(f"Rows: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    print(combined.head(10).to_string(index=False))

    output_path = DATA_DIR / "combined_market_data.csv"
    save_dataset(combined, output_path)

    print("\nSummary:")
    print(combined[["consumption_kwh", "production_kwh", "battery_kwh", "price_czk_per_kwh"]].describe().to_string())


if __name__ == "__main__":
    main()
