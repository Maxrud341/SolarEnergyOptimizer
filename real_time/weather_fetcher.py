"""Solar Irradiance and Photovoltaic Power Output Forecast Fetcher.

Fetches hourly atmospheric solar radiation (GHI, DNI) and temperature
from Open-Meteo for the facility GPS coordinates (Liberec / TUL).
Converts physical irradiance into estimated electrical AC generation (kW)
using physical photovoltaic derating and cell temperature models.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import requests

try:
    from real_time.config import (
        LOCATION_LATITUDE,
        LOCATION_LONGITUDE,
        LOCATION_TIMEZONE,
        OPEN_METEO_FORECAST_URL,
        PV_CAPACITY_KWP,
        PV_PERFORMANCE_RATIO,
        PV_TEMP_COEFF_PER_C,
    )
except ImportError:
    from config import (
        LOCATION_LATITUDE,
        LOCATION_LONGITUDE,
        LOCATION_TIMEZONE,
        OPEN_METEO_FORECAST_URL,
        PV_CAPACITY_KWP,
        PV_PERFORMANCE_RATIO,
        PV_TEMP_COEFF_PER_C,
    )

logger = logging.getLogger("WeatherFetcher")


class SolarWeatherFetcher:
    """Manager for fetching weather and forecasting solar generation."""

    def __init__(self) -> None:
        self._cached_df: pd.DataFrame = pd.DataFrame()
        self._last_fetch_time: dt.datetime | None = None

    def fetch_forecast(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch hourly solar radiation and temperature for next 3 days."""
        now = dt.datetime.now()

        # Cache valid for 90 minutes
        if not force_refresh and not self._cached_df.empty and self._last_fetch_time:
            age_min = (now - self._last_fetch_time).total_seconds() / 60.0
            if age_min < 90.0:
                return self._cached_df

        params = {
            "latitude": LOCATION_LATITUDE,
            "longitude": LOCATION_LONGITUDE,
            "hourly": "shortwave_radiation,direct_normal_irradiance,temperature_2m,cloud_cover",
            "forecast_days": 3,
            "timezone": LOCATION_TIMEZONE,
        }

        try:
            resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=8.0)
            if resp.status_code == 200:
                data = resp.json().get("hourly", {})
                times = data.get("time", [])
                ghi = data.get("shortwave_radiation", [])
                dni = data.get("direct_normal_irradiance", [])
                temps = data.get("temperature_2m", [])
                clouds = data.get("cloud_cover", [])

                records: list[dict[str, Any]] = []
                for t, g, d, temp, cl in zip(times, ghi, dni, temps, clouds):
                    g_val = float(g) if g is not None else 0.0
                    temp_val = float(temp) if temp is not None else 15.0

                    # Photovoltaic conversion:
                    # Cell temperature estimate: T_cell = T_amb + 0.03 * GHI
                    t_cell = temp_val + 0.03 * g_val
                    # Thermal derate factor
                    temp_factor = 1.0 + PV_TEMP_COEFF_PER_C * (t_cell - 25.0)
                    temp_factor = max(0.70, min(1.05, temp_factor))

                    # AC output (kW) = kWp * (GHI / 1000) * System_PR * Temp_Factor
                    pv_kw = PV_CAPACITY_KWP * (g_val / 1000.0) * PV_PERFORMANCE_RATIO * temp_factor
                    pv_kw = max(0.0, round(pv_kw, 3))

                    records.append(
                        {
                            "start": pd.to_datetime(t),
                            "ghi_w_m2": g_val,
                            "dni_w_m2": float(d) if d is not None else 0.0,
                            "temp_c": temp_val,
                            "cloud_pct": float(cl) if cl is not None else 0.0,
                            "pv_forecast_kw": pv_kw,
                        }
                    )

                df = pd.DataFrame(records).sort_values("start").reset_index(drop=True)
                self._cached_df = df
                self._last_fetch_time = now
                return df
        except Exception as exc:
            logger.warning("Failed to fetch weather from Open-Meteo: %s", exc)

        return self._cached_df

    def get_pv_forecast_vector(self, start_dt: dt.datetime | None = None, hours: int = 36) -> pd.DataFrame:
        """Return hourly expected PV generation starting from start_dt."""
        df = self.fetch_forecast()
        if df.empty:
            now = start_dt or dt.datetime.now()
            times = [now + dt.timedelta(hours=i) for i in range(hours)]
            return pd.DataFrame({"start": times, "pv_forecast_kw": [0.0] * hours})

        if start_dt is not None:
            start_hour = start_dt.replace(minute=0, second=0, microsecond=0)
            if start_hour.tzinfo is None and df["start"].dt.tz is not None:
                df["start"] = df["start"].dt.tz_localize(None)
            elif start_hour.tzinfo is not None and df["start"].dt.tz is None:
                start_hour = start_hour.tz_localize(None)
            df = df[df["start"] >= start_hour]

        return df.head(hours).reset_index(drop=True)

    def get_pv_forecast_vector_15min(self, start_dt: dt.datetime | None = None, hours: int = 36) -> pd.DataFrame:
        """Return 15-minute resolution expected PV generation starting from start_dt."""
        df = self.fetch_forecast()
        total_slots = hours * 4
        if df.empty:
            now = start_dt or dt.datetime.now()
            times = [now + dt.timedelta(minutes=15 * i) for i in range(total_slots)]
            return pd.DataFrame({"start": times, "pv_forecast_kw": [0.0] * total_slots, "cloud_pct": [50.0] * total_slots})

        # Interpolate 1-hour forecasts smoothly to 15-minute intervals
        df_indexed = df.set_index("start")
        df_15m = df_indexed.resample("15min").interpolate(method="time").reset_index()
        df_15m["pv_forecast_kw"] = df_15m["pv_forecast_kw"].clip(lower=0.0)
        if "cloud_pct" in df_15m.columns:
            df_15m["cloud_pct"] = df_15m["cloud_pct"].clip(lower=0.0, upper=100.0)

        if start_dt is not None:
            start_slot = start_dt.replace(
                minute=(start_dt.minute // 15) * 15, second=0, microsecond=0
            )
            if start_slot.tzinfo is None and df_15m["start"].dt.tz is not None:
                df_15m["start"] = df_15m["start"].dt.tz_localize(None)
            elif start_slot.tzinfo is not None and df_15m["start"].dt.tz is None:
                start_slot = start_slot.tz_localize(None)
            df_15m = df_15m[df_15m["start"] >= start_slot]

        return df_15m.head(total_slots).reset_index(drop=True)


if __name__ == "__main__":
    fetcher = SolarWeatherFetcher()
    forecast = fetcher.fetch_forecast(force_refresh=True)
    print(f"Fetched {len(forecast)} hours of solar forecast.")
    print(forecast[["start", "ghi_w_m2", "temp_c", "cloud_pct", "pv_forecast_kw"]].head(8))
