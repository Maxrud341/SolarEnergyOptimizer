"""Czech Day-Ahead Electricity Spot Market Price Fetcher.

Fetches 15-minute and hourly spot prices from OTE ČR via Spotovky.cz API (v2)
with fallback to Fraunhofer Energy-Charts.
Caches prices in memory with automatic updates when tomorrow's auction results
are published (daily at ~13:15 CET).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import requests

try:
    from real_time.config import (
        ENERGY_CHARTS_URL,
        SPOTOVKY_TODAY_URL,
        SPOTOVKY_TOMORROW_URL,
    )
except ImportError:
    from config import (
        ENERGY_CHARTS_URL,
        SPOTOVKY_TODAY_URL,
        SPOTOVKY_TOMORROW_URL,
    )

logger = logging.getLogger("PriceFetcher")


class SpotPriceFetcher:
    """Manager for fetching and caching Czech spot electricity prices."""

    def __init__(self) -> None:
        self._cached_df: pd.DataFrame = pd.DataFrame()
        self._last_fetch_time: dt.datetime | None = None
        self._has_tomorrow: bool = False

    def fetch_prices(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch today and tomorrow's spot prices, caching results for 1 hour."""
        now = dt.datetime.now()

        # Cache valid for 30 minutes unless tomorrow is missing and it's after 13:10
        if not force_refresh and not self._cached_df.empty and self._last_fetch_time:
            age_min = (now - self._last_fetch_time).total_seconds() / 60.0
            if age_min < 30.0:
                # If we already have tomorrow's prices or it's before 13:15, return cached
                if self._has_tomorrow or now.hour < 13:
                    return self._cached_df

        records: list[dict[str, Any]] = []

        # 1. Fetch Today
        try:
            resp_today = requests.get(SPOTOVKY_TODAY_URL, timeout=8.0)
            if resp_today.status_code == 200:
                data_today = resp_today.json()
                for slot in data_today.get("slots", []):
                    # slot["start"] is ISO 8601 string, e.g. "2026-10-26T18:00:00+01:00"
                    records.append(
                        {
                            "start": pd.to_datetime(slot["start"]),
                            "price_czk_kwh": float(slot["czk"]),
                            "price_eur_mwh": float(slot["eur"]),
                        }
                    )
        except Exception as exc:
            logger.warning("Could not fetch today's spot prices: %s", exc)

        # 2. Fetch Tomorrow (usually available after 13:05 CET)
        has_tomorrow = False
        try:
            resp_tom = requests.get(SPOTOVKY_TOMORROW_URL, timeout=8.0)
            if resp_tom.status_code == 200:
                data_tom = resp_tom.json()
                tom_slots = data_tom.get("slots", [])
                if tom_slots:
                    has_tomorrow = True
                    for slot in tom_slots:
                        records.append(
                            {
                                "start": pd.to_datetime(slot["start"]),
                                "price_czk_kwh": float(slot["czk"]),
                                "price_eur_mwh": float(slot["eur"]),
                            }
                        )
        except Exception as exc:
            logger.warning("Tomorrow's prices not available yet or request failed: %s", exc)

        # 3. Fallback to Fraunhofer Energy-Charts if spotovky returned empty
        if not records:
            records = self._fetch_energy_charts_fallback()

        if records:
            df = pd.DataFrame(records).drop_duplicates(subset=["start"]).sort_values("start").reset_index(drop=True)
            self._cached_df = df
            self._last_fetch_time = now
            self._has_tomorrow = has_tomorrow
            return df

        # If all fetches failed, return last cached dataframe (even if stale)
        return self._cached_df

    def _fetch_energy_charts_fallback(self) -> list[dict[str, Any]]:
        """Fallback to Fraunhofer Energy-Charts API in EUR/MWh."""
        records: list[dict[str, Any]] = []
        try:
            resp = requests.get(ENERGY_CHARTS_URL, timeout=8.0)
            if resp.status_code == 200:
                d = resp.json()
                timestamps = d.get("unix_seconds", [])
                prices_eur = d.get("price", [])
                eur_to_czk = 25.0  # nominal baseline exchange rate
                for ts, p_eur in zip(timestamps, prices_eur):
                    start_dt = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
                    p_czk_kwh = (p_eur * eur_to_czk) / 1000.0
                    records.append(
                        {
                            "start": pd.to_datetime(start_dt),
                            "price_czk_kwh": round(p_czk_kwh, 4),
                            "price_eur_mwh": round(p_eur, 2),
                        }
                    )
        except Exception as exc:
            logger.error("Energy-Charts fallback also failed: %s", exc)
        return records

    def get_hourly_price_vector(self, start_dt: dt.datetime | None = None, hours: int = 36) -> pd.DataFrame:
        """Return an hourly price profile starting from start_dt."""
        df = self.fetch_prices()
        if df.empty:
            # Fallback dummy price profile if offline
            now = start_dt or dt.datetime.now()
            times = [now + dt.timedelta(hours=i) for i in range(hours)]
            return pd.DataFrame({"start": times, "price_czk_kwh": [3.50] * hours})

        # Resample to hourly mean
        df_hourly = (
            df.set_index("start")
            .resample("1h")
            .agg({"price_czk_kwh": "mean", "price_eur_mwh": "mean"})
            .reset_index()
        )

        if start_dt is not None:
            # Localize / align timestamp comparison
            if start_dt.tzinfo is None and df_hourly["start"].dt.tz is not None:
                df_hourly["start"] = df_hourly["start"].dt.tz_localize(None)
            df_hourly = df_hourly[df_hourly["start"] >= start_dt]

        return df_hourly.head(hours).reset_index(drop=True)


if __name__ == "__main__":
    fetcher = SpotPriceFetcher()
    prices = fetcher.fetch_prices(force_refresh=True)
    print(f"Fetched {len(prices)} slots. Tomorrow available: {fetcher._has_tomorrow}")
    print(prices.head(4))
    print(prices.tail(4))
