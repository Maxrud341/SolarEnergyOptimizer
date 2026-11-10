#!/usr/bin/env python3
"""Show tomorrow's solar forecast, OTE spot prices, and optimal battery schedule."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

# Ensure this directory is in sys.path
this_dir = Path(__file__).resolve().parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))
parent_dir = this_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from real_time.config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        EXPORT_FEE_CZK_PER_KWH,
        IMPORT_FEE_CZK_PER_KWH,
        VAT_RATE,
    )
    from real_time.lp_optimizer import LpArbitrageOptimizerRT
    from real_time.price_fetcher import SpotPriceFetcher
    from real_time.weather_fetcher import SolarWeatherFetcher
except ImportError:
    from config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        EXPORT_FEE_CZK_PER_KWH,
        IMPORT_FEE_CZK_PER_KWH,
        VAT_RATE,
    )
    from lp_optimizer import LpArbitrageOptimizerRT
    from price_fetcher import SpotPriceFetcher
    from weather_fetcher import SolarWeatherFetcher


def get_tomorrow_plan(target_date: dt.date | None = None) -> None:
    now = dt.datetime.now()
    if target_date is None:
        target_date = (now + dt.timedelta(days=1)).date()

    start_dt = dt.datetime.combine(target_date, dt.time.min)

    print("=" * 80)
    print(f"🔮 ПЛАН ОПТИМИЗАЦИИ И АРБИТРАЖА НА: {target_date.strftime('%d.%m.%Y (%A)')}")
    print("=" * 80)

    # 1. Fetch spot prices
    pf = SpotPriceFetcher()
    prices_df = pf.get_hourly_price_vector(start_dt=start_dt, hours=24)

    if prices_df.empty or len(prices_df) < 12:
        print(f"\n⚠️  Цены OTE ČR на {target_date.strftime('%d.%m.%Y')} еще не опубликованы.")
        print("   Биржевой аукцион закрывается ежедневно в 13:00, публикация в 13:10–13:15 CET.")
        print("   Запустите скрипт после 13:15.\n")
        return

    # 2. Fetch solar forecast
    wf = SolarWeatherFetcher()
    pv_df = wf.get_pv_forecast_vector(start_dt=start_dt, hours=24)
    pv_forecast = pv_df["pv_forecast_kw"].tolist() if not pv_df.empty else [0.0] * 24

    # 3. Typical household load profile (kW)
    hours_of_day = [r["start"].hour for _, r in prices_df.iterrows()]
    spot_prices = prices_df["price_czk_kwh"].tolist()

    load_forecast: list[float] = []
    for h in hours_of_day:
        if 17 <= h <= 22:
            load_forecast.append(0.80)
        elif 7 <= h <= 16:
            load_forecast.append(0.45)
        else:
            load_forecast.append(0.25)

    # 4. Run Linear Programming Optimizer with Adaptive Risk
    opt = LpArbitrageOptimizerRT()
    cloud_forecast = pv_df["cloud_pct"].tolist() if "cloud_pct" in pv_df.columns else None
    schedule = opt.solve_horizon(
        current_soc_kwh=BATTERY_CAPACITY_KWH * 0.20,  # Assume 20% initial floor
        spot_prices=spot_prices,
        pv_forecast_kw=pv_forecast,
        load_forecast_kw=load_forecast,
        hours_of_day=hours_of_day,
        cloud_cover_pct=cloud_forecast,
    )

    # Summary Stats
    total_pv_kwh = sum(pv_forecast)
    total_load_kwh = sum(load_forecast)
    min_price_idx = int(prices_df["price_czk_kwh"].idxmin())
    max_price_idx = int(prices_df["price_czk_kwh"].idxmax())
    min_price_row = prices_df.iloc[min_price_idx]
    max_price_row = prices_df.iloc[max_price_idx]

    print("\n📊 Сводка прогноза:")
    print(f"  ☀️ Ожидаемая солнечная генерация (FVE):   {total_pv_kwh:.2f} кВт·ч")
    print(f"  🏠 Ожидаемое базовое потребление дома:   {total_load_kwh:.2f} кВт·ч")
    print(f"  📉 Самый дешевый час (спот):             {min_price_row['start'].strftime('%H:%M')} -> {min_price_row['price_czk_kwh']:.2f} Kč/кВт·ч")
    print(f"  📈 Самый дорогой час (пик):              {max_price_row['start'].strftime('%H:%M')} -> {max_price_row['price_czk_kwh']:.2f} Kč/кВт·ч")
    print(f"  ⚡ Спред спотовой цены:                  {(max_price_row['price_czk_kwh'] - min_price_row['price_czk_kwh']):.2f} Kč/кВт·ч")
    print("-" * 80)

    # Hourly Table
    print(f"{'Час':^5} | {'Спот (Kč)':^9} | {'FVE (кВт)':^9} | {'Дом (кВт)':^9} | {'Команда инвертора':^25} | {'SoC (%)':^8} | {'Обоснование':^30}")
    print("-" * 115)

    for h, a in sorted(schedule.items()):
        price = spot_prices[h]
        pv = pv_forecast[h]
        load = load_forecast[h]
        cmd_str = f"{a.command} ({a.target_power_w:4.0f}W)" if a.target_power_w > 0 else a.command
        print(f"{h:02d}:00 | {price:7.2f}   | {pv:7.2f}   | {load:7.2f}   | {cmd_str:25s} | {a.planned_end_soc_pct:6.1f}% | {a.reason}")

    print("-" * 115)
    print("💡 Пояснение стратегии:")
    print("  • Ночью (дешевые часы): алгоритм выполняет экономически обоснованный дозаряд батареи из сети,")
    print("    если будущие утренний/вечерний пики окупают покупку, потери КПД инвертора и деградацию.")
    print("  • Днем (11:00-14:00): зарядка из сети запрещена; батарея сохраняет буфер емкости под бесплатное солнце.")
    print("  • В пиковые часы (07:00-10:00 и 16:00-20:00): батарея полностью закрывает потребности дома, экономя дорогую электроэнергию.")
    print("=" * 80)


if __name__ == "__main__":
    target = None
    if len(sys.argv) > 1:
        try:
            target = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            pass
    get_tomorrow_plan(target)
