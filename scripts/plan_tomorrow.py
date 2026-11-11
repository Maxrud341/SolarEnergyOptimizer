#!/usr/bin/env python3
"""Show tomorrow's solar forecast, OTE spot prices, and optimal battery schedule."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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


def get_tomorrow_plan(target_date: dt.date | None = None) -> None:
    now = dt.datetime.now()
    if target_date is None:
        target_date = (now + dt.timedelta(days=1)).date()

    start_dt = dt.datetime.combine(target_date, dt.time.min)

    print("=" * 80)
    print(f"🔮 ПЛАН ОПТИМИЗАЦИИ И АРБИТРАЖА НА: {target_date.strftime('%d.%m.%Y (%A)')}")
    print("=" * 80)

    # 1. Fetch spot prices (15-minute resolution)
    pf = SpotPriceFetcher()
    prices_df = pf.get_price_vector_15min(start_dt=start_dt, hours=24)

    if prices_df.empty or len(prices_df) < 48:
        print(f"\n⚠️  Цены OTE ČR на {target_date.strftime('%d.%m.%Y')} еще не опубликованы.")
        print("   Биржевой аукцион закрывается ежедневно в 13:00, публикация в 13:10–13:15 CET.")
        print("   Запустите скрипт после 13:15.\n")
        return

    # 2. Fetch solar forecast (15-minute interpolated)
    wf = SolarWeatherFetcher()
    pv_df = wf.get_pv_forecast_vector_15min(start_dt=start_dt, hours=24)
    pv_forecast = pv_df["pv_forecast_kw"].tolist() if not pv_df.empty else [0.0] * len(prices_df)

    # 3. Typical household load profile (kW)
    hours_of_day = [r["start"].hour for _, r in prices_df.iterrows()]
    minute_slots = [r["start"].minute // 15 for _, r in prices_df.iterrows()]
    spot_prices = prices_df["price_czk_kwh"].tolist()

    load_forecast: list[float] = []
    for h in hours_of_day:
        if 17 <= h <= 22:
            load_forecast.append(0.80)
        elif 7 <= h <= 16:
            load_forecast.append(0.45)
        else:
            load_forecast.append(0.25)

    # 4. Run Linear Programming Optimizer with Adaptive Risk (15-min intervals)
    opt = LpArbitrageOptimizerRT()
    cloud_forecast = pv_df["cloud_pct"].tolist() if "cloud_pct" in pv_df.columns else None
    schedule = opt.solve_horizon(
        current_soc_kwh=BATTERY_CAPACITY_KWH * 0.20,  # Assume 20% initial floor
        spot_prices=spot_prices,
        pv_forecast_kw=pv_forecast,
        load_forecast_kw=load_forecast,
        hours_of_day=hours_of_day,
        cloud_cover_pct=cloud_forecast,
        dt_hours=0.25,
        minute_slots=minute_slots,
    )

    # Summary Stats
    total_pv_kwh = sum(pv_forecast) * 0.25
    total_load_kwh = sum(load_forecast) * 0.25
    min_price_idx = int(prices_df["price_czk_kwh"].idxmin())
    max_price_idx = int(prices_df["price_czk_kwh"].idxmax())
    min_price_row = prices_df.iloc[min_price_idx]
    max_price_row = prices_df.iloc[max_price_idx]

    print("\n📊 Сводка прогноза (15-минутная сетка MTU OTE ČR):")
    print(f"  ☀️ Ожидаемая солнечная генерация (FVE):   {total_pv_kwh:.2f} кВт·ч")
    print(f"  🏠 Ожидаемое базовое потребление дома:   {total_load_kwh:.2f} кВт·ч")
    print(f"  📉 Самый дешевый слот (спот):            {min_price_row['start'].strftime('%H:%M')} -> {min_price_row['price_czk_kwh']:.2f} Kč/кВт·ч")
    print(f"  📈 Самый дорогой слот (пик):             {max_price_row['start'].strftime('%H:%M')} -> {max_price_row['price_czk_kwh']:.2f} Kč/кВт·ч")
    print(f"  ⚡ Спред спотовой цены:                  {(max_price_row['price_czk_kwh'] - min_price_row['price_czk_kwh']):.2f} Kč/кВт·ч")
    print("-" * 80)

    # Interval Table (Grouped for readability)
    print(f"{'Интервал':^13} | {'Спот (Kč)':^11} | {'Команда инвертора':^26} | {'SoC (%)':^8} | {'Обоснование':^35}")
    print("-" * 115)

    intervals: list[dict[str, Any]] = []
    cur_group: dict[str, Any] | None = None

    for idx, (_, a) in enumerate(sorted(schedule.items())):
        row = prices_df.iloc[idx]
        cmd = a.command
        pwr = a.target_power_w
        reason = a.reason
        soc = a.planned_end_soc_pct
        price = spot_prices[idx]
        t_slot_start = row["start"]
        t_slot_end = t_slot_start + dt.timedelta(minutes=15)

        if cur_group is None:
            cur_group = {
                "start": t_slot_start,
                "end": t_slot_end,
                "cmd": cmd,
                "pwr": pwr,
                "reason": reason,
                "min_price": price,
                "max_price": price,
                "end_soc": soc,
            }
        elif cur_group["cmd"] == cmd and abs(cur_group["pwr"] - pwr) < 100:
            cur_group["end"] = t_slot_end
            cur_group["min_price"] = min(cur_group["min_price"], price)
            cur_group["max_price"] = max(cur_group["max_price"], price)
            cur_group["end_soc"] = soc
        else:
            intervals.append(cur_group)
            cur_group = {
                "start": t_slot_start,
                "end": t_slot_end,
                "cmd": cmd,
                "pwr": pwr,
                "reason": reason,
                "min_price": price,
                "max_price": price,
                "end_soc": soc,
            }
    if cur_group:
        intervals.append(cur_group)

    for inv in intervals:
        s_str = inv["start"].strftime("%H:%M")
        e_str = inv["end"].strftime("%H:%M")
        p_min = inv["min_price"]
        p_max = inv["max_price"]
        p_str = f"{p_min:5.2f}" if p_min == p_max else f"{p_min:4.2f}-{p_max:4.2f}"
        cmd = inv["cmd"]
        pwr = inv["pwr"]
        cmd_str = f"{cmd} ({pwr:4.0f}W)" if pwr > 0 else cmd
        soc = inv["end_soc"]
        reason = inv["reason"]
        print(f"{s_str} - {e_str} | {p_str:>9s}   | {cmd_str:26s} | {soc:6.1f}% | {reason}")

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
