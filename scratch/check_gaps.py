import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from prepare_data import load_and_prepare_maly

maly = load_and_prepare_maly(1)
maly['dt_diff'] = maly['datetime'].diff()
gaps_2025 = maly[(maly['dt_diff'] > pd.Timedelta(hours=2)) & (maly['datetime'] < '2025-07-01')]

print(f"Всего пропусков > 2 часов в мае-июне 2025: {len(gaps_2025)}")
for idx, r in gaps_2025.iterrows():
    prev = maly.loc[idx - 1]
    dur = r['dt_diff'].total_seconds() / 3600.0
    print(f"Пропуск: с {prev['datetime']} до {r['datetime']} ({dur:.1f} ч.) | Зафиксированное потребление: {prev['consumption_kwh']:.3f} kWh")

print("\n--- Проверка топ-дней мая-июня 2025 на непрерывность данных в Maly.csv ---")
for d_str in ['2025-05-25', '2025-05-26', '2025-05-29', '2025-06-02', '2025-06-06', '2025-06-08', '2025-06-10', '2025-06-11', '2025-06-12', '2025-06-14', '2025-06-15']:
    sub = maly[maly['datetime'].dt.strftime('%Y-%m-%d') == d_str]
    hours_count = len(sub)
    load_sum = sub['consumption_kwh'].sum()
    pv_sum = sub['production_kwh'].sum()
    print(f"Дата: {d_str} | Часов в сырых данных: {hours_count:2d}/24 | Нагрузка: {load_sum:5.2f} kWh | СЭС: {pv_sum:5.2f} kWh")
