"""Финальная проверка системы"""
import sys
import pandas as pd
from pathlib import Path

print("=" * 70)
print("ФИНАЛЬНАЯ ПРОВЕРКА ENERGY DATA VISUALIZER")
print("=" * 70)

# 1. Проверка датасета
print("\n✓ 1. ПРОВЕРКА ДАТАСЕТА")
try:
    df = pd.read_csv(Path("data") / "datasource.csv", sep=";")
    print(f"   Загружено: {len(df)} строк × {len(df.columns)} столбцов")
    print(f"   Колонки: {', '.join(df.columns)}")
    print(f"   Период: {df['datetime'].min()} до {df['datetime'].max()}")
    
    # Проверить полные дни
    df['date'] = df['datetime'].str[:10]
    full_days = (df['date'].value_counts() == 288).sum()
    print(f"   Полные дни (288 точек): {full_days}")
    print("   ✓ Датасет готов")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    sys.exit(1)

# 2. Проверка визуализатора
print("\n✓ 2. ПРОВЕРКА ВИЗУАЛИЗАТОРА")
try:
    from visualizer import Visualizer
    print("   ✓ Класс Visualizer успешно импортирован")
    print("   ✓ Все методы присутствуют:")
    methods = ['_setup_style', '_setup_gui', '_setup_controls', '_setup_plot', 
               '_init_defaults', 'get_dates', 'set_days', 'next_day', 'prev_day', 
               'update_plot', 'save']
    for m in methods:
        if hasattr(Visualizer, m):
            print(f"     - {m}")
        else:
            print(f"     ✗ ОТСУТСТВУЕТ: {m}")
            sys.exit(1)
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    sys.exit(1)

# 3. Проверка create_datasource
print("\n✓ 3. ПРОВЕРКА create_datasource.py")
try:
    from create_datasource import load_source_data, process_to_5min
    print("   ✓ Функции импортированы успешно")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    sys.exit(1)

# 4. Статистика
print("\n✓ 4. СТАТИСТИКА ДАННЫХ")
print(f"   Потребление (кВт·ч):")
print(f"     Min: {df['consumption_kwh'].min():.6f}")
print(f"     Max: {df['consumption_kwh'].max():.6f}")
print(f"     Mean: {df['consumption_kwh'].mean():.6f}")
print(f"   Производство (кВт·ч):")
print(f"     Min: {df['production_kwh'].min():.6f}")
print(f"     Max: {df['production_kwh'].max():.6f}")
print(f"     Mean: {df['production_kwh'].mean():.6f}")
print(f"   Батарея (%):")
print(f"     Min: {df['battery_soc'].min():.1f}")
print(f"     Max: {df['battery_soc'].max():.1f}")
print(f"     Mean: {df['battery_soc'].mean():.1f}")

print("\n" + "=" * 70)
print("✓ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
print("=" * 70)
print("\nГотово к запуску:")
print("  python visualizer.py")
print("\n" + "=" * 70)
