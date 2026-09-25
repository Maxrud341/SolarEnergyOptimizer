import pandas as pd

df = pd.read_csv('data/datasource.csv', sep=';')
df['date'] = df['datetime'].str[:10]
dates = df['date'].value_counts().sort_index()

print("=" * 70)
print("АНАЛИЗ ДАННЫХ")
print("=" * 70)

full_days = dates[dates == 288]
partial_days = dates[dates != 288]

print(f"\nПолные дни (288 точек): {len(full_days)}")
print(f"Неполные дни: {len(partial_days)}")

if len(full_days) > 0:
    first_full = full_days.index[0]
    print(f"\nПервый полный день: {first_full}")
    print(f"  Это будет идеален для просмотра '1Д' (дневной вид)")
    
    # Показать данные этого дня
    day_data = df[df['date'] == first_full]
    print(f"\n  Данные за день {first_full}:")
    print(f"    Первая запись: {day_data.iloc[0]['datetime']}")
    print(f"    Последняя запись: {day_data.iloc[-1]['datetime']}")
    print(f"    Всего записей: {len(day_data)}")
    
    print(f"\n  Примеры данных:")
    print(day_data[['datetime', 'consumption_kwh', 'production_kwh']].head(10).to_string(index=False))
