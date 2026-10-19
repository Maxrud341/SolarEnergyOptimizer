import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('trading_log_arbitrage.csv', sep=';')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')

daily = df.groupby('date').agg({
    'cost_buy_czk': 'sum',
    'revenue_sell_czk': 'sum',
    'degradation_cost_czk': 'sum',
    'net_profit_czk': 'sum',
    'load_kwh': 'sum',
    'pv_kwh': 'sum',
    'buy_kwh': 'sum',
    'sell_kwh': 'sum',
    'spot_price_czk': 'mean',
    'p_buy': 'mean'
}).reset_index()

total_cost = daily['cost_buy_czk'].sum()
top15 = daily.sort_values('cost_buy_czk', ascending=False).head(15).reset_index(drop=True)
top15_cost = top15['cost_buy_czk'].sum()

print(f"Всего дней в логе: {len(daily)}")
print(f"Общие расходы на покупку за все 488 дней: {total_cost:,.2f} CZK")
print(f"Расходы за ТОП-15 дней: {top15_cost:,.2f} CZK ({top15_cost / total_cost * 100:.1f}% от всех расходов)")
print()
print("| № | Дата | Расход на покупку | Чистый результат | Куплено из сети | Потребление дома | Генерация СЭС | Ср. спот | Ср. цена покупки |")
print("|---|------|-------------------|------------------|-----------------|------------------|---------------|----------|------------------|")
for i, r in top15.iterrows():
    print(f"| {i+1} | {r['date']} | **{r['cost_buy_czk']:.2f} CZK** | {r['net_profit_czk']:.2f} CZK | {r['buy_kwh']:.2f} kWh | {r['load_kwh']:.2f} kWh | {r['pv_kwh']:.2f} kWh | {r['spot_price_czk']:.2f} CZK | {r['p_buy']:.2f} CZK |")
