"""Update monthly arbitrage benchmark files from trading_log_arbitrage.csv."""

from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def generate_arbitrage_benchmark(
    trading_log_path: str = "trading_log_arbitrage.csv",
    v3_benchmark_path: str = "benchmarks/monthly_benchmark_results.csv",
    out_csv: str = "benchmarks/monthly_arbitrage_benchmark.csv",
    out_xlsx: str = "benchmarks/monthly_arbitrage_benchmark.xlsx",
) -> pd.DataFrame:
    log_p = Path(trading_log_path)
    if not log_p.exists():
        raise FileNotFoundError(f"Trading log not found: {log_p}")

    df_log = pd.read_csv(log_p, sep=";")
    df_log["datetime"] = pd.to_datetime(df_log["datetime"])
    df_log["Month"] = df_log["datetime"].dt.strftime("%Y-%m")

    # Monthly aggregations from arbitrage trading log
    monthly = (
        df_log.groupby("Month")
        .agg(
            Hours=("datetime", "count"),
            PV_kWh=("pv_kwh", "sum"),
            Load_kWh=("load_kwh", "sum"),
            Cost_CZK=("cost_buy_czk", "sum"),
            Rev_CZK=("revenue_sell_czk", "sum"),
            Deg_CZK=("degradation_cost_czk", "sum"),
            Net_Arbitrage_CZK=("net_profit_czk", "sum"),
            Export_Sold_kWh=("sell_kwh", "sum"),
            Discharge_kWh=("discharge_kwh", "sum"),
            Curtailment_kWh=("curtailment_kwh", "sum"),
        )
        .reset_index()
    )

    monthly["Battery_Cycles"] = (monthly["Discharge_kWh"] / 9.5).round(1)
    monthly["PV_kWh"] = monthly["PV_kWh"].round(1)
    monthly["Load_kWh"] = monthly["Load_kWh"].round(1)
    monthly["Export_Sold_kWh"] = monthly["Export_Sold_kWh"].round(1)
    monthly["Curtailment_kWh"] = monthly["Curtailment_kWh"].round(1)
    monthly["Net_Arbitrage_CZK"] = monthly["Net_Arbitrage_CZK"].round(2)

    # Load baseline benchmarks (Naive and V2)
    v3_p = Path(v3_benchmark_path)
    if v3_p.exists():
        df_base = pd.read_csv(v3_p)
        merged = pd.merge(
            monthly,
            df_base[["Month", "Net_Naive_CZK", "Net_V2_CZK"]],
            on="Month",
            how="left",
        )
    else:
        merged = monthly
        merged["Net_Naive_CZK"] = 0.0
        merged["Net_V2_CZK"] = 0.0

    merged["Arbitrage_vs_Naive_CZK"] = (merged["Net_Arbitrage_CZK"] - merged["Net_Naive_CZK"]).round(2)
    merged["Arbitrage_vs_V2_CZK"] = (merged["Net_Arbitrage_CZK"] - merged["Net_V2_CZK"]).round(2)

    # Reorder columns
    cols = [
        "Month",
        "Hours",
        "PV_kWh",
        "Load_kWh",
        "Net_Naive_CZK",
        "Net_V2_CZK",
        "Net_Arbitrage_CZK",
        "Arbitrage_vs_Naive_CZK",
        "Arbitrage_vs_V2_CZK",
        "Export_Sold_kWh",
        "Curtailment_kWh",
        "Battery_Cycles",
    ]
    res_df = merged[cols].copy()

    # Create summary total row for CSV
    total_row = {
        "Month": "ИТОГО",
        "Hours": int(res_df["Hours"].sum()),
        "PV_kWh": round(res_df["PV_kWh"].sum(), 1),
        "Load_kWh": round(res_df["Load_kWh"].sum(), 1),
        "Net_Naive_CZK": round(res_df["Net_Naive_CZK"].sum(), 2),
        "Net_V2_CZK": round(res_df["Net_V2_CZK"].sum(), 2),
        "Net_Arbitrage_CZK": round(res_df["Net_Arbitrage_CZK"].sum(), 2),
        "Arbitrage_vs_Naive_CZK": round(res_df["Arbitrage_vs_Naive_CZK"].sum(), 2),
        "Arbitrage_vs_V2_CZK": round(res_df["Arbitrage_vs_V2_CZK"].sum(), 2),
        "Export_Sold_kWh": round(res_df["Export_Sold_kWh"].sum(), 1),
        "Curtailment_kWh": round(res_df["Curtailment_kWh"].sum(), 1),
        "Battery_Cycles": round(res_df["Battery_Cycles"].sum(), 1),
    }
    df_with_total = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)

    # Save CSV
    out_csv_p = Path(out_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    df_with_total.to_csv(out_csv_p, index=False)
    print(f"Saved benchmark CSV to {out_csv_p}")

    # Build rich formatted Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "B2G Arbitrage Сводка"

    # Title Banner
    ws.merge_cells("A1:L1")
    ws["A1"] = "Коммерческий биржевой арбитраж СЭС (Battery-to-Grid) vs Базовые стратегии"
    ws["A1"].font = Font(name="Segoe UI", size=14, bold=True, color="1F497D")
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 32

    # Headers mapping: explicit column 'ИТОГО Arbitrage B2G (CZK)'
    headers_map = [
        ("Month", "Месяц"),
        ("Hours", "Часов в выборке"),
        ("PV_kWh", "Генерация СЭС (кВт⋅ч)"),
        ("Load_kWh", "Потребление дома (кВт⋅ч)"),
        ("Net_Naive_CZK", "Фин. итог Naive (CZK)"),
        ("Net_V2_CZK", "Фин. итог V2 (CZK)"),
        ("Net_Arbitrage_CZK", "ИТОГО Arbitrage B2G (CZK)"),
        ("Arbitrage_vs_Naive_CZK", "Выгода B2G vs Naive (CZK)"),
        ("Arbitrage_vs_V2_CZK", "Выгода B2G vs V2 (CZK)"),
        ("Export_Sold_kWh", "Экспорт в сеть (кВт⋅ч)"),
        ("Curtailment_kWh", "Ограничение FVE (кВт⋅ч)"),
        ("Battery_Cycles", "Циклов АКБ"),
    ]

    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    ws.row_dimensions[3].height = 28

    for col_idx, (_, label) in enumerate(headers_map, start=1):
        cell = ws.cell(row=3, column=col_idx, value=label)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    data_font = Font(name="Segoe UI", size=10)
    green_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    dark_green_font = Font(name="Segoe UI", size=10, bold=True, color="276A3C")

    # Data Rows (4 to 20)
    row_start = 4
    for r_idx, row in res_df.iterrows():
        curr_row = row_start + r_idx
        ws.row_dimensions[curr_row].height = 20
        for c_idx, (col_name, _) in enumerate(headers_map, start=1):
            val = row[col_name]
            cell = ws.cell(row=curr_row, column=c_idx, value=val)
            cell.font = data_font
            cell.border = thin_border

            # Number formatting & alignment
            if c_idx == 1:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif c_idx == 2:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "#,##0"
            elif c_idx in [3, 4, 10, 11]:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "#,##0.0"
            elif c_idx in [5, 6, 7, 8, 9]:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "#,##0.00"
            elif c_idx == 12:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "0.0"

            # Highlight savings columns in green
            if col_name in ["Arbitrage_vs_Naive_CZK", "Arbitrage_vs_V2_CZK"] and val > 0:
                cell.fill = green_fill
                cell.font = dark_green_font

    # ИТОГО / Total Row
    tot_row = row_start + len(res_df)
    ws.row_dimensions[tot_row].height = 26
    tot_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    tot_font = Font(name="Segoe UI", size=10, bold=True)
    double_border = Border(
        top=Side(style="thin", color="1F497D"),
        bottom=Side(style="double", color="1F497D"),
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
    )

    cell_tot_lbl = ws.cell(row=tot_row, column=1, value="ИТОГО")
    cell_tot_lbl.alignment = Alignment(horizontal="center", vertical="center")
    cell_tot_lbl.font = tot_font
    cell_tot_lbl.fill = tot_fill
    cell_tot_lbl.border = double_border

    for c in range(2, len(headers_map) + 1):
        col_let = get_column_letter(c)
        cell = ws.cell(row=tot_row, column=c)
        cell.font = tot_font
        cell.fill = tot_fill
        cell.border = double_border
        cell.value = f"=SUM({col_let}4:{col_let}{tot_row-1})"
        cell.alignment = Alignment(horizontal="right", vertical="center")

        if c == 2:
            cell.number_format = "#,##0"
        elif c in [3, 4, 10, 11]:
            cell.number_format = "#,##0.0"
        elif c in [5, 6, 7, 8, 9]:
            cell.number_format = "#,##0.00"
        elif c == 12:
            cell.number_format = "0.0"

    # Column widths auto-adjustment
    widths = {
        "A": 16,
        "B": 18,
        "C": 24,
        "D": 26,
        "E": 22,
        "F": 20,
        "G": 26,
        "H": 25,
        "I": 25,
        "J": 24,
        "K": 24,
        "L": 16,
    }
    for col_letter, width in widths.items():
        ws.column_dimensions[col_letter].width = width

    # Save primary Excel file
    out_xlsx_p = Path(out_xlsx)
    try:
        wb.save(out_xlsx_p)
        print(f"Saved benchmark Excel to {out_xlsx_p}")
    except PermissionError:
        print(f"Warning: '{out_xlsx_p}' is currently open in Excel or locked. Please close it to overwrite.")

    # Also save a copy with '_latest' suffix to guarantee access
    latest_xlsx = out_xlsx_p.with_name(f"{out_xlsx_p.stem}_latest.xlsx")
    try:
        wb.save(latest_xlsx)
        print(f"Saved latest benchmark Excel copy to {latest_xlsx}")
    except Exception as e:
        print(f"Note saving latest copy: {e}")

    return res_df


if __name__ == "__main__":
    generate_arbitrage_benchmark()
