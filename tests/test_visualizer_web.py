import pandas as pd
from pathlib import Path
import pytest

from visualizer_web import (
    detect_separator,
    load_trading_log,
    get_available_csv_files,
    format_month_label,
    find_target_date_in_month,
    get_missing_ranges,
    CZECH_MONTHS,
)


def test_detect_separator(tmp_path):
    semicolon_file = tmp_path / "semi.csv"
    semicolon_file.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    assert detect_separator(semicolon_file) == ";"

    comma_file = tmp_path / "comma.csv"
    comma_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert detect_separator(comma_file) == ","


def test_get_available_csv_files():
    files = get_available_csv_files()
    assert len(files) > 0
    # trading_log_arbitrage should be prioritized first if present
    names = [f.name for f in files]
    assert any("trading_log" in n for n in names)


def test_load_trading_log():
    files = get_available_csv_files()
    arb_files = [f for f in files if "arbitrage" in f.name.lower()]
    target = arb_files[0] if arb_files else files[0]
    
    df = load_trading_log(str(target))
    assert not df.empty
    assert "dt" in df.columns
    assert "date_str" in df.columns
    assert "time_str" in df.columns


def test_format_month_label():
    assert format_month_label("2025-07") == "Červenec 2025 (2025-07)"
    assert format_month_label("2025-12") == "Prosinec 2025 (2025-12)"
    assert format_month_label("2026-01") == "Leden 2026 (2026-01)"
    assert format_month_label("invalid") == "invalid"


def test_find_target_date_in_month():
    month_to_dates = {
        "2025-05": [f"2025-05-{d:02d}" for d in range(24, 32)],
        "2025-08": [f"2025-08-{d:02d}" for d in range(1, 32)],
        "2025-09": [f"2025-09-{d:02d}" for d in range(1, 31)],
    }
    # Exact day match exists
    assert find_target_date_in_month("2025-08", 15, month_to_dates) == "2025-08-15"
    # Day 31 clamped to 30 in September
    assert find_target_date_in_month("2025-09", 31, month_to_dates) == "2025-09-30"
    # Day 5 clamped to first date (24th) in May
    assert find_target_date_in_month("2025-05", 5, month_to_dates) == "2025-05-24"
    # Non-existent month returns empty string
    assert find_target_date_in_month("2024-01", 10, month_to_dates) == ""


def test_get_missing_ranges():
    # Day with all 24 hours missing
    dts_full = pd.date_range("2026-03-27 00:00:00", periods=24, freq="h")
    df_missing_all = pd.DataFrame({"dt": dts_full, "has_telemetry": [False] * 24})
    ranges_all = get_missing_ranges(df_missing_all)
    assert len(ranges_all) == 1
    assert ranges_all[0][0] == dts_full[0]
    assert ranges_all[0][1] == dts_full[-1]

    # Day with partial missing (e.g. 00:00 - 08:00 missing, 09:00 - 23:00 present)
    df_part = pd.DataFrame({"dt": dts_full, "has_telemetry": [False] * 9 + [True] * 15})
    ranges_part = get_missing_ranges(df_part)
    assert len(ranges_part) == 1
    assert ranges_part[0][0] == dts_full[0]
    assert ranges_part[0][1] == dts_full[8]

    # Day with no missing telemetry
    df_valid = pd.DataFrame({"dt": dts_full, "has_telemetry": [True] * 24})
    assert get_missing_ranges(df_valid) == []
