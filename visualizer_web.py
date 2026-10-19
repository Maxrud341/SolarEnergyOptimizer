"""Interactive Streamlit Web Dashboard for Solar Energy Arbitrage logs.

Features:
- 2 synchronized Plotly subplots: Energy balance (kWh) & Spot market prices / Finance (CZK)
- Unified time hover (crosshair across both graphs)
- Day navigation: [< Previous Day], [Next Day >], and Calendar DatePicker
- Daily summary KPI cards (Net Profit, Solar Generation, Home Load, Battery Cycles, Grid Export)
- Toggleable series via sidebar and interactive legend click
- Direct launch support: python visualizer_web.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Support running directly via `python visualizer_web.py`
if __name__ == "__main__" and "streamlit" not in sys.modules and not os.environ.get("STREAMLIT_SERVER_PORT"):
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

import streamlit as st


# =====================================================================
# 1. Page Configuration & Styling
# =====================================================================

st.set_page_config(
    page_title="Optimalizátor FVE & Baterie | Arbitráž",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark microgrid control center theme */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    header, footer {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 0.8rem 1.1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #475569;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #334155;
        border-color: #38bdf8;
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 2. Data Loading & CSV Utilities
# =====================================================================

def detect_separator(path: Path) -> str:
    """Detect CSV delimiter (; or ,)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except Exception:
        return ";" if ";" in sample else ","


@st.cache_data(show_spinner=False)
def load_trading_log(csv_path: str) -> pd.DataFrame:
    """Load and prepare trading log data."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()

    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep)

    # Find datetime column
    dt_col = None
    for cand in ("datetime", "DatumCas", "DatumCasOTE", "timestamp", "time"):
        if cand in df.columns:
            dt_col = cand
            break

    if dt_col is None:
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                dt_col = c
                break

    if dt_col is None:
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["time_str"] = df["dt"].dt.strftime("%H:%M")

    # Flag for valid measured telemetry (vs missing data periods)
    if "has_telemetry" in df.columns:
        df["has_telemetry"] = df["has_telemetry"].astype(str).str.lower().isin(["true", "1"])
    elif "load_kwh" in df.columns:
        df["has_telemetry"] = df["load_kwh"].notna()
    else:
        df["has_telemetry"] = True

    return df


def get_available_csv_files() -> list[Path]:
    """Find all relevant CSV files in root and data/."""
    candidates = []
    root = Path(__file__).resolve().parent

    # Priority trading logs
    for p in root.glob("trading_log*.csv"):
        if p.is_file():
            candidates.append(p)

    # Data directory logs
    data_dir = root / "data"
    if data_dir.exists():
        for p in data_dir.glob("*.csv"):
            if p.is_file() and p not in candidates:
                candidates.append(p)

    # Legacy / archived trading logs
    legacy_dir = root / "legacy" / "trading_logs"
    if legacy_dir.exists():
        for p in legacy_dir.glob("*.csv"):
            if p.is_file() and p not in candidates:
                candidates.append(p)

    candidates.sort(key=lambda x: (0 if "arbitrage" in x.name.lower() else (1 if "trading_log" in x.name.lower() else 2), x.name.lower()))
    return candidates


# Czech month names dictionary for UI display
CZECH_MONTHS = {
    1: "Leden",
    2: "Únor",
    3: "Březen",
    4: "Duben",
    5: "Květen",
    6: "Červen",
    7: "Červenec",
    8: "Srpen",
    9: "Září",
    10: "Říjen",
    11: "Listopad",
    12: "Prosinec",
}


def format_month_label(month_str: str) -> str:
    """Format 'YYYY-MM' into Czech label, e.g. 'Červenec 2025 (2025-07)'."""
    try:
        parts = month_str.split("-")
        year = int(parts[0])
        month = int(parts[1])
        name = CZECH_MONTHS.get(month, "")
        return f"{name} {year} ({month_str})"
    except Exception:
        return month_str


def find_target_date_in_month(target_month: str, preferred_day_num: int, month_to_dates: dict[str, list[str]]) -> str:
    """Find matching or closest date in target month preserving day of month."""
    dates_in_m = month_to_dates.get(target_month, [])
    if not dates_in_m:
        return ""
    exact_candidate = f"{target_month}-{preferred_day_num:02d}"
    if exact_candidate in dates_in_m:
        return exact_candidate
    return min(dates_in_m, key=lambda d: abs(int(d.split("-")[2]) - preferred_day_num))


def get_missing_ranges(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Find contiguous ranges of missing telemetry hours in day dataframe."""
    if "has_telemetry" in df.columns:
        is_missing = (~df["has_telemetry"].fillna(False)).to_numpy()
    elif "load_kwh" in df.columns:
        is_missing = df["load_kwh"].isna().to_numpy()
    else:
        return []

    dts = pd.to_datetime(df["dt"]).to_numpy()
    ranges = []
    in_gap = False
    start_dt = None

    for i, missing in enumerate(is_missing):
        if missing and not in_gap:
            in_gap = True
            start_dt = dts[i]
        elif not missing and in_gap:
            in_gap = False
            ranges.append((pd.to_datetime(start_dt), pd.to_datetime(dts[i - 1])))

    if in_gap:
        ranges.append((pd.to_datetime(start_dt), pd.to_datetime(dts[-1])))

    return ranges


# =====================================================================
# 3. Sidebar: File Selection & Options
# =====================================================================

csv_files = get_available_csv_files()
if not csv_files:
    st.error("❌ V projektu nebyly nalezeny žádné CSV soubory. Nejprve spusťte optimalizátor.")
    st.stop()

file_names = [f.name for f in csv_files]
default_file_idx = 0
for idx, fn in enumerate(file_names):
    if "trading_log_arbitrage" in fn:
        default_file_idx = idx
        break

selected_filename = st.sidebar.selectbox("📁 Výběr souboru protokolu", file_names, index=default_file_idx)
selected_path = str(next(f for f in csv_files if f.name == selected_filename))

df = load_trading_log(selected_path)
if df.empty:
    st.error(f"❌ Nelze načíst nebo zpracovat data ze souboru `{selected_filename}`.")
    st.stop()

available_dates = df["date_str"].unique().tolist()
if not available_dates:
    st.error("V souboru nebyla nalezena žádná platná data.")
    st.stop()

# Group dates by month
available_months = sorted(list({d[:7] for d in available_dates}))
month_to_dates: dict[str, list[str]] = {}
for d in available_dates:
    month_to_dates.setdefault(d[:7], []).append(d)

# Initialize session state for active date
if "active_date_idx" not in st.session_state:
    st.session_state.active_date_idx = 0
if st.session_state.active_date_idx >= len(available_dates):
    st.session_state.active_date_idx = len(available_dates) - 1

current_date_str = available_dates[st.session_state.active_date_idx]
current_month_str = current_date_str[:7]
current_month_idx = available_months.index(current_month_str) if current_month_str in available_months else 0
current_day_num = int(current_date_str.split("-")[2])

# Sidebar Month Quick Jump
st.sidebar.markdown("---")
st.sidebar.subheader("🗓️ Rychlý výběr měsíce")
sidebar_selected_month = st.sidebar.selectbox(
    "Přejít na měsíc",
    available_months,
    index=current_month_idx,
    format_func=format_month_label,
)
if sidebar_selected_month != current_month_str:
    target_date = find_target_date_in_month(sidebar_selected_month, current_day_num, month_to_dates)
    if target_date in available_dates:
        st.session_state.active_date_idx = available_dates.index(target_date)
        st.rerun()

# Sidebar Series Customization
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filtry křivek a zobrazení")

show_phases = st.sidebar.checkbox("Fázový detail spotřeby (L1, L2, L3)", value=False)
show_soc = st.sidebar.checkbox("Stav nabití baterie (SoC v kWh)", value=True)
show_curtailment = st.sidebar.checkbox("Omezení přetoků (Zero Export)", value=True)
show_breakdown_costs = st.sidebar.checkbox("Rozpad nákladů a výnosů (CZK)", value=False)


# =====================================================================
# 4. Top Header & Navigation (Month & Day)
# =====================================================================

st.title("⚡ Vizualizátor komerční arbitráže FVE & Baterie")

# --- Row 1: Month Navigation ---
m_col1, m_col2, m_col3, m_col4 = st.columns([1.5, 3, 1.5, 4])

with m_col1:
    prev_m_disabled = (current_month_idx <= 0)
    if st.button("⏮️ Předchozí měsíc", use_container_width=True, disabled=prev_m_disabled):
        target_month = available_months[current_month_idx - 1]
        target_date = find_target_date_in_month(target_month, current_day_num, month_to_dates)
        if target_date in available_dates:
            st.session_state.active_date_idx = available_dates.index(target_date)
            st.rerun()

with m_col3:
    next_m_disabled = (current_month_idx >= len(available_months) - 1)
    if st.button("Následující měsíc ⏭️", use_container_width=True, disabled=next_m_disabled):
        target_month = available_months[current_month_idx + 1]
        target_date = find_target_date_in_month(target_month, current_day_num, month_to_dates)
        if target_date in available_dates:
            st.session_state.active_date_idx = available_dates.index(target_date)
            st.rerun()

with m_col2:
    selected_month = st.selectbox(
        "Výběr měsíce",
        available_months,
        index=current_month_idx,
        format_func=format_month_label,
        label_visibility="collapsed",
    )
    if selected_month != current_month_str:
        target_date = find_target_date_in_month(selected_month, current_day_num, month_to_dates)
        if target_date in available_dates:
            st.session_state.active_date_idx = available_dates.index(target_date)
            st.rerun()

with m_col4:
    total_months = len(available_months)
    days_in_month = len(month_to_dates.get(current_month_str, []))
    month_czech_name = format_month_label(current_month_str)
    st.markdown(
        f"<div style='padding-top: 6px; color: #94a3b8; font-size: 0.95rem;'>"
        f"🗓️ Měsíc <b>{current_month_idx + 1}</b> z <b>{total_months}</b> "
        f"• <span style='color: #a78bfa; font-weight: bold;'>{month_czech_name}</span> "
        f"<span style='color: #64748b;'>({days_in_month} dní)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# --- Row 2: Day Navigation ---
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1.5, 3, 1.5, 4])

with nav_col1:
    prev_d_disabled = (st.session_state.active_date_idx <= 0)
    if st.button("⬅️ Předchozí den", use_container_width=True, disabled=prev_d_disabled):
        if st.session_state.active_date_idx > 0:
            st.session_state.active_date_idx -= 1
            st.rerun()

with nav_col3:
    next_d_disabled = (st.session_state.active_date_idx >= len(available_dates) - 1)
    if st.button("Následující den ➡️", use_container_width=True, disabled=next_d_disabled):
        if st.session_state.active_date_idx < len(available_dates) - 1:
            st.session_state.active_date_idx += 1
            st.rerun()

# Date picker in col2
current_date_obj = pd.to_datetime(current_date_str).date()

with nav_col2:
    selected_date = st.date_input(
        "Výběr data",
        value=current_date_obj,
        min_value=pd.to_datetime(available_dates[0]).date(),
        max_value=pd.to_datetime(available_dates[-1]).date(),
        label_visibility="collapsed",
    )
    selected_date_str = selected_date.strftime("%Y-%m-%d")
    if selected_date_str in available_dates:
        new_idx = available_dates.index(selected_date_str)
        if new_idx != st.session_state.active_date_idx:
            st.session_state.active_date_idx = new_idx
            st.rerun()
    else:
        # Find closest date
        closest_idx = min(range(len(available_dates)), key=lambda i: abs((pd.to_datetime(available_dates[i]).date() - selected_date).days))
        if closest_idx != st.session_state.active_date_idx:
            st.session_state.active_date_idx = closest_idx
            st.rerun()

with nav_col4:
    total_days = len(available_dates)
    dates_in_curr_month = month_to_dates.get(current_month_str, [])
    day_in_month_idx = dates_in_curr_month.index(current_date_str) + 1 if current_date_str in dates_in_curr_month else 1
    st.markdown(
        f"<div style='padding-top: 6px; color: #94a3b8; font-size: 0.95rem;'>"
        f"📅 Den <b>{st.session_state.active_date_idx + 1}</b> z <b>{total_days}</b> "
        f"• Vybráno: <span style='color: #38bdf8; font-weight: bold;'>{current_date_str}</span> "
        f"<span style='color: #64748b;'>({day_in_month_idx}. v měsíci)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# 5. Filter Data for Selected Day & Compute Daily KPIs
# =====================================================================

day_df = df[df["date_str"] == current_date_str].copy()
if day_df.empty:
    st.warning(f"Žádná data pro den {current_date_str}.")
    st.stop()

# Missing telemetry check
valid_mask = day_df["has_telemetry"] == True if "has_telemetry" in day_df.columns else day_df["load_kwh"].notna()
valid_day = day_df[valid_mask]
missing_hours_count = len(day_df) - len(valid_day)

if missing_hours_count == len(day_df):
    st.warning(
        "⚠️ **Tento den neobsahuje žádná telemetrická data** (24 hodin bez záznamu z invertoru). "
        "Tyto hodiny nejsou započítávány do nákladů ani do bilance a v grafu jsou vyznačeny šedou přerušovanou oblastí."
    )
elif missing_hours_count > 0:
    st.info(
        f"ℹ️ **Poznámka k datům:** V tomto dni chybí telemetrie pro **{missing_hours_count} hod.** "
        f"(vyznačeno v grafu šedou přerušovanou čarou). Tyto hodiny nejsou započítávány do denního součtu."
    )

# Compute Daily KPIs (only on valid telemetry hours)
load_sum = valid_day["load_kwh"].dropna().sum() if "load_kwh" in valid_day.columns else 0.0
pv_sum = valid_day["pv_kwh"].dropna().sum() if "pv_kwh" in valid_day.columns else 0.0
buy_sum = valid_day["buy_kwh"].dropna().sum() if "buy_kwh" in valid_day.columns else 0.0
sell_sum = valid_day["sell_kwh"].dropna().sum() if "sell_kwh" in valid_day.columns else 0.0
dis_sum = valid_day["discharge_kwh"].dropna().sum() if "discharge_kwh" in valid_day.columns else 0.0
curt_sum = valid_day["curtailment_kwh"].dropna().sum() if "curtailment_kwh" in valid_day.columns else 0.0
cycles = dis_sum / 9.5
profit_sum = valid_day["net_profit_czk"].dropna().sum() if "net_profit_czk" in valid_day.columns else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    delta_str = "Zisk" if profit_sum >= 0 else "Náklad"
    st.metric("💰 Čistý denní výsledek", f"{profit_sum:+.2f} CZK", delta=f"{delta_str}")
with kpi2:
    st.metric("🏠 Spotřeba domu", f"{load_sum:.2f} kWh", delta=f"Nákup ze sítě: {buy_sum:.2f} kWh", delta_color="inverse")
with kpi3:
    if curt_sum > 0.05:
        st.metric("☀️ Výroba FVE", f"{pv_sum:.2f} kWh", delta=f"Omezeno: {curt_sum:.2f} kWh", delta_color="inverse")
    else:
        st.metric("☀️ Výroba FVE", f"{pv_sum:.2f} kWh", delta="100% využito bez omezení")
with kpi4:
    st.metric("⚡ Prodej do sítě (B2G)", f"{sell_sum:.2f} kWh", delta="Export na spot")
with kpi5:
    st.metric("🔋 Využití baterie", f"{cycles:.2f} cyklu", delta=f"Vybito: {dis_sum:.2f} kWh")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# =====================================================================
# 6. Plotly Subplots (2 Synchronized Rows)
# =====================================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "<b>⚡ Energetická bilance a baterie (kWh)</b>",
        "<b>💰 Tržní spotové ceny OTE a finanční bilance (CZK)</b>",
    ),
    row_heights=[0.58, 0.42],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
)

# Time axis array
times = day_df["dt"]

# -------------------------------------------------------------
# SUBPLOT 1: ENERGY FLOWS (kWh)
# -------------------------------------------------------------

# 1. Solar Generation (PV) - Amber Gold
if "pv_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["pv_kwh"],
            name="Výroba FVE",
            line=dict(color="#eab308", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(234, 179, 8, 0.12)",
            hovertemplate="FVE: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

# 2. Total House Load - Strong Blue
if "load_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["load_kwh"],
            name="Spotřeba domu",
            line=dict(color="#0284c7", width=2.5),
            hovertemplate="Dům: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

# Optional Phase details
if show_phases:
    if "load_l1_kwh" in day_df.columns:
        fig.add_trace(
            go.Scatter(x=times, y=day_df["load_l1_kwh"], name="Fáze L1", line=dict(color="#93c5fd", width=1, dash="dot")),
            row=1,
            col=1,
            secondary_y=False,
        )
    if "load_l2_kwh" in day_df.columns:
        fig.add_trace(
            go.Scatter(x=times, y=day_df["load_l2_kwh"], name="Fáze L2", line=dict(color="#60a5fa", width=1, dash="dot")),
            row=1,
            col=1,
            secondary_y=False,
        )
    if "load_l3_kwh" in day_df.columns:
        fig.add_trace(
            go.Scatter(x=times, y=day_df["load_l3_kwh"], name="Fáze L3 (Bojler)", line=dict(color="#2563eb", width=1.2, dash="dash")),
            row=1,
            col=1,
            secondary_y=False,
        )

# 3. Battery Charge - Emerald Green
if "charge_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["charge_kwh"],
            name="Nabíjení baterie",
            line=dict(color="#10b981", width=2.0),
            hovertemplate="Nabíjení: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

# 4. Battery Discharge - Violet Purple
if "discharge_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["discharge_kwh"],
            name="Vybíjení baterie",
            line=dict(color="#8b5cf6", width=2.2),
            hovertemplate="Vybíjení: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

# 5. Grid Buy - Crimson Red
if "buy_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["buy_kwh"],
            name="Nákup ze sítě",
            line=dict(color="#ef4444", width=1.8),
            hovertemplate="Nákup: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

# 6. Grid Sell / B2G Export - Vibrant Hot Pink / Magenta (Distinct from House Blue!)
if "sell_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["sell_kwh"],
            name="Prodej do sítě (B2G)",
            line=dict(color="#ec4899", width=2.5),
            hovertemplate="Prodej: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

# 7. Curtailment (if present) - Coral Rose Dash
if show_curtailment and "curtailment_kwh" in day_df.columns:
    if day_df["curtailment_kwh"].sum() > 1e-4:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=day_df["curtailment_kwh"],
                name="Omezení přetoků (Zero Export)",
                line=dict(color="#f43f5e", width=1.5, dash="dash"),
                hovertemplate="Omezení: %{y:.2f} kWh<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

# 8. Battery SoC (Secondary Y-axis, 0 - 10 kWh) - Silver Platinum
if show_soc and "soc_kwh" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["soc_kwh"],
            name="Stav nabití (SoC)",
            line=dict(color="#cbd5e1", width=2.2, dash="dashdot"),
            hovertemplate="SoC: %{y:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

# -------------------------------------------------------------
# SUBPLOT 2: PRICES (CZK/kWh) & FINANCIALS (CZK)
# -------------------------------------------------------------

# Spot Price OTE (Gold)
if "spot_price_czk" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["spot_price_czk"],
            name="Spot OTE (CZK/kWh)",
            line=dict(color="#fbbf24", width=2.2),
            hovertemplate="Spot OTE: %{y:.2f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )

# Effective Buy Price (with Grid Fee & VAT)
if "p_buy" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["p_buy"],
            name="Nákupní cena (vč. distribuce a DPH)",
            line=dict(color="#f87171", width=1.5, dash="dot"),
            hovertemplate="Nákupní cena: %{y:.2f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )

# Effective Sell Price
if "p_sell" in day_df.columns:
    fig.add_trace(
        go.Scatter(
            x=times,
            y=day_df["p_sell"],
            name="Výkupní cena (čistá)",
            line=dict(color="#34d399", width=1.5, dash="dot"),
            hovertemplate="Výkupní cena: %{y:.2f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )

# Hourly Net Profit (Bar chart: Green if >=0, Red if <0, hidden if NaN)
if "net_profit_czk" in day_df.columns:
    bar_colors = ["#22c55e" if (pd.notna(val) and val >= 0) else "#ef4444" for val in day_df["net_profit_czk"]]
    fig.add_trace(
        go.Bar(
            x=times,
            y=day_df["net_profit_czk"],
            name="Hodinová bilance (CZK)",
            marker_color=bar_colors,
            opacity=0.6,
            hovertemplate="Výsledek hodiny: %{y:+.2f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )

# Optional breakdown costs
if show_breakdown_costs:
    if "cost_buy_czk" in day_df.columns:
        fig.add_trace(go.Scatter(x=times, y=day_df["cost_buy_czk"], name="Náklady na nákup", line=dict(color="#f43f5e", width=1)), row=2, col=1)
    if "revenue_sell_czk" in day_df.columns:
        fig.add_trace(go.Scatter(x=times, y=day_df["revenue_sell_czk"], name="Příjmy z prodeje", line=dict(color="#10b981", width=1)), row=2, col=1)
    if "degradation_cost_czk" in day_df.columns:
        fig.add_trace(go.Scatter(x=times, y=day_df["degradation_cost_czk"], name="Opotřebení baterie", line=dict(color="#8b5cf6", width=1)), row=2, col=1)

# Zero reference line on Price/Financial plot
fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1, row=2, col=1)

# Highlight missing telemetry gaps: vertical gray dashed bands & annotations
missing_ranges = get_missing_ranges(day_df)
for start_dt, end_dt in missing_ranges:
    fig.add_vrect(
        x0=start_dt - pd.Timedelta(minutes=30),
        x1=end_dt + pd.Timedelta(minutes=30),
        fillcolor="rgba(100, 116, 139, 0.14)",
        line=dict(color="#64748b", width=1.5, dash="dash"),
        annotation_text="⚠️ Žádná data",
        annotation_position="inside top",
        annotation=dict(
            font=dict(color="#94a3b8", size=11, family="Segoe UI, sans-serif"),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#475569",
            borderwidth=1,
        ),
    )


# =====================================================================
# 7. Layout Polish & Theme Styling
# =====================================================================

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b1120",
    plot_bgcolor="#0f172a",
    font=dict(family="Segoe UI, Inter, sans-serif", color="#e2e8f0"),
    height=740,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=55, r=55, t=70, b=45),
)

# Axis Styling
fig.update_xaxes(
    tickformat="%H:%M",
    gridcolor="#1e293b",
    zerolinecolor="#334155",
    showspikes=True,
    spikecolor="#38bdf8",
    spikesnap="cursor",
    spikemode="across",
    spikethickness=1,
)

fig.update_yaxes(
    title_text="Tok energie (kWh)",
    gridcolor="#1e293b",
    zerolinecolor="#334155",
    row=1,
    col=1,
    secondary_y=False,
)

if show_soc:
    fig.update_yaxes(
        title_text="Kapacita baterie SoC (kWh)",
        range=[0.0, 10.5],
        gridcolor="rgba(0,0,0,0)",
        row=1,
        col=1,
        secondary_y=True,
    )

fig.update_yaxes(
    title_text="Cena / Bilance (CZK)",
    gridcolor="#1e293b",
    zerolinecolor="#334155",
    row=2,
    col=1,
)

try:
    st.plotly_chart(fig, width="stretch")
except TypeError:
    st.plotly_chart(fig, use_container_width=True)

# Footer info
st.markdown(
    f"<div style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 15px;'>"
    f"💡 <i>Tip: Kliknutím na položku v legendě nahoře skryjete/zobrazíte danou křivku; dvojklikem izolujete pouze ji. "
    f"Kolečkem myši přiblížíte oba grafy současně.</i>"
    f"</div>",
    unsafe_allow_html=True,
)
