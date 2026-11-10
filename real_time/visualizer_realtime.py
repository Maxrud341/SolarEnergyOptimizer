"""Interactive Streamlit Web Dashboard for Real-Time FVE & Battery Arbitrage.

Features identical to visualizer_web.py:
- Day & Month navigation: [< Previous Day], [Next Day >], Calendar DatePicker, Month Quick Jump
- Clear handling of days without data with shaded gap annotations (⚠️ Žádná data)
- 5 Standard Daily KPI cards (Net Profit, Home Load, Solar Generation, B2G Export, Battery Cycles)
- 2 Synchronized Plotly subplots with unified crosshair hover (Energy Balance & Spot Market/Finance)
- Live Real-Time Header with Pulse Status (🟢 ONLINE / 🟡 ZPOŽDĚNÍ / 🔴 OFFLINE)
- Toggleable series via sidebar (SoC, phases, cost breakdown)

Run directly via:
    python real_time/visualizer_realtime.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import sys
import time

# Auto-relaunch inside Streamlit if invoked via `python visualizer_realtime.py`
if __name__ == "__main__" and "streamlit" not in sys.modules and not os.environ.get("STREAMLIT_SERVER_PORT"):
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add this directory and parent to sys.path
this_dir = Path(__file__).resolve().parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))
project_root = this_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from real_time.config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        DB_PATH,
        EXPORT_FEE_CZK_PER_KWH,
        IMPORT_FEE_CZK_PER_KWH,
        MAX_DB_SIZE_BYTES,
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        MQTT_TOPIC_RX,
        MQTT_TOPIC_TX,
        SOC_MAX,
        SOC_MIN,
        VAT_RATE,
        WATCHDOG_TIMEOUT_SEC,
    )
    from real_time.database import (
        get_all_recorded_dates,
        get_latest_tick,
        get_ticks_for_date,
        init_db,
    )
except ImportError:
    from config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        DB_PATH,
        EXPORT_FEE_CZK_PER_KWH,
        IMPORT_FEE_CZK_PER_KWH,
        MAX_DB_SIZE_BYTES,
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        MQTT_TOPIC_RX,
        MQTT_TOPIC_TX,
        SOC_MAX,
        SOC_MIN,
        VAT_RATE,
        WATCHDOG_TIMEOUT_SEC,
    )
    from database import (
        get_all_recorded_dates,
        get_latest_tick,
        get_ticks_for_date,
        init_db,
    )

# =====================================================================
# 1. Page Configuration & Microgrid Dark Theme (Identical to visualizer_web.py)
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
        padding-top: 1.2rem;
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
    /* Live Status Pill */
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.90rem;
        letter-spacing: 0.5px;
    }
    .status-online {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #059669;
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #d97706;
    }
    .status-offline {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

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
    """Format 'YYYY-MM' into Czech label, e.g. 'Říjen 2026 (2026-10)'."""
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


@st.cache_data(ttl=600)
def compute_tomorrow_plan():
    """Fetch spot prices & weather forecast for tomorrow, returning the LP schedule."""
    try:
        from real_time.price_fetcher import SpotPriceFetcher
        from real_time.weather_fetcher import SolarWeatherFetcher
        from real_time.lp_optimizer import LpArbitrageOptimizerRT
    except ImportError:
        from price_fetcher import SpotPriceFetcher
        from weather_fetcher import SolarWeatherFetcher
        from lp_optimizer import LpArbitrageOptimizerRT

    now = dt.datetime.now()
    tomorrow_start = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    pf = SpotPriceFetcher()
    prices_df = pf.get_hourly_price_vector(start_dt=tomorrow_start, hours=24)
    if prices_df.empty or len(prices_df) < 12:
        return None, None, None

    wf = SolarWeatherFetcher()
    pv_df = wf.get_pv_forecast_vector(start_dt=tomorrow_start, hours=24)
    pv_forecast = pv_df["pv_forecast_kw"].tolist() if not pv_df.empty else [0.0] * 24
    spot_prices = prices_df["price_czk_kwh"].tolist()
    hours_of_day = [r["start"].hour for _, r in prices_df.iterrows()]

    load_forecast = [0.80 if 17 <= h <= 22 else 0.45 if 7 <= h <= 16 else 0.25 for h in hours_of_day]

    opt = LpArbitrageOptimizerRT()
    cloud_forecast = pv_df["cloud_pct"].tolist() if "cloud_pct" in pv_df.columns else None
    schedule = opt.solve_horizon(
        current_soc_kwh=BATTERY_CAPACITY_KWH * 0.20,
        spot_prices=spot_prices,
        pv_forecast_kw=pv_forecast,
        load_forecast_kw=load_forecast,
        hours_of_day=hours_of_day,
        cloud_cover_pct=cloud_forecast,
    )

    rows = []
    for h, a in sorted(schedule.items()):
        rows.append({
            "Čas": f"{h:02d}:00",
            "Spotová cena (CZK)": round(spot_prices[h], 2),
            "FVE (kW)": round(pv_forecast[h], 2),
            "Spotřeba (kW)": round(load_forecast[h], 2),
            "Příkaz střídači": a.command,
            "Výkon (W)": int(a.target_power_w),
            "Plánované SoC (%)": round(a.planned_end_soc_pct, 1),
            "Důvod": a.reason,
        })
    df_plan = pd.DataFrame(rows)

    summary = {
        "date_str": tomorrow_start.strftime("%d.%m.%Y"),
        "total_pv_kwh": sum(pv_forecast),
        "total_load_kwh": sum(load_forecast),
        "min_price": min(spot_prices),
        "min_hour": f"{spot_prices.index(min(spot_prices)):02d}:00",
        "max_price": max(spot_prices),
        "max_hour": f"{spot_prices.index(max(spot_prices)):02d}:00",
        "spread": max(spot_prices) - min(spot_prices),
    }
    return df_plan, summary, None


# =====================================================================
# 2. Sidebar: Settings & Series Customization
# =====================================================================

with st.sidebar:
    st.markdown("### ⚡ Parametry Systému")
    st.markdown(
        f"""
        * **Baterie:** Fox-ESS HV2600 HS 10.4
        * **Kapacita:** `{BATTERY_CAPACITY_KWH} kWh`
        * **Limity:** `SoC {int(SOC_MIN*100)}% – {int(SOC_MAX*100)}%`
        * **Max. výkon:** `{BATTERY_POWER_MAX_KW} kW`
        * **Broker:** `{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}`
        * **Inbound:** `{MQTT_TOPIC_TX}`
        * **Outbound:** `{MQTT_TOPIC_RX}`
        """
    )
    st.markdown("---")
    st.subheader("🎛️ Filtry křivek a zobrazení")
    show_soc = st.checkbox("Stav nabití baterie (SoC)", value=True)
    show_breakdown_costs = st.checkbox("Rozpad nákupních/výkupních cen", value=False)

    st.markdown("---")
    st.markdown("### 💾 Úložiště SQLite")
    if DB_PATH.exists():
        db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        pct_used = (DB_PATH.stat().st_size / MAX_DB_SIZE_BYTES) * 100.0
        st.write(f"Velikost DB: **{db_size_mb:.2f} MB** ({pct_used:.2f}%)")
        st.progress(min(1.0, pct_used / 100.0))
    else:
        st.write("DB zatím nevytvořena.")

    st.markdown("---")
    st.markdown("### 🔄 Obnovování dat")
    enable_auto_refresh = st.toggle("Živé automatické obnovení", value=True)
    if enable_auto_refresh:
        auto_refresh_sec = st.slider(
            "Interval živého obnovení (sekundy)",
            min_value=10,
            max_value=120,
            value=30,
            step=5,
        )
        st.caption("Plynulá aktualizace na pozadí bez problikávání nastavení.")
    else:
        auto_refresh_sec = None
        st.caption("Automatické obnovování vypnuto.")


# =====================================================================
# 3. Available Dates & Month Discovery
# =====================================================================

# Get distinct dates from SQLite
raw_dates = get_all_recorded_dates()
today_dt = dt.datetime.now()
today_raw = today_dt.strftime("%d.%m.%Y")
today_iso = today_dt.strftime("%Y-%m-%d")

# Map DD.MM.YYYY <-> YYYY-MM-DD
iso_to_raw: dict[str, str] = {}
for rd in raw_dates:
    try:
        iso_str = dt.datetime.strptime(rd, "%d.%m.%Y").strftime("%Y-%m-%d")
        iso_to_raw[iso_str] = rd
    except Exception:
        pass

# Ensure today is present in the list
if today_iso not in iso_to_raw:
    iso_to_raw[today_iso] = today_raw

available_dates = sorted(list(iso_to_raw.keys()))

# Group dates by month
available_months = sorted(list({d[:7] for d in available_dates}))
month_to_dates: dict[str, list[str]] = {}
for d in available_dates:
    month_to_dates.setdefault(d[:7], []).append(d)

# Initialize session state for active date (default to today / latest)
if "active_date_idx" not in st.session_state:
    st.session_state.active_date_idx = len(available_dates) - 1
if st.session_state.active_date_idx >= len(available_dates):
    st.session_state.active_date_idx = len(available_dates) - 1

current_date_str = available_dates[st.session_state.active_date_idx]
current_month_str = current_date_str[:7]
current_month_idx = available_months.index(current_month_str) if current_month_str in available_months else 0
current_day_num = int(current_date_str.split("-")[2])
raw_date_query = iso_to_raw.get(current_date_str, today_raw)


# =====================================================================
# 4. Top Header & Manual Refresh
# =====================================================================

head_col1, head_col2 = st.columns([3.5, 1.2])
with head_col1:
    st.title("⚡ Vizualizátor komerční arbitráže FVE & Baterie")
with head_col2:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Obnovit data", key="manual_refresh_btn", use_container_width=True):
        st.rerun()


# =====================================================================
# 5. Month & Day Navigation (Exact visualizer_web.py structure)
# =====================================================================

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
        f"<span style='color: #64748b;'>({days_in_month} dní v databázi)</span>"
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
        closest_idx = min(range(len(available_dates)), key=lambda i: abs((pd.to_datetime(available_dates[i]).date() - selected_date).days))
        if closest_idx != st.session_state.active_date_idx:
            st.session_state.active_date_idx = closest_idx
            st.rerun()

with nav_col4:
    total_days = len(available_dates)
    dates_in_curr_month = month_to_dates.get(current_month_str, [])
    day_in_month_idx = dates_in_curr_month.index(current_date_str) + 1 if current_date_str in dates_in_curr_month else 1
    is_today_badge = " (Dnes - Živý přenos ⚡)" if current_date_str == today_iso else ""
    st.markdown(
        f"<div style='padding-top: 6px; color: #94a3b8; font-size: 0.95rem;'>"
        f"📅 Den <b>{st.session_state.active_date_idx + 1}</b> z <b>{total_days}</b> "
        f"• Vybráno: <span style='color: #38bdf8; font-weight: bold;'>{current_date_str}</span> "
        f"<span style='color: #64748b;'>({day_in_month_idx}. v měsíci){is_today_badge}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# 6. Dynamic Live Dashboard (Fragmented Rendering)
# =====================================================================

def get_fragment_decorator(interval_sec):
    if hasattr(st, "fragment"):
        return st.fragment(run_every=interval_sec)
    elif hasattr(st, "experimental_fragment"):
        return st.experimental_fragment(run_every=interval_sec)
    def dummy_dec(func):
        return func
    return dummy_dec

fragment_interval = auto_refresh_sec if (current_date_str == today_iso and enable_auto_refresh) else None

@get_fragment_decorator(fragment_interval)
def render_live_dashboard(
    current_date_str: str,
    raw_date_query: str,
    today_iso: str,
    show_soc: bool,
    show_breakdown_costs: bool,
    auto_refresh_sec: int | None,
    enable_auto_refresh: bool,
):
    latest = get_latest_tick()
    now = dt.datetime.now()

    status_class = "status-offline"
    status_text = "🔴 OFFLINE (Žádná data)"
    if latest:
        try:
            last_dt = dt.datetime.fromisoformat(latest["timestamp"])
            age_seconds = (now - last_dt).total_seconds()
            if age_seconds < 150.0:
                status_class = "status-online"
                status_text = f"🟢 ONLINE ({int(age_seconds)}s | {latest['time_str']})"
            elif age_seconds < WATCHDOG_TIMEOUT_SEC:
                status_class = "status-warning"
                status_text = f"🟡 ZPOŽDĚNÍ ({int(age_seconds)}s | {latest['time_str']})"
            else:
                status_class = "status-offline"
                status_text = f"🔴 OFFLINE (> 5.5 min | Failsafe)"
        except Exception:
            pass

    stat_col1, stat_col2 = st.columns([3, 1])
    with stat_col1:
        if current_date_str == today_iso:
            refresh_note = (
                f" • Živá aktualizace každých {auto_refresh_sec}s (plynulý režim)"
                if enable_auto_refresh
                else " • Živé obnovování pozastaveno"
            )
            time_info = latest["time_str"] if latest else "-"
            st.markdown(
                f"<div style='padding-top: 6px; color: #94a3b8; font-size: 0.9rem;'>"
                f"📡 Poslední telemetrie FVE: <span style='color: #f8fafc; font-weight: 600;'>{time_info}</span>"
                f"<span style='color: #64748b;'>{refresh_note}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with stat_col2:
        st.markdown(
            f'<div style="text-align: right; padding-top: 2px;"><span class="status-pill {status_class}">{status_text}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    # =====================================================================
    # 6. Data Fetching & Handling Days Without Data
    # =====================================================================

    day_ticks = get_ticks_for_date(raw_date_query)

    if day_ticks.empty:
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.warning(
            f"⚠️ **Tento den ({current_date_str}) neobsahuje žádná telemetrická data** (bez záznamu z invertoru). "
            f"Tyto hodiny nejsou započítávány do nákladů ani do bilance."
        )

        # Empty KPI placeholders matching theme
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("💰 Čistý denní výsledek", "0.00 CZK", "-")
        kpi2.metric("🏠 Spotřeba domu", "0.00 kWh", "Nákup ze sítě: 0.00 kWh")
        kpi3.metric("☀️ Výroba FVE", "0.00 kWh", "Bez výroby")
        kpi4.metric("⚡ Prodej do sítě (B2G)", "0.00 kWh", "Export na spot")
        kpi5.metric("🔋 Využití baterie", "0.00 cyklu", "Vybito: 0.00 kWh")

        # Stop rendering further for empty day
        return


    # =====================================================================
    # 7. Energy Calculation & Daily KPIs (Identical to visualizer_web.py)
    # =====================================================================

    day_ticks["dt"] = pd.to_datetime(day_ticks["timestamp"], format="mixed", errors="coerce")
    day_ticks = day_ticks.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    # Instantaneous power in kW (smooth, independent of time gaps)
    day_ticks["pv_kw"] = day_ticks["p_pv_w"] / 1000.0
    day_ticks["load_kw"] = day_ticks["p_home_w"] / 1000.0
    day_ticks["charge_kw"] = day_ticks["p_bat_w"].apply(lambda p: abs(p) if p < 0 else 0.0) / 1000.0
    day_ticks["discharge_kw"] = day_ticks["p_bat_w"].apply(lambda p: p if p > 0 else 0.0) / 1000.0
    day_ticks["buy_kw"] = day_ticks["p_grid_w"].apply(lambda p: abs(p) if p < 0 else 0.0) / 1000.0
    day_ticks["sell_kw"] = day_ticks["p_grid_w"].apply(lambda p: p if p > 0 else 0.0) / 1000.0

    # Convert power (kW) and step duration (s) to accumulated energy (kWh)
    day_ticks["dt_hours"] = day_ticks["dt_seconds"] / 3600.0
    day_ticks["load_kwh"] = day_ticks["load_kw"] * day_ticks["dt_hours"]
    day_ticks["pv_kwh"] = day_ticks["pv_kw"] * day_ticks["dt_hours"]
    day_ticks["charge_kwh"] = day_ticks["charge_kw"] * day_ticks["dt_hours"]
    day_ticks["discharge_kwh"] = day_ticks["discharge_kw"] * day_ticks["dt_hours"]
    day_ticks["buy_kwh"] = day_ticks["buy_kw"] * day_ticks["dt_hours"]
    day_ticks["sell_kwh"] = day_ticks["sell_kw"] * day_ticks["dt_hours"]

    day_ticks["soc_kwh"] = (day_ticks["sim_soc"] / 100.0) * BATTERY_CAPACITY_KWH
    day_ticks["spot_price_czk"] = day_ticks["price_czk"]
    day_ticks["p_buy"] = (day_ticks["price_czk"] + IMPORT_FEE_CZK_PER_KWH) * (1.0 + VAT_RATE)
    day_ticks["p_sell"] = day_ticks["price_czk"] - EXPORT_FEE_CZK_PER_KWH
    day_ticks["net_profit_czk"] = day_ticks["profit_step_czk"]

    # Sums
    load_sum = float(day_ticks["load_kwh"].sum())
    pv_sum = float(day_ticks["pv_kwh"].sum())
    buy_sum = float(day_ticks["buy_kwh"].sum())
    sell_sum = float(day_ticks["sell_kwh"].sum())
    dis_sum = float(day_ticks["discharge_kwh"].sum())
    cycles = dis_sum / 8.192  # Fox-ESS HV2600 HS10.4: 10.24 kWh * 0.80 usable
    profit_sum = float(day_ticks["net_profit_czk"].sum())

    # 5 Top KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        delta_str = "Zisk" if profit_sum >= 0 else "Náklad"
        st.metric("💰 Čistý denní výsledek", f"{profit_sum:+.2f} CZK", delta=f"{delta_str}")
    with kpi2:
        st.metric("🏠 Spotřeba domu", f"{load_sum:.2f} kWh", delta=f"Nákup ze sítě: {buy_sum:.2f} kWh", delta_color="inverse")
    with kpi3:
        st.metric("☀️ Výroba FVE", f"{pv_sum:.2f} kWh", delta="100% využito bez omezení")
    with kpi4:
        st.metric("⚡ Prodej do sítě (B2G)", f"{sell_sum:.2f} kWh", delta="Export na spot")
    with kpi5:
        st.metric("🔋 Využití baterie", f"{cycles:.2f} cyklu", delta=f"Vybito: {dis_sum:.2f} kWh")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


    # =====================================================================
    # 8. Synchronized Plotly Subplots (Exact visualizer_web.py graphics)
    # =====================================================================

    # Prepare plotting DataFrame with explicit gap breaks (inserting NaN rows when gap > 600s / 10m)
    GAP_THRESHOLD_SEC = 600.0  # 10 minutes

    # Segment tagging for gaps > 600s to allow independent polygon fills
    seg_id = 0
    seg_ids = [0]
    for i in range(1, len(day_ticks)):
        gap_sec = (day_ticks.iloc[i]["dt"] - day_ticks.iloc[i - 1]["dt"]).total_seconds()
        if gap_sec > GAP_THRESHOLD_SEC:
            seg_id += 1
        seg_ids.append(seg_id)
    day_ticks["segment"] = seg_ids

    plot_rows = []
    for i in range(len(day_ticks)):
        if i > 0:
            gap_sec = (day_ticks.iloc[i]["dt"] - day_ticks.iloc[i - 1]["dt"]).total_seconds()
            if gap_sec > GAP_THRESHOLD_SEC:
                # 1. NaN row right after gap start to break the preceding segment
                row_nan1 = {col: np.nan for col in day_ticks.columns}
                row_nan1["dt"] = day_ticks.iloc[i - 1]["dt"] + pd.Timedelta(seconds=1)
                plot_rows.append(row_nan1)
                # 2. NaN row right before gap end to prevent connecting to next segment
                row_nan2 = {col: np.nan for col in day_ticks.columns}
                row_nan2["dt"] = day_ticks.iloc[i]["dt"] - pd.Timedelta(seconds=1)
                plot_rows.append(row_nan2)
        plot_rows.append(day_ticks.iloc[i].to_dict())

    plot_df = pd.DataFrame(plot_rows)

    # Split net profit into positive (green) and negative (red) in CZK
    plot_df["profit_pos_czk"] = plot_df["net_profit_czk"].apply(lambda v: v if (pd.notna(v) and v > 0) else np.nan)
    plot_df["profit_neg_czk"] = plot_df["net_profit_czk"].apply(lambda v: v if (pd.notna(v) and v < 0) else np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>⚡ Výkon energetických toků (kW) a kapacita baterie (kWh)</b>",
            "<b>💰 Tržní spotové ceny OTE a finanční bilance (CZK)</b>",
        ),
        row_heights=[0.58, 0.42],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    times = plot_df["dt"]

    # -------------------------------------------------------------
    # SUBPLOT 1: POWER FLOWS (kW) & BATTERY CAPACITY (kWh)
    # -------------------------------------------------------------

    # 1. Solar PV - Amber Gold (segmented to prevent Plotly's tozeroy fill from bleeding across gaps)
    for k, (s_id, seg_df) in enumerate(day_ticks.groupby("segment", sort=True)):
        fig.add_trace(
            go.Scatter(
                x=seg_df["dt"],
                y=seg_df["pv_kw"],
                name="Výroba FVE",
                legendgroup="Výroba FVE",
                showlegend=(k == 0),
                line=dict(color="#eab308", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(234, 179, 8, 0.12)",
                connectgaps=False,
                hovertemplate="FVE: %{y:.2f} kW<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # 2. House Load - Strong Blue
    fig.add_trace(
        go.Scatter(
            x=times,
            y=plot_df["load_kw"],
            name="Spotřeba domu",
            line=dict(color="#0284c7", width=2.5),
            connectgaps=False,
            hovertemplate="Dům: %{y:.2f} kW<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 3. Battery Charge - Emerald Green
    fig.add_trace(
        go.Scatter(
            x=times,
            y=plot_df["charge_kw"],
            name="Nabíjení baterie",
            line=dict(color="#10b981", width=2.0),
            connectgaps=False,
            hovertemplate="Nabíjení: %{y:.2f} kW<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 4. Battery Discharge - Violet Purple
    fig.add_trace(
        go.Scatter(
            x=times,
            y=plot_df["discharge_kw"],
            name="Vybíjení baterie",
            line=dict(color="#8b5cf6", width=2.2),
            connectgaps=False,
            hovertemplate="Vybíjení: %{y:.2f} kW<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 5. Grid Buy - Crimson Red
    fig.add_trace(
        go.Scatter(
            x=times,
            y=plot_df["buy_kw"],
            name="Nákup ze sítě",
            line=dict(color="#ef4444", width=1.8),
            connectgaps=False,
            hovertemplate="Nákup: %{y:.2f} kW<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 6. Grid Sell (B2G Export) - Hot Pink / Magenta
    fig.add_trace(
        go.Scatter(
            x=times,
            y=plot_df["sell_kw"],
            name="Prodej do sítě (B2G)",
            line=dict(color="#ec4899", width=2.5),
            connectgaps=False,
            hovertemplate="Prodej: %{y:.2f} kW<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 7. Battery SoC (Secondary Y-axis, 0 - 10.5 kWh) - Silver Platinum
    if show_soc:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=plot_df["soc_kwh"],
                name="Stav nabití (SoC)",
                line=dict(color="#cbd5e1", width=2.2, dash="dashdot"),
                connectgaps=False,
                hovertemplate="SoC: %{y:.2f} kWh (%{text:.1f}%)<extra></extra>",
                text=plot_df["sim_soc"],
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    # -------------------------------------------------------------
    # SUBPLOT 2: SPOT PRICES & FINANCIALS (CZK)
    # -------------------------------------------------------------

    # Spot Price OTE (Gold)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=plot_df["spot_price_czk"],
            name="Spot OTE (CZK/kWh)",
            line=dict(color="#fbbf24", width=2.2),
            connectgaps=False,
            hovertemplate="Spot OTE: %{y:.2f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Effective Buy Price
    if show_breakdown_costs:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=plot_df["p_buy"],
                name="Nákupní cena (vč. distribuce a DPH)",
                line=dict(color="#f87171", width=1.5, dash="dot"),
                connectgaps=False,
                hovertemplate="Nákupní cena: %{y:.2f} CZK<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=plot_df["p_sell"],
                name="Výkupní cena (čistá)",
                line=dict(color="#34d399", width=1.5, dash="dot"),
                connectgaps=False,
                hovertemplate="Výkupní cena: %{y:.2f} CZK<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Step Financial Balance (Green for Profit / Sell, Red for Cost / Buy) in CZK
    fig.add_trace(
        go.Bar(
            x=times,
            y=plot_df["profit_pos_czk"],
            name="Zisk (CZK)",
            marker_color="#22c55e",
            width=45000,
            opacity=0.85,
            hovertemplate="Zisk: +%{y:.3f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=times,
            y=plot_df["profit_neg_czk"],
            name="Náklad (CZK)",
            marker_color="#ef4444",
            width=45000,
            opacity=0.85,
            hovertemplate="Náklad: %{y:.3f} CZK<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Zero line on financial plot
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1, row=2, col=1)

    # Check for gaps > 10 minutes between ticks and highlight them with gray bands
    if len(day_ticks) > 1:
        dts = day_ticks["dt"].tolist()
        for i in range(1, len(dts)):
            gap_sec = (dts[i] - dts[i - 1]).total_seconds()
            if gap_sec > 600.0:  # gap greater than 10 minutes
                fig.add_vrect(
                    x0=dts[i - 1],
                    x1=dts[i],
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

    # Layout styling matching visualizer_web.py
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0f172a",
        font=dict(family="Segoe UI, Inter, sans-serif", color="#e2e8f0"),
        height=720,
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
        barmode="relative",
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
        title_text="Výkon toků (kW)",
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


    # =====================================================================
    # 9. Recent Commands Log Table (for Current Day)
    # =====================================================================

    if current_date_str == today_iso:
        st.markdown("### 📋 Poslední Odeslané Příkazy Střídači (Dnes)")
        display_cols = ["time_str", "price_czk", "p_home_w", "p_pv_w", "sim_soc", "command", "power_w", "reason", "profit_step_czk"]
        table_df = (
            day_ticks[display_cols]
            .tail(12)
            .iloc[::-1]
            .rename(
                columns={
                    "time_str": "Čas",
                    "price_czk": "Cena (CZK)",
                    "p_home_w": "Dům (W)",
                    "p_pv_w": "FVE (W)",
                    "sim_soc": "SoC (%)",
                    "command": "Příkaz Střídači",
                    "power_w": "Výkon (W)",
                    "reason": "Důvod Rozhodnutí",
                    "profit_step_czk": "Krok Zisk (CZK)",
                }
            )
        )
        try:
            st.dataframe(table_df, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(table_df, use_container_width=True, hide_index=True)

        # 10. Tomorrow's Strategy & Forecast Plan
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        with st.expander("🔮 Zobrazit plán a předpověď arbitráže na zítřek (Spot & FVE)", expanded=False):
            df_plan, summary_plan, _ = compute_tomorrow_plan()
            if df_plan is not None and summary_plan is not None:
                st.markdown(f"#### 📅 Předpověď a optimální plán pro zítřek: **{summary_plan['date_str']}**")
                p_c1, p_c2, p_c3, p_c4 = st.columns(4)
                p_c1.metric("☀️ Očekávaná výroba FVE", f"{summary_plan['total_pv_kwh']:.2f} kWh")
                p_c2.metric("🏠 Očekávaná spotřeba domu", f"{summary_plan['total_load_kwh']:.2f} kWh")
                p_c3.metric("📉 Nejlevnější hodina (nákup)", f"{summary_plan['min_price']:.2f} CZK", delta=summary_plan['min_hour'], delta_color="inverse")
                p_c4.metric("📈 Nejdražší hodina (špička)", f"{summary_plan['max_price']:.2f} CZK", delta=summary_plan['max_hour'])
                
                try:
                    st.dataframe(df_plan, width="stretch", hide_index=True)
                except TypeError:
                    st.dataframe(df_plan, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Ceny OTE ČR pro zítřejší den zatím nejsou k dispozici. Výsledky denního trhu se zveřejňují každý den ve 13:15 CET.")




# =====================================================================
# 7. Render Fragment & Compatibility Fallback
# =====================================================================

render_live_dashboard(
    current_date_str=current_date_str,
    raw_date_query=raw_date_query,
    today_iso=today_iso,
    show_soc=show_soc,
    show_breakdown_costs=show_breakdown_costs,
    auto_refresh_sec=auto_refresh_sec,
    enable_auto_refresh=enable_auto_refresh,
)

# Fallback auto-refresh only for older Streamlit versions without st.fragment
if not hasattr(st, "fragment") and not hasattr(st, "experimental_fragment"):
    if current_date_str == today_iso and enable_auto_refresh and auto_refresh_sec:
        time.sleep(auto_refresh_sec)
        st.rerun()
