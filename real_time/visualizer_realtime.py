"""Interactive Real-Time Streamlit Web Dashboard for Live MQTT Telemetry.

Reads live ticks from SQLite (telemetry.db), displays a real-time status indicator
(🟢 ONLINE / 🟡 ZPOŽDĚNÍ / 🔴 OFFLINE), animated KPI cards, synchronized Plotly
charts of power flows and spot prices, and live inverter command log.

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
        MAX_DB_SIZE_BYTES,
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        MQTT_TOPIC_RX,
        MQTT_TOPIC_TX,
        SOC_MAX,
        SOC_MIN,
        WATCHDOG_TIMEOUT_SEC,
    )
    from real_time.database import (
        get_latest_tick,
        get_recent_ticks,
        get_today_ticks,
        init_db,
    )
except ImportError:
    from config import (
        BATTERY_CAPACITY_KWH,
        BATTERY_POWER_MAX_KW,
        DB_PATH,
        MAX_DB_SIZE_BYTES,
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        MQTT_TOPIC_RX,
        MQTT_TOPIC_TX,
        SOC_MAX,
        SOC_MIN,
        WATCHDOG_TIMEOUT_SEC,
    )
    from database import (
        get_latest_tick,
        get_recent_ticks,
        get_today_ticks,
        init_db,
    )

# =====================================================================
# 1. Page Configuration & Microgrid Dark Styling
# =====================================================================

st.set_page_config(
    page_title="Živý Dispečink FVE | Real-Time MQTT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark microgrid control center theme matching visualizer_web.py */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    header, footer {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 0.8rem 1.1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
    }
    /* Status banner pill */
    .status-pill {
        display: inline-block;
        padding: 0.4rem 1.0rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
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


# =====================================================================
# 2. Sidebar Information & Diagnostics
# =====================================================================

with st.sidebar:
    st.markdown("### ⚡ Parametry Systému")
    st.markdown(
        f"""
        * **Baterie:** Fox-ESS HV2600 HS 10.4
        * **Kapacita:** `{BATTERY_CAPACITY_KWH} kWh`
        * **Limity:** `SoC {int(SOC_MIN*100)}% – {int(SOC_MAX*100)}%`
        * **Max. výkon:** `{BATTERY_POWER_MAX_KW} kW`
        * **MQTT Broker:** `{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}`
        * **Inbound TX:** `{MQTT_TOPIC_TX}`
        * **Outbound RX:** `{MQTT_TOPIC_RX}`
        """
    )
    st.markdown("---")
    st.markdown("### 💾 Úložiště SQLite (Limit 800 MB)")
    if DB_PATH.exists():
        db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        pct_used = (DB_PATH.stat().st_size / MAX_DB_SIZE_BYTES) * 100.0
        st.write(f"Velikost DB: **{db_size_mb:.2f} MB** ({pct_used:.2f}%)")
        st.progress(min(1.0, pct_used / 100.0))
    else:
        st.write("DB zatím nevytvořena.")

    st.markdown("---")
    auto_refresh_sec = st.slider("Interval obnovení stránky (sekundy)", min_value=2, max_value=30, value=5)
    st.caption("Stránka se automaticky překresluje při každém novém tiku.")


# =====================================================================
# 3. Live Header & Status Indicator
# =====================================================================

latest = get_latest_tick()
now = dt.datetime.now()

status_class = "status-offline"
status_text = "🔴 OFFLINE / PORUCHA SPOJENÍ (Žádná data v databázi)"
age_seconds = 9999.0

if latest:
    try:
        last_dt = dt.datetime.fromisoformat(latest["timestamp"])
        age_seconds = (now - last_dt).total_seconds()
        if age_seconds < 150.0:
            status_class = "status-online"
            status_text = f"🟢 ONLINE (Poslední telemetrie před {int(age_seconds)} s | {latest['time_str']})"
        elif age_seconds < WATCHDOG_TIMEOUT_SEC:
            status_class = "status-warning"
            status_text = f"🟡 ZPOŽDĚNÍ SPOJENÍ (Čekání na data {int(age_seconds)} s | Poslední: {latest['time_str']})"
        else:
            status_class = "status-offline"
            status_text = f"🔴 OFFLINE / WATCHDOG AKTIVNÍ (Výpadek {int(age_seconds/60)} min | Aktivován bezpečný režim)"
    except Exception:
        pass

col_title, col_stat = st.columns([2, 1])
with col_title:
    st.title("⚡ Živý Dispečink FVE a Baterie")
    st.caption("Reálný čas: autonomní řízení střídače Fox-ESS, spotový trh OTE ČR a ochrana sítě")

with col_stat:
    st.markdown(f'<div style="text-align: right;"><span class="status-pill {status_class}">{status_text}</span></div>', unsafe_allow_html=True)


# =====================================================================
# 4. KPI Top Metrics
# =====================================================================

m1, m2, m3, m4, m5 = st.columns(5)

if latest:
    cmd_name = latest["command"]
    cmd_pwr = latest["power_w"]
    price = latest["price_czk"]
    p_home = latest["p_home_w"]
    p_pv = latest["p_pv_w"]
    sim_soc = latest["sim_soc"]
    real_soc = latest["real_soc"]

    # Compute today's profit sum
    today_df = get_today_ticks()
    today_profit = today_df["profit_step_czk"].sum() if not today_df.empty else 0.0

    m1.metric("Aktuální Příkaz", f"{cmd_name}", f"{int(cmd_pwr)} W")
    m2.metric("Simulovaný SoC", f"{sim_soc:.1f} %", f"Fyzický: {real_soc:.0f}%")
    m3.metric("Spotová Cena OTE", f"{price:.2f} CZK/kWh", f"{'Záporná/Nízká' if price < 2.0 else ('Vysoká špička' if price > 5.0 else 'Běžná')}")
    m4.metric("Spotřeba Domu / FVE", f"{int(p_home)} W", f"FVE: {int(p_pv)} W")
    profit_delta = f"{'+' if today_profit >= 0 else ''}{today_profit:.2f} CZK"
    m5.metric("Dnešní Arbitrážní Bilance", profit_delta, "Čistý zisk za dnešek")
else:
    m1.metric("Aktuální Příkaz", "Čekání na data", "-")
    m2.metric("Simulovaný SoC", "50.0 %", "-")
    m3.metric("Spotová Cena OTE", "- CZK/kWh", "-")
    m4.metric("Spotřeba Domu / FVE", "- W", "-")
    m5.metric("Dnešní Arbitrážní Bilance", "0.00 CZK", "-")


# =====================================================================
# 5. Interactive Real-Time Plotly Charts
# =====================================================================

ticks_df = get_recent_ticks(limit=180)  # last ~6 hours of 2-minute ticks

if not ticks_df.empty:
    ticks_df["dt"] = pd.to_datetime(ticks_df["timestamp"], format="mixed", errors="coerce")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=("Okamžité Toky Výkonu (W) a Stav Baterie (%)", "Spotová Cena OTE ČR (CZK/kWh) a Příkazy"),
    )

    # Subplot 1: Power flows & Battery SoC
    fig.add_trace(
        go.Scatter(
            x=ticks_df["dt"],
            y=ticks_df["p_home_w"],
            name="Spotřeba Domu (W)",
            line=dict(color="#fb923c", width=2),
            hovertemplate="%{y:.0f} W",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ticks_df["dt"],
            y=ticks_df["p_pv_w"],
            name="Výroba FVE (W)",
            line=dict(color="#facc15", width=2),
            fill="tozeroy",
            fillcolor="rgba(250, 204, 21, 0.15)",
            hovertemplate="%{y:.0f} W",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ticks_df["dt"],
            y=ticks_df["p_bat_w"],
            name="Baterie Výkon (W)",
            line=dict(color="#2dd4bf", width=2.5),
            hovertemplate="%{y:.0f} W (+ vybíjení, - nabíjení)",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ticks_df["dt"],
            y=ticks_df["p_grid_w"],
            name="Síť Import/Export (W)",
            line=dict(color="#a855f7", width=1.5, dash="dot"),
            hovertemplate="%{y:.0f} W",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    # Secondary Y-axis: SoC (%)
    fig.add_trace(
        go.Scatter(
            x=ticks_df["dt"],
            y=ticks_df["sim_soc"],
            name="Simulovaný SoC (%)",
            line=dict(color="#38bdf8", width=3, dash="dash"),
            hovertemplate="%{y:.1f} %",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Subplot 2: Spot Prices & Command Overlay
    fig.add_trace(
        go.Scatter(
            x=ticks_df["dt"],
            y=ticks_df["price_czk"],
            name="Spotová Cena (CZK/kWh)",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.10)",
            hovertemplate="%{y:.2f} CZK/kWh",
        ),
        row=2,
        col=1,
    )

    # Layout styling matching dark control center
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0b1120",
        paper_bgcolor="#0b1120",
        height=580,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Výkon (W)", row=1, col=1, secondary_y=False, gridcolor="#1e293b")
    fig.update_yaxes(title_text="SoC (%)", range=[0, 105], row=1, col=1, secondary_y=True, gridcolor="#1e293b")
    fig.update_yaxes(title_text="Cena (CZK)", row=2, col=1, gridcolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("💡 V databázi zatím nejsou žádná data. Spusťte `python real_time/mqtt_service.py` (nebo s příznakem `--mock`) pro zahájení sběru dat.")


# =====================================================================
# 6. Live Telemetry & Commands Log Table
# =====================================================================

st.markdown("### 📋 Poslední Odeslané Příkazy Střídači")
if not ticks_df.empty:
    display_cols = ["time_str", "price_czk", "p_home_w", "p_pv_w", "sim_soc", "command", "power_w", "reason", "profit_step_czk"]
    recent_table = (
        ticks_df[display_cols]
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
    st.dataframe(recent_table, use_container_width=True, hide_index=True)


# =====================================================================
# 7. Auto-refresh Hook
# =====================================================================

time.sleep(auto_refresh_sec)
st.rerun()
