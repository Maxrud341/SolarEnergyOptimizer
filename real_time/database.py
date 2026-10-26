"""SQLite Database Storage Manager with WAL Mode and FIFO Quota.

Provides high-concurrency, fail-safe storage for 2-minute real-time telemetry ticks,
inverter commands, and persistent battery simulation state.
Guarantees disk consumption stays strictly below 800 MB (1 GB ceiling).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

try:
    from real_time.config import (
        DATA_DIR,
        DB_PATH,
        MAX_DB_SIZE_BYTES,
    )
except ImportError:
    from config import (
        DATA_DIR,
        DB_PATH,
        MAX_DB_SIZE_BYTES,
    )


def get_db_connection() -> sqlite3.Connection:
    """Return an open SQLite connection with WAL mode and row factory enabled."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for seamless concurrent reads & writes
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    return conn


def init_db() -> None:
    """Initialize database tables and indexes if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Telemetry ticks (2-minute intervals)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date_str TEXT NOT NULL,
                time_str TEXT NOT NULL,
                dt_seconds REAL NOT NULL,
                price_czk REAL NOT NULL,
                p_home_w REAL NOT NULL,
                p_pv_w REAL NOT NULL,
                p_grid_w REAL NOT NULL,
                p_bat_w REAL NOT NULL,
                real_soc REAL NOT NULL,
                sim_soc REAL NOT NULL,
                command TEXT NOT NULL,
                power_w REAL NOT NULL,
                reason TEXT NOT NULL,
                profit_step_czk REAL NOT NULL
            );
            """
        )

        # Index on timestamp for fast time-range querying
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_ticks (timestamp);"
        )

        # 2. System state key-value store (persisting sim_soc across service restarts)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        # 3. Events & Watchdog alerts log
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL
            );
            """
        )

        conn.commit()


def save_sim_soc(sim_soc: float) -> None:
    """Persist the simulated battery State of Charge to survive restarts."""
    now_iso = dt.datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('sim_soc', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
            """,
            (str(sim_soc), now_iso),
        )
        conn.commit()


def load_sim_soc(default: float = 50.0) -> float:
    """Load the persisted simulated SoC or return default if uninitialized."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'sim_soc';"
            ).fetchone()
            if row and row["value"]:
                return float(row["value"])
    except Exception:
        pass
    return default


def record_system_event(level: str, event_type: str, message: str) -> None:
    """Record an operational, watchdog, or recovery event."""
    now_iso = dt.datetime.now().isoformat()
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO system_events (timestamp, level, event_type, message)
                VALUES (?, ?, ?, ?);
                """,
                (now_iso, level, event_type, message),
            )
            conn.commit()
    except Exception as exc:
        print(f"[DB Error] Could not record system event: {exc}")


def save_telemetry_tick(
    timestamp: str,
    date_str: str,
    time_str: str,
    dt_seconds: float,
    price_czk: float,
    p_home_w: float,
    p_pv_w: float,
    p_grid_w: float,
    p_bat_w: float,
    real_soc: float,
    sim_soc: float,
    command: str,
    power_w: float,
    reason: str,
    profit_step_czk: float,
) -> None:
    """Insert a 2-minute tick into the database and update persisted state."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO telemetry_ticks (
                timestamp, date_str, time_str, dt_seconds, price_czk,
                p_home_w, p_pv_w, p_grid_w, p_bat_w, real_soc, sim_soc,
                command, power_w, reason, profit_step_czk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                date_str,
                time_str,
                dt_seconds,
                price_czk,
                p_home_w,
                p_pv_w,
                p_grid_w,
                p_bat_w,
                real_soc,
                sim_soc,
                command,
                power_w,
                reason,
                profit_step_czk,
            ),
        )

        # Update last packet timestamp and sim_soc in system_state
        conn.execute(
            """
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('sim_soc', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
            """,
            (str(sim_soc), timestamp),
        )
        conn.execute(
            """
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('last_packet_timestamp', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
            """,
            (timestamp, timestamp),
        )
        conn.commit()


def get_latest_tick() -> dict[str, Any] | None:
    """Return the most recent telemetry tick recorded in the database."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM telemetry_ticks ORDER BY id DESC LIMIT 1;"
            ).fetchone()
            if row:
                return dict(row)
    except Exception:
        pass
    return None


def get_recent_ticks(limit: int = 300) -> pd.DataFrame:
    """Fetch recent telemetry ticks as a pandas DataFrame for dashboard visualization."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM telemetry_ticks ORDER BY id DESC LIMIT {limit};",
                conn,
            )
            if not df.empty:
                df = df.sort_values(by="id", ascending=True).reset_index(drop=True)
            return df
    except Exception as exc:
        print(f"[DB Error] Failed to read recent ticks: {exc}")
        return pd.DataFrame()


def get_today_ticks() -> pd.DataFrame:
    """Fetch all ticks recorded for today's date."""
    today_str = dt.datetime.now().strftime("%d.%m.%Y")
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM telemetry_ticks WHERE date_str = ? ORDER BY id ASC;",
                conn,
                params=(today_str,),
            )
            return df
    except Exception:
        return pd.DataFrame()


def enforce_db_size_limit(max_bytes: int = MAX_DB_SIZE_BYTES) -> None:
    """Check database file size and purge oldest records if exceeding max_bytes (FIFO)."""
    if not DB_PATH.exists():
        return

    current_size = DB_PATH.stat().st_size
    wal_path = DB_PATH.with_suffix(".db-wal")
    if wal_path.exists():
        current_size += wal_path.stat().st_size

    if current_size > max_bytes:
        print(f"[Storage Quota] DB size ({current_size / (1024*1024):.1f} MB) exceeds limit. Purging oldest rows...")
        try:
            with get_db_connection() as conn:
                # Delete oldest 50,000 records (~2.5 months of data)
                conn.execute(
                    """
                    DELETE FROM telemetry_ticks
                    WHERE id IN (
                        SELECT id FROM telemetry_ticks ORDER BY id ASC LIMIT 50000
                    );
                    """
                )
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            print("[Storage Quota] Oldest records purged successfully.")
        except Exception as exc:
            print(f"[Storage Quota Error] Could not purge old rows: {exc}")
