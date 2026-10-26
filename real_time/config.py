"""Real-Time Configuration and Physical System Constants.

All physical parameters (Fox-ESS battery, PV system), operational timeouts,
storage quotas, and API endpoints are centralized here.
Sensitive variables (passwords, tokens) are read from the .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from real_time/.env or project root .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()


# =====================================================================
# 1. Hardware Calibration: Fox-ESS HV2600 HS 10.4 Battery
# =====================================================================

BATTERY_CAPACITY_KWH: float = 10.24
SOC_MIN: float = 0.20          # 20% minimum depth of discharge (DoD 80%)
SOC_MAX: float = 1.00          # 100% maximum charge limit
BATTERY_POWER_MAX_KW: float = 5.0  # 5.0 kW continuous charge/discharge rating
BATTERY_POWER_MAX_W: float = 5000.0

EFFICIENCY_CHARGE: float = 0.975
EFFICIENCY_DISCHARGE: float = 0.975
ROUNDTRIP_EFFICIENCY: float = EFFICIENCY_CHARGE * EFFICIENCY_DISCHARGE  # ~95.06%

DEGRADATION_COST_CZK_PER_KWH: float = 0.20  # LFP cell aging cost (~6000 cycles)

# Czech spot electricity market tariff structure
IMPORT_FEE_CZK_PER_KWH: float = 1.80   # Czech distribution grid fee (ČEZ/EG.D)
VAT_RATE: float = 0.21                 # Czech VAT (DPH 21%)
EXPORT_FEE_CZK_PER_KWH: float = 0.45   # Trader export deduction fee
GRID_LIMIT_KW: float = 10.0            # Point of Common Coupling (PCC) physical limit

# Mathematical Optimization (MPC & Safety Margins)
MPC_HORIZON_HOURS: int = 24
FORECAST_PV_SAFETY_FACTOR: float = 0.80    # 20% conservative uncertainty discount
FORECAST_LOAD_SAFETY_FACTOR: float = 1.15  # 15% safety buffer on home consumption
PREFLUSH_MIN_SURPLUS_KWH: float = 3.0
PREFLUSH_MIN_SELL_PRICE_CZK: float = 2.00
ALLOW_CURTAILMENT: bool = True



# =====================================================================
# 2. Solar Photovoltaic (PV) Plant Constants
# =====================================================================

PV_CAPACITY_KWP: float = 6.0    # Nominal peak solar array rating (kWp)
PV_PERFORMANCE_RATIO: float = 0.82  # System balance derating (inverter, wiring, soiling)
PV_TEMP_COEFF_PER_C: float = -0.0038  # Power loss above 25°C cell temperature (-0.38%/°C)


# =====================================================================
# 3. MQTT Broker Settings (cassandra2.tul.cz)
# =====================================================================

MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "cassandra2.tul.cz")
MQTT_BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USERNAME: str | None = os.getenv("MQTT_USERNAME") or None
MQTT_PASSWORD: str | None = os.getenv("MQTT_PASSWORD") or None

MQTT_TOPIC_TX: str = os.getenv("MQTT_TOPIC_TX", "FVE/Maly_TX")  # Incoming telemetry
MQTT_TOPIC_RX: str = os.getenv("MQTT_TOPIC_RX", "FVE/Maly_RX")  # Outgoing inverter commands
MQTT_KEEPALIVE_SEC: int = 60


# =====================================================================
# 4. Synchronization, Jitter & Watchdog Timeouts
# =====================================================================

NOMINAL_STEP_SEC: int = 120           # 2 minutes nominal telemetry step
MIN_VALID_STEP_SEC: int = 15          # Reject packets arriving faster than 15s (duplicate spam)
MAX_VALID_STEP_SEC: int = 3600        # Maximum dynamic delta t before forcing a resync (1 hour)
STALE_PACKET_MAX_AGE_SEC: int = 300   # Discard buffered packets older than 5 minutes on reconnect

# Watchdog fires if no packet arrives within 330 seconds (~5.5 minutes / ~3 missed packets)
WATCHDOG_TIMEOUT_SEC: int = 330


# =====================================================================
# 5. Czech Inverter Command Vocabulary
# =====================================================================

class CzechCommand:
    CHARGE_GRID = "Nabíjet ze sítě"
    DISCHARGE_GRID = "Vybíjet do sítě"
    COVER_HOME = "Pokrytí spotřeby domu"
    CHARGE_PV = "Nabíjet z FVE"
    STANDBY = "Nečinnost"

class CzechReason:
    LOW_PRICE = "Nízká / záporná cena elektřiny"
    HIGH_PRICE = "Vysoká cena (arbitrážní prodej do sítě)"
    HOME_SAVING = "Úspora nákupu (pokrytí domácí spotřeby)"
    PV_SURPLUS = "Ukládání přebytku sluneční výroby"
    BATTERY_PROTECT_MIN = "Ochrana baterie: dosažen limit vybití (SoC 20%)"
    BATTERY_PROTECT_MAX = "Ochrana baterie: plně nabito (SoC 100%)"
    STANDBY_SCHEDULE = "Udržování kapacity pro dražší večerní hodiny"
    FAILSAFE_TIMEOUT = "Bezpečnostní režim: ztráta telemetrie (> 5 min)"
    MANUAL_OVERRIDE = "Manuální zásah operátora"


# =====================================================================
# 6. Database and Storage Quota (< 1 GB)
# =====================================================================

DATA_DIR: Path = Path(__file__).parent / "data"
DB_PATH: Path = DATA_DIR / "telemetry.db"

# FIFO limit: 800 MB triggers automatic purge of oldest records
MAX_DB_SIZE_BYTES: int = 800 * 1024 * 1024


# =====================================================================
# 7. External API Endpoints & Coordinates
# =====================================================================

LOCATION_LATITUDE: float = float(os.getenv("LOCATION_LATITUDE", "50.767"))
LOCATION_LONGITUDE: float = float(os.getenv("LOCATION_LONGITUDE", "15.056"))
LOCATION_TIMEZONE: str = os.getenv("LOCATION_TIMEZONE", "Europe/Prague")

# Spot prices (Czech Day-Ahead OTE market)
SPOTOVKY_TODAY_URL: str = "https://api.spotovky.cz/v2/prices/today"
SPOTOVKY_TOMORROW_URL: str = "https://api.spotovky.cz/v2/prices/tomorrow"
ENERGY_CHARTS_URL: str = "https://api.energy-charts.info/price?bzn=CZ"

# Weather & Solar radiation (Open-Meteo free API)
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
