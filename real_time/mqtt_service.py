"""Production Real-Time MQTT Inverter Service and Telemetry Daemon.

Subscribes to incoming telemetry on FVE/Maly_TX (cassandra2.tul.cz),
runs dynamic battery simulation with exact delta t integration,
dispatches optimal arbitrage commands in Czech JSON to FVE/Simulated_RX,
safeguards against communication dropouts via a dedicated Watchdog thread,
and records telemetry into SQLite (telemetry.db).
Supports --mock mode for offline testing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt

try:
    from real_time.config import (
        CzechCommand,
        CzechReason,
        MAX_VALID_STEP_SEC,
        MIN_VALID_STEP_SEC,
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        MQTT_KEEPALIVE_SEC,
        MQTT_PASSWORD,
        MQTT_TOPIC_RX,
        MQTT_TOPIC_TX,
        MQTT_USERNAME,
        NOMINAL_STEP_SEC,
        STALE_PACKET_MAX_AGE_SEC,
        WATCHDOG_TIMEOUT_SEC,
    )
    from real_time.controller import RealTimeController
    from real_time.database import (
        enforce_db_size_limit,
        init_db,
        load_sim_soc,
        record_system_event,
        save_sim_soc,
        save_telemetry_tick,
    )
except ImportError:
    from config import (
        CzechCommand,
        CzechReason,
        MAX_VALID_STEP_SEC,
        MIN_VALID_STEP_SEC,
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        MQTT_KEEPALIVE_SEC,
        MQTT_PASSWORD,
        MQTT_TOPIC_RX,
        MQTT_TOPIC_TX,
        MQTT_USERNAME,
        NOMINAL_STEP_SEC,
        STALE_PACKET_MAX_AGE_SEC,
        WATCHDOG_TIMEOUT_SEC,
    )
    from controller import RealTimeController
    from database import (
        enforce_db_size_limit,
        init_db,
        load_sim_soc,
        record_system_event,
        save_sim_soc,
        save_telemetry_tick,
    )

# Setup robust logging supporting both console and pythonw background daemon
handlers: list[logging.Handler] = []
if sys.stderr is not None:
    handlers.append(logging.StreamHandler(sys.stderr))

try:
    from logging.handlers import RotatingFileHandler
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handlers.append(RotatingFileHandler(DATA_DIR / "service.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"))
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=handlers or [logging.NullHandler()],
)
logger = logging.getLogger("MQTTService")


class MQTTRealtimeService:
    """Production service managing MQTT communication, battery simulation, and watchdog."""

    def __init__(
        self,
        broker_host: str = MQTT_BROKER_HOST,
        broker_port: int = MQTT_BROKER_PORT,
        topic_tx: str = MQTT_TOPIC_TX,
        topic_rx: str = MQTT_TOPIC_RX,
        username: str | None = MQTT_USERNAME,
        password: str | None = MQTT_PASSWORD,
        mock_mode: bool = False,
        mock_interval_sec: float = 2.0,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_tx = topic_tx
        self.topic_rx = topic_rx
        self.username = username
        self.password = password
        self.mock_mode = mock_mode
        self.mock_interval_sec = mock_interval_sec

        # Initialize SQLite database
        init_db()
        initial_soc = load_sim_soc(default=50.0)

        # Initialize controller with persistent SoC
        self.controller = RealTimeController(initial_soc=initial_soc)

        # State tracking for jitter, watchdog, and dynamic delta t
        self._last_packet_datetime: dt.datetime | None = None
        self._last_packet_wall_time: float = time.time()
        self._watchdog_active: bool = False
        self._running: bool = True
        self._lock = threading.Lock()

        # Initialize Paho MQTT client
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        else:
            self.client = mqtt.Client()  # type: ignore

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client: Any, userdata: Any, *args: Any, **kwargs: Any) -> None:
        # VERSION2 signature: (client, userdata, flags, reason_code, properties)
        # VERSION1 signature: (client, userdata, flags, rc)
        rc = None
        if len(args) >= 2:
            rc = args[1]
        elif len(args) == 1:
            rc = args[0]
        elif "reason_code" in kwargs:
            rc = kwargs["reason_code"]
        elif "rc" in kwargs:
            rc = kwargs["rc"]
        rc_code = rc if isinstance(rc, int) else getattr(rc, "value", rc)
        if rc_code == 0:
            logger.info("Connected to MQTT broker %s:%d. Subscribing to %s...", self.broker_host, self.broker_port, self.topic_tx)
            self.client.subscribe(self.topic_tx, qos=1)
            record_system_event("INFO", "MQTT_CONNECT", f"Connected to {self.broker_host}:{self.broker_port}")
        else:
            logger.error("Failed to connect to MQTT broker, return code: %s", rc_code)
            record_system_event("ERROR", "MQTT_CONNECT_FAIL", f"Return code: {rc_code}")

    def _on_disconnect(self, client: Any, userdata: Any, *args: Any, **kwargs: Any) -> None:
        # VERSION2 signature: (client, userdata, disconnect_flags, reason_code, properties)
        # VERSION1 signature: (client, userdata, rc)
        rc = None
        if len(args) >= 2:
            rc = args[1]
        elif len(args) == 1:
            rc = args[0]
        elif "reason_code" in kwargs:
            rc = kwargs["reason_code"]
        elif "rc" in kwargs:
            rc = kwargs["rc"]
        rc_code = rc if isinstance(rc, int) else getattr(rc, "value", rc)
        logger.warning("Disconnected from MQTT broker (code %s). Automatic reconnect engaged.", rc_code)
        record_system_event("WARN", "MQTT_DISCONNECT", f"Disconnected with code {rc_code}")

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Handle incoming telemetry message on FVE/Maly_TX."""
        now_wall = time.time()
        try:
            payload_str = msg.payload.decode("utf-8")
            data = json.loads(payload_str)
        except Exception as exc:
            logger.error("Could not decode incoming JSON message: %s", exc)
            return

        self._process_telemetry_payload(data, now_wall)

    def _process_telemetry_payload(self, data: dict[str, Any], now_wall: float) -> None:
        """Process decoded telemetry dict, execute simulation step, and publish response."""
        with self._lock:
            # 1. Parse timestamp and fields
            date_str = str(data.get("Date", dt.datetime.now().strftime("%d.%m.%Y")))
            time_str = str(data.get("Time", dt.datetime.now().strftime("%H:%M:%S")))
            try:
                packet_dt = dt.datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M:%S")
            except Exception:
                packet_dt = dt.datetime.now()

            # Physical measurements
            # P_HOME in raw data is negative (-495 W = 495 W consumption)
            p_home_raw = float(data.get("P_HOME", 0.0))
            p_home_w = abs(p_home_raw)
            p_pv_w = max(0.0, float(data.get("P_PV", 0.0)))
            price_czk = float(data.get("PRICE_CZK", 3.50))
            real_soc = float(data.get("SoC", 50.0))

            # 2. Timing Jitter & Stale Packet Handling
            dt_seconds = NOMINAL_STEP_SEC
            if self._last_packet_datetime is not None:
                delta_sec = (packet_dt - self._last_packet_datetime).total_seconds()

                # Duplicate / out-of-order check
                if delta_sec <= MIN_VALID_STEP_SEC and delta_sec >= 0:
                    logger.debug("Ignoring duplicate or too rapid packet (delta %.1fs)", delta_sec)
                    return

                # Stale packet arriving from old buffer
                if delta_sec < 0:
                    logger.warning("Out-of-order packet timestamp (%s < %s), skipping.", packet_dt, self._last_packet_datetime)
                    return

                # Prolonged disconnection recovery (> 1 hour)
                if delta_sec > MAX_VALID_STEP_SEC:
                    logger.info("Prolonged outage detected (%.1f min). Re-aligning sim SoC (%.1f%% -> %.1f%%)",
                                delta_sec / 60.0, self.controller.battery_sim.soc, real_soc)
                    self.controller.sync_sim_soc_with_hardware(real_soc)
                    record_system_event("INFO", "RECOVERY_REALIGN", f"Outage {delta_sec/60:.1f}m. SoC realigned to {real_soc}%")
                    dt_seconds = NOMINAL_STEP_SEC
                else:
                    # Dynamic delta t for exact physical energy conservation
                    dt_seconds = delta_sec
            else:
                # First cold start: align simulation with actual reported hardware SoC
                logger.info("Cold start: initializing simulated SoC to match hardware (%.1f%%)", real_soc)
                self.controller.sync_sim_soc_with_hardware(real_soc)

            # Update packet timestamps and clear watchdog
            self._last_packet_datetime = packet_dt
            self._last_packet_wall_time = now_wall
            if self._watchdog_active:
                logger.info("Watchdog CLEARED: fresh telemetry received.")
                record_system_event("INFO", "WATCHDOG_CLEAR", "Telemetry restored.")
                self._watchdog_active = False

            # 3. Decision & Simulation Step
            old_sim_soc = self.controller.battery_sim.soc
            cmd_payload, step_res = self.controller.decide_tick_action(
                current_time_dt=packet_dt,
                price_czk=price_czk,
                p_home_w=p_home_w,
                p_pv_w=p_pv_w,
                real_soc=real_soc,
                dt_seconds=dt_seconds,
            )

            # 4. Publish Czech JSON command to FVE/Simulated_RX
            cmd_json = json.dumps(cmd_payload, ensure_ascii=False)
            if not self.mock_mode:
                self.client.publish(self.topic_rx, cmd_json, qos=1)

            # 5. Save to SQLite database
            save_telemetry_tick(
                timestamp=packet_dt.isoformat(),
                date_str=date_str,
                time_str=time_str,
                dt_seconds=round(dt_seconds, 1),
                price_czk=price_czk,
                p_home_w=p_home_w,
                p_pv_w=p_pv_w,
                p_grid_w=step_res.p_grid_w,
                p_bat_w=step_res.p_bat_w,
                real_soc=real_soc,
                sim_soc=step_res.new_sim_soc,
                command=step_res.effective_command,
                power_w=cmd_payload["vykon_w"],
                reason=step_res.effective_reason,
                profit_step_czk=step_res.profit_step_czk,
            )

            # 6. Concise terminal log
            profit_sign = "+" if step_res.profit_step_czk >= 0 else ""
            logger.info(
                "[%s] Spot: %.2f CZK | Home: %4.0fW | PV: %4.0fW | SoC: %4.1f%% -> %4.1f%% | Příkaz: %-21s (%4dW) | %s%.3f CZK",
                time_str,
                price_czk,
                p_home_w,
                p_pv_w,
                old_sim_soc,
                step_res.new_sim_soc,
                step_res.effective_command,
                cmd_payload["vykon_w"],
                profit_sign,
                step_res.profit_step_czk,
            )

    def _watchdog_loop(self) -> None:
        """Background thread monitoring telemetry stream health."""
        while self._running:
            time.sleep(2.0)
            now_wall = time.time()
            elapsed = now_wall - self._last_packet_wall_time

            if elapsed > WATCHDOG_TIMEOUT_SEC and not self._watchdog_active:
                with self._lock:
                    self._watchdog_active = True
                    logger.warning("[WATCHDOG ALERT] No telemetry packets received for %.1f seconds (> %ds)! Activating failsafe.",
                                   elapsed, WATCHDOG_TIMEOUT_SEC)

                    # Build safe fallback command
                    now_dt = dt.datetime.now()
                    failsafe_cmd = self.controller.build_failsafe_command(now_dt)
                    cmd_json = json.dumps(failsafe_cmd, ensure_ascii=False)

                    if not self.mock_mode:
                        try:
                            self.client.publish(self.topic_rx, cmd_json, qos=1)
                            self.client.reconnect()
                        except Exception as exc:
                            logger.error("Watchdog reconnect error: %s", exc)

                    record_system_event(
                        "WARN",
                        "WATCHDOG_FAILSAFE",
                        f"No telemetry for {elapsed:.0f}s. Sent Pokrytí spotřeby domu failsafe.",
                    )

    def run_mock_replay(self) -> None:
        """Stream sample packets for local offline testing."""
        logger.info("Starting MOCK mode: generating realistic 2-minute packets every %.1fs...", self.mock_interval_sec)
        base_time = dt.datetime.now()
        mock_ticks = 0

        # Sample price / load cycle
        while self._running:
            mock_ticks += 1
            tick_time = base_time + dt.timedelta(minutes=mock_ticks * 2)

            # Simulated physical conditions
            hour = tick_time.hour
            # Solar peak at noon
            p_pv = max(0.0, 4200.0 * (1.0 - ((hour - 12.5) / 5.0) ** 2)) if 7 <= hour <= 18 else 0.0
            # Home load with evening peak
            p_home = 350.0 + (600.0 if (17 <= hour <= 21) else 150.0)
            # Spot price pattern
            price = 6.20 if (17 <= hour <= 21) else (1.40 if (1 <= hour <= 5) else 3.80)

            sample_packet = {
                "SoC": 65,
                "P_PV": round(p_pv, 1),
                "PRICE_CZK": price,
                "P_HOME": -round(p_home, 1),
                "P_GRID": -50,
                "P_BAT": 0,
                "Date": tick_time.strftime("%d.%m.%Y"),
                "Time": tick_time.strftime("%H:%M:%S"),
            }

            self._process_telemetry_payload(sample_packet, time.time())
            time.sleep(self.mock_interval_sec)

    def start(self) -> None:
        """Start the real-time service daemon."""
        logger.info("Initializing Real-Time Service...")
        logger.info("Broker: %s:%d | Inbound: %s | Outbound: %s",
                    self.broker_host, self.broker_port, self.topic_tx, self.topic_rx)

        # Enforce SQLite storage limits on startup
        enforce_db_size_limit()

        # Start Watchdog background thread
        watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="Watchdog")
        watchdog_thread.start()

        if self.mock_mode:
            self.run_mock_replay()
        else:
            try:
                self.client.connect(self.broker_host, self.broker_port, keepalive=MQTT_KEEPALIVE_SEC)
                self.client.loop_forever()
            except KeyboardInterrupt:
                logger.info("Stopping MQTT Service...")
            except Exception as exc:
                logger.critical("Fatal MQTT Service exception: %s", exc)
            finally:
                self._running = False
                self.client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Solar Energy Optimizer - Real-Time MQTT Inverter Service")
    parser.add_argument("--host", default=MQTT_BROKER_HOST, help=f"MQTT Broker Host (default: {MQTT_BROKER_HOST})")
    parser.add_argument("--port", type=int, default=MQTT_BROKER_PORT, help=f"MQTT Port (default: {MQTT_BROKER_PORT})")
    parser.add_argument("--topic-tx", default=MQTT_TOPIC_TX, help=f"Incoming Telemetry Topic (default: {MQTT_TOPIC_TX})")
    parser.add_argument("--topic-rx", default=MQTT_TOPIC_RX, help=f"Outgoing Commands Topic (default: {MQTT_TOPIC_RX})")
    parser.add_argument("--mock", action="store_true", help="Run in local offline mock mode without connecting to MQTT")
    parser.add_argument("--mock-interval", type=float, default=2.0, help="Interval in seconds between mock ticks (default: 2.0s)")

    args = parser.parse_args()

    service = MQTTRealtimeService(
        broker_host=args.host,
        broker_port=args.port,
        topic_tx=args.topic_tx,
        topic_rx=args.topic_rx,
        mock_mode=args.mock,
        mock_interval_sec=args.mock_interval,
    )
    service.start()


if __name__ == "__main__":
    main()
