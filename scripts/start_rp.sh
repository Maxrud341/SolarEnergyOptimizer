#!/bin/bash
# ==============================================================================
# Start Real-Time FVE MQTT Service and Streamlit Visualizer in background
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "real_time/.venv/bin/activate" ]; then
    source real_time/.venv/bin/activate
fi

mkdir -p real_time/data

echo "==> Starting MQTT Inverter Service..."
nohup python3 -m real_time.mqtt_service > real_time/mqtt.log 2>&1 &
MQTT_PID=$!
echo "    MQTT Service started with PID: $MQTT_PID (Log: real_time/mqtt.log)"

echo "==> Starting Streamlit Real-Time Dashboard..."
nohup streamlit run real_time/visualizer_realtime.py --server.port 8501 --server.address 0.0.0.0 > real_time/visualizer.log 2>&1 &
VIS_PID=$!
echo "    Visualizer started with PID: $VIS_PID (Log: real_time/visualizer.log)"

TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "100.92.72.6")

echo ""
echo "✅ Both services are running in background!"
echo "   📊 Web Visualizer: http://$TAILSCALE_IP:8501"
echo "   📝 View MQTT log:       tail -f real_time/mqtt.log"
echo "   📝 View Visualizer log: tail -f real_time/visualizer.log"
echo "   🛑 To stop everything:  ./scripts/stop_rp.sh"
