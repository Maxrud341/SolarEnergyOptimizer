#!/bin/bash
# ==============================================================================
# Start Real-Time FVE Services (Self-contained in real_time folder)
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

mkdir -p data

# Keep log files under 5 MB (truncate if oversized on restart)
for logfile in mqtt.log visualizer.log; do
    if [ -f "$logfile" ] && [ $(stat -c%s "$logfile" 2>/dev/null || echo 0) -gt 5242880 ]; then
        tail -n 3000 "$logfile" > "${logfile}.tmp" && mv "${logfile}.tmp" "$logfile"
    fi
done

echo "==> Starting MQTT Inverter Service..."
nohup python3 mqtt_service.py > mqtt.log 2>&1 &
MQTT_PID=$!
echo "    MQTT Service started with PID: $MQTT_PID (Log: mqtt.log)"

echo "==> Starting Streamlit Real-Time Dashboard..."
nohup streamlit run visualizer_realtime.py --server.port 8501 --server.address 0.0.0.0 > visualizer.log 2>&1 &
VIS_PID=$!
echo "    Visualizer started with PID: $VIS_PID (Log: visualizer.log)"

TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "100.92.72.6")

echo ""
echo "✅ Both services are running in background!"
echo "   📊 Web Visualizer: http://$TAILSCALE_IP:8501"
echo "   📝 View MQTT log:       tail -f mqtt.log"
echo "   📝 View Visualizer log: tail -f visualizer.log"
echo "   🛑 To stop everything:  ./stop.sh"
