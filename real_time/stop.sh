#!/bin/bash
# ==============================================================================
# Stop Real-Time FVE Services (Self-contained in real_time folder)
# ==============================================================================

echo "==> Stopping MQTT Service and Streamlit Dashboard..."
pkill -f mqtt_service
pkill -f visualizer_realtime
pkill -f streamlit

sleep 1

REMAINING=$(pgrep -f "mqtt_service|visualizer_realtime|streamlit")

if [ -z "$REMAINING" ]; then
    echo "✅ All services successfully stopped."
else
    echo "⚠️ Warning: Some processes still running (PIDs: $REMAINING). Force killing..."
    pkill -9 -f mqtt_service
    pkill -9 -f visualizer_realtime
    pkill -9 -f streamlit
    echo "✅ Forced stop completed."
fi
