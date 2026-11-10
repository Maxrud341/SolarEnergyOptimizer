#!/bin/bash
# ==============================================================================
# Calculate and display tomorrow's spot prices, solar forecast & arbitrage plan
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "real_time/.venv/bin/activate" ]; then
    source real_time/.venv/bin/activate
fi

python3 scripts/plan_tomorrow.py "$@"
