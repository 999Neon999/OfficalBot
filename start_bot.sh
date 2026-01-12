#!/bin/bash

# ==========================================
# ☢️ V17 ALPHA HUNTER - DIETPI PRODUCTION
# ==========================================

# Navigate to the bot directory
cd /TradingBot

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 1. Start Support Services
echo "Launching V17 Glass Dashboard (Persistent)..."
# Check if already running to avoid duplicates
if ! pgrep -f "dashboard_server.py" > /dev/null; then
    python3 dashboard_server.py > dashboard.log 2>&1 &
    DASHBOARD_PID=$!
fi

# 2. Initial Login (Run as soon as started)
echo "Initial Kite Session Check..."
python3 kite_login.py

while true; do
    CURRENT_TIME=$(date +%H:%M)
    
    # 1. Daily Session Refresh (09:10 AM)
    if [ "$CURRENT_TIME" == "09:10" ]; then
        echo "09:10 AM: Refreshing Daily Session..."
        python3 kite_login.py
        sleep 60
    fi

    # 2. Trading Window (09:14 AM - 03:15 PM)
    if [[ "$CURRENT_TIME" > "09:13" && "$CURRENT_TIME" < "15:15" ]]; then
        if ! pgrep -f "Live_Trader_Kite.py" > /dev/null; then
            echo "$CURRENT_TIME: [WATCHDOG] V17 Live Bot not running. Starting..."
            python3 Live_Trader_Kite.py >> live_bot.log 2>&1 &
        fi
    fi

    # 3. Safety Shutdown (03:15 PM / 15:15)
    # The bot exits itself at 15:13, this is a safety kill
    if [ "$CURRENT_TIME" == "15:15" ]; then
        if pgrep -f "Live_Trader_Kite.py" > /dev/null; then
            echo "15:15: Safety Stop. Killing remaining V17 processes."
            pkill -f "Live_Trader_Kite.py"
        fi
    fi

    # 4. Service Watchdog (Ensure Dashboard is always up)
    if ! pgrep -f "dashboard_server.py" > /dev/null; then
        echo "Dashboard down. Restarting..."
        python3 dashboard_server.py > dashboard.log 2>&1 &
    fi

    sleep 30
done

# Cleanup on exit
trap "pkill -f dashboard_server.py; pkill -f Live_Trader_Kite.py; exit" SIGINT SIGTERM
