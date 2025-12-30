#!/bin/bash

# Navigate to the bot directory
cd ~/TradingBot

echo "Starting Trading System..."

# 1. Start Support Services
echo "Launching Dashboard and Tunnel..."
python3 dashboard_server.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

if command -v cloudflared &> /dev/null && [ -f ~/.cloudflared/config.yml ]; then
    cloudflared tunnel run trading-bot > cf_tunnel.log 2>&1 &
    CF_PID=$!
fi

echo "Scheduler active. Waiting for 07:00 AM for login..."

while true; do
    CURRENT_TIME=$(date +%H:%M)
    DAY=$(date +%u) # 1-7 (Monday-Sunday)

    # Only run on weekdays (Mon-Fri)
    if [ "$DAY" -le 5 ]; then
        
        # 1. Morning Login (7:00 AM)
        if [ "$CURRENT_TIME" == "07:00" ]; then
            echo "07:00 AM: Running kite_login.py..."
            # Note: Ensure Selenium is setup for headless mode on Pi
            python3 kite_login.py > login.log 2>&1
            sleep 65 # Prevent double execution
        fi

        # 2. Market Start & Watchdog (09:15 AM - 03:30 PM)
        if [[ "$CURRENT_TIME" > "09:14" && "$CURRENT_TIME" < "15:30" ]]; then
            if ! pgrep -f "Live_Trader_Kite.py" > /dev/null; then
                echo "$CURRENT_TIME: [WATCHDOG] Live Trader not running. Starting/Restarting..."
                python3 Live_Trader_Kite.py >> trader.log 2>&1 &
                TRADER_PID=$!
            fi
        fi

        # 3. Market Close Cleanup (03:31 PM)
        if [ "$CURRENT_TIME" == "15:31" ]; then
            if pgrep -f "Live_Trader_Kite.py" > /dev/null; then
                echo "03:31 PM: Markets Closed. Stopping Trader."
                pkill -f "Live_Trader_Kite.py"
            fi
        fi
    fi

    sleep 30
done

# Cleanup on exit
trap "kill $DASHBOARD_PID $CF_PID; pkill -f Live_Trader_Kite.py; exit" SIGINT SIGTERM
