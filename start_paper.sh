#!/bin/bash

# Navigate to the bot directory
cd ~/TradingBot

echo "Starting Paper Trading System..."

# 1. Start Support Services
echo "Launching Dashboard..."
python3 dashboard_server.py > dashboard_paper.log 2>&1 &
DASHBOARD_PID=$!

if command -v cloudflared &> /dev/null && [ -f ~/.cloudflared/config.yml ]; then
    echo "Launching Cloudflare Tunnel..."
    cloudflared tunnel run trading-bot > cf_tunnel_paper.log 2>&1 &
    CF_PID=$!
fi

echo "Paper Scheduler active. Waiting for 09:15 AM for market open..."

while true; do
    CURRENT_TIME=$(date +%H:%M)
    DAY=$(date +%u) # 1-7 (Monday-Sunday)

    # Only run on weekdays (Mon-Fri)
    if [ "$DAY" -le 5 ]; then
        
        # 1. Market Start & Watchdog (09:15 AM - 03:30 PM)
        if [[ "$CURRENT_TIME" > "09:14" && "$CURRENT_TIME" < "15:30" ]]; then
            if ! pgrep -f "Paper.py" > /dev/null; then
                echo "$CURRENT_TIME: [WATCHDOG] Paper Trader not running. Starting/Restarting..."
                python3 Paper.py >> paper.log 2>&1 &
                TRADER_PID=$!
            fi
        fi

        # 2. Market Close Cleanup (03:31 PM)
        if [ "$CURRENT_TIME" == "15:31" ]; then
            if pgrep -f "Paper.py" > /dev/null; then
                echo "03:31 PM: Markets Closed. Stopping Paper Trader."
                pkill -f "Paper.py"
            fi
        fi
    fi

    sleep 30
done

# Cleanup on exit
trap "kill $DASHBOARD_PID $CF_PID; pkill -f Paper.py; exit" SIGINT SIGTERM
