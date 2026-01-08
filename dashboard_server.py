from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import pytz

app = Flask(__name__, template_folder='.') # Look for index.html in root
CORS(app)

LOG_DIR = "" # Use root directory
TIMEZONE = "Asia/Kolkata"

def get_current_date():
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).strftime("%Y-%m-%d")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def get_stats():
    summary_file = os.path.join(LOG_DIR, "summary.json")
    daily_file = os.path.join(LOG_DIR, f"{get_current_date()}.json")
    
    
    stats = {
        "total_profit": 0.0,
        "daily_profit": 0.0,
        "total_win_rate": 0.0,
        "daily_win_rate": 0.0,
        "roi_pct": 0.0,
        "avg_duration": 0,
        "total_trades": 0,
        "daily_trades": 0,
        "pnl_curve": [],
        "trades": [],
        "active_positions": {}
    }
    
    # 1. Parse CSV for Master Stats
    csv_file = "v17_trade_log.csv"
    if os.path.exists(csv_file):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            df['time'] = pd.to_datetime(df['time'])
            
            # Filter for SELLs to calculate PnL metrics
            sells = df[df['side'] == 'SELL'].copy()
            if not sells.empty:
                # --- TOTAL STATS ---
                stats["total_profit"] = float(sells['pnl'].sum())
                stats["total_trades"] = len(sells)
                total_wins = len(sells[sells['pnl'] > 0])
                stats["total_win_rate"] = (total_wins / len(sells)) * 100
                stats["roi_pct"] = (stats["total_profit"] / 11700.0) * 100 
                
                # --- DAILY STATS ---
                today = get_current_date()
                daily_sells = sells[sells['time'].dt.strftime('%Y-%m-%d') == today]
                stats["daily_profit"] = float(daily_sells['pnl'].sum())
                stats["daily_trades"] = len(daily_sells)
                if len(daily_sells) > 0:
                    daily_wins = len(daily_sells[daily_sells['pnl'] > 0])
                    stats["daily_win_rate"] = (daily_wins / len(daily_sells)) * 100
                else:
                    stats["daily_win_rate"] = 0.0
                
                # PnL Curve (Total)
                sells.sort_values('time', inplace=True)
                sells['cum_pnl'] = sells['pnl'].cumsum()
                stats["pnl_curve"] = sells[['time', 'cum_pnl']].apply(lambda x: [x['time'].isoformat(), x['cum_pnl']], axis=1).tolist()
                
                # Avg Duration (Approximate)
                buys = df[df['side'] == 'BUY']
                durations = []
                for idx, sell in sells.iterrows():
                    match = buys[(buys['ticker'] == sell['ticker']) & (buys['time'] < sell['time'])].sort_values('time').iloc[-1:]
                    if not match.empty:
                        delta = sell['time'] - match.iloc[0]['time']
                        durations.append(delta.total_seconds() / 60)
                
                if durations:
                    stats["avg_duration"] = int(sum(durations) / len(durations))

        except Exception as e:
            print(f"Stats error: {e}")
            
    # 2. Get Live Positions from Daily JSON (fallback/supplement)
    if os.path.exists(daily_file):
        try:
            with open(daily_file, "r") as f:
                daily = json.load(f)
                stats["active_positions"] = daily.get("active_positions", {})
        except: pass
            
    return jsonify(stats)

@app.route('/api/settings')
def get_settings():
    settings = {}
    try:
        with open("Live_Trader_Kite.py", "r") as f:
            for line in f:
                if "=" in line:
                    parts = line.split("=")
                    key = parts[0].strip()
                    val = parts[1].strip().split("#")[0].strip() 
                    
                    if key in ["AI_THRESHOLD", "BASE_TARGET_PCT", "STOP_LOSS_PCT", "MAX_HOLD_BARS", "MAX_CONCURRENT_TRADES", "USE_BETA", "MAX_DAILY_LOSS", "INITIAL_CAPITAL"]:
                         try:
                             settings[key] = eval(val) 
                         except:
                             settings[key] = val
    except Exception as e:
        print(f"Settings error: {e}")
        settings = {"error": str(e)}
        
    return jsonify(settings)

@app.route("/api/logs")
def get_logs():
    csv_file = "v17_trade_log.csv"
    if not os.path.exists(csv_file):
        return jsonify([])
    
    import pandas as pd
    try:
        df = pd.read_csv(csv_file)
        # Convert timestamp to string if needed and return as list of dicts
        return jsonify(df.tail(100).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
