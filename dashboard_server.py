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
        "trades": [],
        "active_positions": {}
    }
    
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary = json.load(f)
            stats["total_profit"] = summary.get("total_profit", 0.0)
            
    if os.path.exists(daily_file):
        with open(daily_file, "r") as f:
            daily = json.load(f)
            stats["daily_profit"] = daily.get("daily_profit", 0.0)
            stats["trades"] = daily.get("trades", [])
            stats["active_positions"] = daily.get("active_positions", {})
            
    return jsonify(stats)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
