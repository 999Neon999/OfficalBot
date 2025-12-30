import json
import os
from datetime import datetime
import pytz

TIMEZONE = "Asia/Kolkata"
LOG_DIR = ""  # Use root directory

class TradingLogger:
    def __init__(self):
        self.tz = pytz.timezone(TIMEZONE)
        self.summary_file = os.path.join(LOG_DIR, "summary.json")
        self.load_summary()

    def get_date_str(self):
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def get_log_file(self):
        return os.path.join(LOG_DIR, f"{self.get_date_str()}.json")

    def load_summary(self):
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, "r") as f:
                    self.summary = json.load(f)
            except:
                self.summary = {"total_profit": 0.0, "total_trades": 0}
        else:
            self.summary = {"total_profit": 0.0, "total_trades": 0}

    def save_summary(self):
        with open(self.summary_file, "w") as f:
            json.dump(self.summary, f, indent=4)

    def load_daily_log(self):
        file_path = self.get_log_file()
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"date": self.get_date_str(), "daily_profit": 0.0, "trades": [], "active_positions": {}}

    def save_daily_log(self, data):
        with open(self.get_log_file(), "w") as f:
            json.dump(data, f, indent=4)

    def log_trade(self, trade_type, ticker, price, quantity, pnl=0.0, reason=""):
        data = self.load_daily_log()
        timestamp = datetime.now(self.tz).strftime("%H:%M:%S")
        
        trade_entry = {
            "timestamp": timestamp,
            "type": trade_type,
            "ticker": ticker,
            "price": price,
            "quantity": quantity,
            "pnl": pnl,
            "reason": reason
        }
        
        data["trades"].append(trade_entry)
        
        if trade_type == "sell":
            data["daily_profit"] += pnl
            self.summary["total_profit"] += pnl
            self.summary["total_trades"] += 1
            self.save_summary()
            
            if ticker in data["active_positions"]:
                del data["active_positions"][ticker]
        else:
            data["active_positions"][ticker] = {
                "entry_price": price,
                "quantity": quantity,
                "entry_time": timestamp
            }
            
        self.save_daily_log(data)

    def update_active_position(self, ticker, current_price):
        """Optional: Update Unrealized P&L in logs if needed"""
        data = self.load_daily_log()
        if ticker in data["active_positions"]:
            data["active_positions"][ticker]["current_price"] = current_price
            data["active_positions"][ticker]["unrealized_pnl"] = (current_price - data["active_positions"][ticker]["entry_price"]) * data["active_positions"][ticker]["quantity"]
            self.save_daily_log(data)

# Global Instance
logger = TradingLogger()
