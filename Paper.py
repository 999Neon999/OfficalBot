# NUCLEAR_PAPER_TRADER_V13_SUPREME.py
# V13 Upgrade: "Nuclear Hyper-Squeeze" Edition
# Optimized for maximum profit retention using Price Velocity Stall detection

import yfinance as yf
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import time
from datetime import datetime, timedelta
import pytz
import warnings
from logger import logger

warnings.filterwarnings("ignore")

# ================== CONFIGURATION & ASSETS ==================
STOCK_META = {
    "TATASTEEL": {"Beta": 1.8, "Sector": "Metals"},
    "ADANIENT": {"Beta": 2.1, "Sector": "Energy"},
    "ADANIPORTS": {"Beta": 1.9, "Sector": "Logistics"},
    "JSWSTEEL": {"Beta": 2.0, "Sector": "Metals"},
    "HINDALCO": {"Beta": 1.7, "Sector": "Metals"},
    "VEDL": {"Beta": 2.3, "Sector": "Metals"},
    "M&M": {"Beta": 1.7, "Sector": "Auto"},
    "BPCL": {"Beta": 2.1, "Sector": "Energy"},
    "IOC": {"Beta": 2.0, "Sector": "Energy"},
    "RELIANCE": {"Beta": 1.4, "Sector": "Conglomerate"},
    "SBIN": {"Beta": 1.5, "Sector": "Banking"},
    "HDFCBANK": {"Beta": 1.3, "Sector": "Banking"},
    "ICICIBANK": {"Beta": 1.4, "Sector": "Banking"},
    "AXISBANK": {"Beta": 1.5, "Sector": "Banking"},
    "BAJFINANCE": {"Beta": 1.6, "Sector": "Finance"},
    "SHRIRAMFIN": {"Beta": 1.7, "Sector": "Finance"},
    "INDUSINDBK": {"Beta": 1.6, "Sector": "Banking"},
    "EICHERMOT": {"Beta": 1.6, "Sector": "Auto"},
    "HEROMOTOCO": {"Beta": 1.5, "Sector": "Auto"},
    "BAJAJ-AUTO": {"Beta": 1.4, "Sector": "Auto"},
    "MARUTI": {"Beta": 1.4, "Sector": "Auto"},
    "LT": {"Beta": 1.5, "Sector": "Infra"},
    "ULTRACEMCO": {"Beta": 1.5, "Sector": "Cement"},
    "GRASIM": {"Beta": 1.4, "Sector": "Diversified"},
    "TITAN": {"Beta": 1.4, "Sector": "Jewellery"},
    "TMPV": {"Beta": 1.9, "Sector": "Tech"},
    "PAYTM": {"Beta": 2.0, "Sector": "Fintech"},
    "NYKAA": {"Beta": 1.8, "Sector": "Retail"},
    "DELHIVERY": {"Beta": 1.7, "Sector": "Logistics"},
    "RVNL": {"Beta": 2.2, "Sector": "Railway"},
    "IRFC": {"Beta": 1.9, "Sector": "Railway Finance"},
    "HUDCO": {"Beta": 1.7, "Sector": "Housing Finance"},
    "POLYCAB": {"Beta": 1.6, "Sector": "Cables"},
    "DIXON": {"Beta": 1.8, "Sector": "Electronics"},
    "SAIL": {"Beta": 1.8, "Sector": "Metals"},
    "JINDALSTEL": {"Beta": 1.9, "Sector": "Metals"},
    "NMDC": {"Beta": 1.6, "Sector": "Mining"},
    "COALINDIA": {"Beta": 1.5, "Sector": "Mining"},
    "ONGC": {"Beta": 1.5, "Sector": "Energy"},
    "GAIL": {"Beta": 1.5, "Sector": "Gas"},
    "POWERGRID": {"Beta": 1.4, "Sector": "Power"},
    "NTPC": {"Beta": 1.4, "Sector": "Power"},
    "HINDPETRO": {"Beta": 1.6, "Sector": "Energy"},
    "BHARTIARTL": {"Beta": 1.4, "Sector": "Telecom"},
    "INFY": {"Beta": 1.3, "Sector": "IT"},
    "TCS": {"Beta": 1.2, "Sector": "IT"},
    "WIPRO": {"Beta": 1.3, "Sector": "IT"},
    "HCLTECH": {"Beta": 1.3, "Sector": "IT"},
    "SUNPHARMA": {"Beta": 1.4, "Sector": "Pharma"},
    "ETERNAL": {"Beta": 1.8, "Sector": "Auto"}
}

STOCKS = [f"{key}.NS" if not key.endswith('.NS') else key for key in STOCK_META.keys()]
UNIQUE_SECTORS = ['Auto', 'Banking', 'Cables', 'Cement', 'Conglomerate', 'Diversified', 'Electronics', 'Energy', 'Finance', 'Fintech', 'Gas', 'Housing Finance', 'IT', 'Infra', 'Jewellery', 'Logistics', 'Metals', 'Mining', 'Pharma', 'Power', 'Railway', 'Railway Finance', 'Retail', 'Tech', 'Telecom']

# Market Settings (India)
MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:15:00" 
CHECK_INTERVAL_SCAN = 15  # 15s Scan Frequency
CHECK_INTERVAL_HOLD = 10  # 10s Position Monitoring
TIMEZONE = "Asia/Kolkata"
TOTAL_INITIAL_CAPITAL = 11700  # User Request: ₹11,700 capital

AI_THRESHOLD = 0.70
BASE_TARGET_PCT = 0.0070   # Optimized Apex V11
STOP_LOSS_PCT = 2.0        # Optimized Apex V11
TRAILING_EFFICIENCY = 0.90 # Optimized Apex V11
MAX_HOLD_MINUTES = 180
FIXED_TP_ENABLED = False # Using Hyper-Squeeze instead

# Indian market holidays for 2025
HOLIDAYS_2025 = [
    "2025-01-26", "2025-03-14", "2025-04-10", "2025-04-14",
    "2025-05-01", "2025-08-15", "2025-10-02", "2025-10-21", "2025-11-07"
]

# Trading Charges
BROKERAGE_PCT = 0.0003
BROKERAGE_CAP = 20.0
STT_SELL_PCT = 0.00025
NSE_TXN_CHARGE_PCT = 0.0000297
SEBI_CHARGE_PCT = 0.0000001
STAMP_DUTY_BUY_PCT = 0.00003
GST_RATE = 0.18

def calculate_charges(buy_price, sell_price, quantity):
    buy_value = buy_price * quantity; sell_value = sell_price * quantity
    turnover = buy_value + sell_value
    brokerage = min(BROKERAGE_PCT * buy_value, BROKERAGE_CAP) + min(BROKERAGE_PCT * sell_value, BROKERAGE_CAP)
    txn = NSE_TXN_CHARGE_PCT * turnover; sebi = SEBI_CHARGE_PCT * turnover; gst = GST_RATE * (brokerage + txn + sebi)
    stt = STT_SELL_PCT * sell_value; stamp = STAMP_DUTY_BUY_PCT * buy_value
    return brokerage + txn + sebi + gst + stt + stamp

# ================== AI MODEL SETUP ==================
MODEL_PATH = "recovered_model.cbm"
print(f"Loading AI Model from: {MODEL_PATH}...")
try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print("OK: AI Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Model loading failed: {e}")
    exit(1)

base_features = ['cum_delta','delta','msb','range_filter_dist','ob_dist_pct',
                'vol_ratio','bars_since_msb','atr_ratio','vwap_dist_pct','ma20_dist_pct',
                'rvol','adx','sess_high_dist_pct','sess_low_dist_pct',
                'is_high_vol_bar','delta_zscore','atr_ratio_slope','rvol_slope','adx_slope']
surgical_features = ['cmf', 'bb_percent', 'macd_hist_slope']
features = base_features + ['beta'] + surgical_features + [f'sector_{sec}' for sec in UNIQUE_SECTORS]

# ================== STATE MANAGEMENT ==================
portfolio = {
    'cash': TOTAL_INITIAL_CAPITAL,
    'initial_cash': TOTAL_INITIAL_CAPITAL,
    'held_etf': None,
    'position': 0,
    'entry_price': 0,
    'stop_loss_price': 0,
    'target_price': 0,
    'trailing_active': False,
    'max_price_seen': 0,
    'entry_time': None,
    'trades': [],
    'total_pnl': 0,
    'stock_data': {ticker: pd.DataFrame() for ticker in STOCKS},
    'price_history': [] # Tracking for Velocity tracking (last 2 mins)
}

def initialize_portfolio():
    portfolio['cash'] = TOTAL_INITIAL_CAPITAL
    portfolio['initial_cash'] = TOTAL_INITIAL_CAPITAL
    portfolio['held_etf'] = None
    portfolio['position'] = 0
    portfolio['entry_price'] = 0
    portfolio['trades'] = []
    portfolio['total_pnl'] = 0
    portfolio['price_history'] = []

# ================== FEATURE ENGINEERING ==================
def add_features(df, stock_symbol):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [col.lower() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[~df.index.duplicated(keep='last')]
    
    try:
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        mf = ((c - l) - (h - c)) / (h - l + 1e-8) * v
        df['cmf'] = mf.rolling(20).sum() / v.rolling(20).sum().replace(0,1)
        ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        df['bb_percent'] = (c - (ma20 - 2*std20)) / (4*std20 + 1e-8)
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        df['macd_hist_slope'] = (macd - macd.ewm(span=9, adjust=False).mean()).diff()
        delta = v * (2*c - h - l) / (h - l + 1e-8)
        df['cum_delta'] = delta.rolling(50).sum(); df['delta'] = delta
        swing_h = h.rolling(20, center=True).max().ffill().shift(1)
        swing_l = l.rolling(20, center=True).min().ffill().shift(1)
        df['msb'] = np.where(h > swing_h, 1, np.where(l < swing_l, -1, 0))
        df['bars_since_msb'] = df['msb'].abs().cumsum()
        tr = np.abs(np.diff(c, prepend=c.iloc[0]))
        avrng = pd.Series(tr).ewm(span=100, adjust=False).mean()
        smrng = avrng.ewm(span=199, adjust=False).mean() * 3
        filt = c.copy().values; c_val = c.values; smrng_val = smrng.values
        for i in range(1, len(df)):
            if c_val[i] > filt[i-1]: filt[i] = max(filt[i-1], c_val[i] - smrng_val[i])
            else: filt[i] = min(filt[i-1], c_val[i] + smrng_val[i])
        df['range_filter_dist'] = (c - filt) / c
        df['ob_dist_pct'] = np.abs(np.where(df['msb'] == 1, c - swing_l, swing_h - c) / c)
        df['vol_ratio'] = v / v.rolling(20).mean()
        tr_s = pd.Series(tr, index=df.index); df['atr_ratio'] = tr_s.rolling(14).mean() / tr_s.rolling(50).mean().replace(0,1)
        df['vwap'] = (( (h + l + c) / 3 ) * v).rolling(50).sum() / v.rolling(50).sum().replace(0,1)
        df['vwap_dist_pct'] = (c - df['vwap']) / c
        df['ma20_dist_pct'] = (c - c.rolling(20).mean()) / c.rolling(20).mean()
        df['rvol'] = v / v.rolling(5).mean().shift(1).replace(0,1)
        plus_di = 100 * ((h - h.shift(1)).clip(lower=0) / (tr_s + 1e-8)).ewm(span=14, adjust=False).mean()
        minus_di = 100 * ((l.shift(1) - l).clip(lower=0) / (tr_s + 1e-8)).ewm(span=14, adjust=False).mean()
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8); df['adx'] = dx.ewm(span=14, adjust=False).mean()
        df['sess_high_dist_pct'] = (h.rolling(50).max() - c) / c
        df['sess_low_dist_pct'] = (c - l.rolling(50).min()) / c
        df['is_high_vol_bar'] = (v > v.rolling(9).mean() * 2).astype(int)
        df['delta_zscore'] = (delta - delta.rolling(9).mean()) / delta.rolling(9).std().replace(0,1)
        df['atr_ratio_slope'] = df['atr_ratio'].diff(); df['rvol_slope'] = df['rvol'].diff(); df['adx_slope'] = df['adx'].diff()
        meta = STOCK_META.get(stock_symbol.replace('.NS', ''), {}); df['beta'] = meta.get('Beta', 1.0); sec = meta.get('Sector', 'Other')
        for s in UNIQUE_SECTORS: df[f'sector_{s}'] = 1 if s == sec else 0
        return df.iloc[50:].dropna()
    except: return pd.DataFrame()

# ================== DATA FETCHING ==================
def fetch_historical_data():
    print("Fetching historical 1-minute data for all Stocks (past 5 days)...")
    end_time = datetime.now(pytz.timezone(TIMEZONE)); start_time = end_time - timedelta(days=5)
    try:
        data = yf.download(STOCKS, start=start_time, end=end_time, interval="1m", group_by='ticker', threads=True, progress=False, auto_adjust=False)
        count = 0
        for ticker in STOCKS:
            df_ticker = pd.DataFrame()
            if ticker in data.columns.levels[0]: df_ticker = data[ticker].copy()
            elif ticker in data: df_ticker = data[ticker].copy()
            if not df_ticker.empty:
                df_ticker = df_ticker[~df_ticker.index.duplicated(keep='last')].sort_index()
                processed_df = add_features(df_ticker, ticker)
                if not processed_df.empty:
                    portfolio['stock_data'][ticker] = processed_df
                    count += 1
        print(f"Initial data loaded and processed for {count}/{len(STOCKS)} stocks.")
    except Exception as e:
        print(f"Error during historical data fetch: {e}")

def fetch_latest_data(tickers):
    if not tickers: return pd.DataFrame()
    end_time = datetime.now(pytz.timezone(TIMEZONE)); start_time = end_time - timedelta(minutes=5)
    try:
        return yf.download(tickers, start=start_time, end=end_time, interval="1m", group_by='ticker', threads=True, progress=False, auto_adjust=False)
    except: return pd.DataFrame()

def update_data(new_data, tickers):
    updated_count = 0
    for ticker in tickers:
        df_new = pd.DataFrame()
        if len(tickers) == 1 and isinstance(new_data.columns, pd.Index) and 'Close' in new_data.columns: df_new = new_data
        elif ticker in new_data.columns.levels[0]: df_new = new_data[ticker]
        elif ticker in new_data: df_new = new_data[ticker]
        if df_new.empty: continue
        current_df = portfolio['stock_data'].get(ticker, pd.DataFrame())
        if not current_df.empty:
            last_idx = current_df.index[-1]
            if df_new.index.tz is None and last_idx.tz is not None: df_new.index = df_new.index.tz_localize(last_idx.tz)
            new_rows = df_new[df_new.index > last_idx]
            if new_rows.empty: continue
            history_subset = current_df.iloc[-400:].copy()
            if isinstance(new_rows.columns, pd.MultiIndex): new_rows.columns = new_rows.columns.get_level_values(0)
            new_rows.columns = [c.lower() for c in new_rows.columns]
            combined = pd.concat([history_subset, new_rows]); combined = combined[~combined.index.duplicated(keep='last')]
            processed_chunk = add_features(combined, ticker)
            if not processed_chunk.empty:
                portfolio['stock_data'][ticker] = pd.concat([current_df.iloc[:-len(history_subset)], processed_chunk]).tail(2000)
                updated_count += 1
        else:
            processed_df = add_features(df_new, ticker)
            if not processed_df.empty:
                portfolio['stock_data'][ticker] = processed_df
                updated_count += 1
    return updated_count

# ================== TRADING LOGIC ==================
def is_trading_day():
    now = datetime.now(pytz.timezone(TIMEZONE))
    return now.weekday() < 5 and now.strftime("%Y-%m-%d") not in HOLIDAYS_2025

def is_market_open():
    now = datetime.now(pytz.timezone(TIMEZONE)).time()
    return is_trading_day() and datetime.strptime(MARKET_OPEN, "%H:%M:%S").time() <= now <= datetime.strptime(MARKET_CLOSE, "%H:%M:%S").time()

def is_before_market_open():
    now = datetime.now(pytz.timezone(TIMEZONE)).time()
    return is_trading_day() and now < datetime.strptime(MARKET_OPEN, "%H:%M:%S").time()

def get_realtime_price(ticker):
    try:
        p = yf.Ticker(ticker).fast_info['last_price']
        if p and p > 0: return p
    except: pass
    return None

def get_ai_prediction(df):
    try:
        if len(df) < 50: return 0.0
        latest_data = df.iloc[-1:].copy()
        for f in features:
            if f not in latest_data.columns: latest_data[f] = 0.0
        # Passing DataFrame directly to ensure feature name mapping
        probs = model.predict_proba(latest_data[features])
        return float(probs[0][1])
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0

def scan_for_buy():
    candidates = []
    for ticker in STOCKS:
        df = portfolio['stock_data'].get(ticker, pd.DataFrame())
        if df.empty: continue
        prob = get_ai_prediction(df)
        if prob >= AI_THRESHOLD: candidates.append((ticker, prob))
    if not candidates: return None, 0.0
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]

def execute_buy(ticker, prob):
    df = portfolio['stock_data'][ticker]; price = df['close'].iloc[-1]
    clean_sym = ticker.replace('.NS', ''); beta = STOCK_META.get(clean_sym, {}).get('Beta', 1.0)
    cash_alloc = portfolio['cash'] * 0.98; shares = int(cash_alloc // price)
    if shares == 0: return False
    cost = shares * price; portfolio['cash'] -= cost; portfolio['position'] = shares; portfolio['entry_price'] = price
    portfolio['held_etf'] = ticker; portfolio['entry_time'] = datetime.now(pytz.timezone(TIMEZONE)); portfolio['trailing_active'] = False; portfolio['max_price_seen'] = 0.0
    target_pct = BASE_TARGET_PCT * beta; portfolio['target_price'] = price * (1 + target_pct); portfolio['stop_loss_price'] = price * (1 - STOP_LOSS_PCT / 100.0)
    timestamp = datetime.now(pytz.timezone(TIMEZONE)); portfolio['trades'].append(('buy', timestamp, ticker, price, shares, 0, prob))
    logger.log_trade("buy", ticker, price, shares, reason=f"Confidence: {prob:.1%}")
    print(f"\n{'='*60}\n🚀 [V13] SNIPER ENTRY: {ticker}\n{'='*60}\nTime: {timestamp.strftime('%H:%M:%S')}\nPrice: ₹{price:.2f}\nQty: {shares}\nConf: {prob:.1%}\nTarget: ₹{portfolio['target_price']:.2f} (+{target_pct*100:.2f}%)\n{'='*60}\n")
    return True

def execute_sell(ticker, price, reason):
    if portfolio['position'] == 0: return False
    qty = portfolio['position']; entry_p = portfolio['entry_price']; charges = calculate_charges(entry_p, price, qty); net_pnl = (price - entry_p) * qty - charges
    portfolio['cash'] += (qty * price) - charges; portfolio['total_pnl'] += net_pnl; timestamp = datetime.now(pytz.timezone(TIMEZONE))
    portfolio['trades'].append(('sell', timestamp, ticker, price, qty, net_pnl, reason))
    logger.log_trade("sell", ticker, price, qty, pnl=net_pnl, reason=reason)
    hold_m = (timestamp - portfolio['entry_time']).total_seconds() / 60
    print(f"\n{'='*60}\n🎯 [V13] POSITION CLOSED: {ticker} ({reason})\n{'='*60}\nPrice: ₹{price:.2f}\nP&L: ₹{net_pnl:,.2f}\nBalance: ₹{portfolio['cash']:,.2f}\n{'='*60}\n")
    portfolio['position'] = 0; portfolio['held_etf'] = None; portfolio['entry_price'] = 0; portfolio['trailing_active'] = False; portfolio['max_price_seen'] = 0.0
    portfolio['price_history'] = []
    return True

def process_hold():
    if not portfolio['held_etf']: return
    ticker = portfolio['held_etf']; rt_p = get_realtime_price(ticker); current_price = rt_p if rt_p else portfolio['stock_data'][ticker]['close'].iloc[-1]
    
    if not portfolio['trailing_active'] and current_price >= portfolio['target_price']:
        if FIXED_TP_ENABLED:
            execute_sell(ticker, current_price, "Fixed Take Profit")
            return
        portfolio['trailing_active'] = True; portfolio['max_price_seen'] = current_price
        print(f"🔥 V13 SUPREME PROFIT LOCK ACTIVATED: {ticker} > ₹{portfolio['target_price']:.2f}")
    
    if portfolio['trailing_active']:
        if current_price > portfolio['max_price_seen']: portfolio['max_price_seen'] = current_price
        
        # Track Price History for Velocity (Last 12 samples of 10s = 2 mins)
        portfolio['price_history'].append(current_price)
        if len(portfolio['price_history']) > 12: portfolio['price_history'].pop(0)
        
        # Calculate Velocity (Speed of growth over last 1 min / 6 samples)
        velocity = 0.0
        if len(portfolio['price_history']) >= 6:
            old_p = portfolio['price_history'][-6]
            velocity = (current_price - old_p) / old_p
            
        # V13 Optimized Profit Lock
        profit_pct = (current_price - portfolio['entry_price']) / portfolio['entry_price']
        
        # Base Trail Width
        if profit_pct >= 0.05:
            trail_width = 0.0015 # 0.15% room
        elif profit_pct >= 0.03:
            trail_width = 0.003  # 0.3% room
        elif profit_pct >= 0.015:
            trail_width = 0.006  # 0.6% room
        else:
            trail_width = 0.012  # 1.2% base room
            
        # HYPER-SQUEEZE: If momentum stalls (velocity < 0.08% per min) while in profit
        if profit_pct > 0.008 and velocity < 0.0008:
            trail_width = 0.0005 # Tighten to 0.05% room immediately
            reason_suffix = " (Hyper-Squeeze Stall)"
        else:
            reason_suffix = f" ({trail_width*100:.2f}%)"
            
        trail_exit = portfolio['max_price_seen'] * (1 - trail_width)
        if current_price <= trail_exit:
            execute_sell(ticker, current_price, f"V13 Profit Lock{reason_suffix}")
            return

    if not portfolio['trailing_active'] and current_price <= portfolio['stop_loss_price']:
        execute_sell(ticker, current_price, "Stop Loss")
        return
        
    hold_m = (datetime.now(pytz.timezone(TIMEZONE)) - portfolio['entry_time']).total_seconds() / 60
    if not portfolio['trailing_active'] and hold_m >= 30 and current_price < portfolio['entry_price']:
        execute_sell(ticker, current_price, "Velocity Exit (30m)")
        return
    if hold_m >= MAX_HOLD_MINUTES:
         execute_sell(ticker, current_price, "Time Exit (300m)")
         return

def print_portfolio_summary():
    curr_t = datetime.now(pytz.timezone(TIMEZONE)); ticker = portfolio['held_etf']
    status = "Scanning..."; val = portfolio['cash']
    if ticker:
        p = get_realtime_price(ticker); p = p if p else portfolio['stock_data'][ticker]['close'].iloc[-1]
        pnl = (p - portfolio['entry_price']) * portfolio['position']; val += (portfolio['position'] * p)
        status = f"HOLD: {ticker} | Price: ₹{p:.2f} | P&L: ₹{pnl:,.2f}"
    print(f"[{curr_t.strftime('%H:%M:%S')}] Status: {status} | Total: ₹{val:,.0f} | Trades: {len([t for t in portfolio['trades'] if t[0]=='sell'])}")

def main():
    initialize_portfolio()
    print("NUCLEAR PAPER TRADER V13 SUPREME STARTED")
    print(f"Targeting: {len(STOCKS)} Stocks")
    print(f"Initial Capital: Rs. {TOTAL_INITIAL_CAPITAL}")
    while True:
        if not is_trading_day(): break
        if is_before_market_open(): time.sleep(60); continue
        if not is_market_open(): print_portfolio_summary(); break
        if not any(not df.empty for df in portfolio['stock_data'].values()): fetch_historical_data()
        
        last_m = None; mode = 'scan' if portfolio['held_etf'] is None else 'hold'
        while is_market_open():
            curr_m = datetime.now(pytz.timezone(TIMEZONE)).replace(second=0, microsecond=0)
            if last_m is None or curr_m > last_m:
                if mode == 'scan':
                    print(f"\nScanning Minute: {curr_m.strftime('%H:%M')}")
                    if update_data(fetch_latest_data(STOCKS), STOCKS) > 0:
                        t, p = scan_for_buy()
                        if t: execute_buy(t, p); mode = 'hold'
                else:
                    print(f"\nMonitoring: {curr_m.strftime('%H:%M')} ({portfolio['held_etf']})")
                    update_data(fetch_latest_data([portfolio['held_etf']]), [portfolio['held_etf']]); process_hold()
                    if portfolio['held_etf'] is None: mode = 'scan'
                last_m = curr_m; print_portfolio_summary()
            time.sleep(CHECK_INTERVAL_HOLD if mode == 'hold' else CHECK_INTERVAL_SCAN)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nTerminated."); print_portfolio_summary()
