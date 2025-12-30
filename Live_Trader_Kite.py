import os
import time
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import pytz
from kiteconnect import KiteConnect
import warnings
from logger import logger

warnings.filterwarnings("ignore")

# ================== CONFIGURATION ==================
API_KEY = "vv25p1x1xjh0gpnr"
TOKEN_FILE = "access_token.txt"
MODEL_PATH = "recovered_model.cbm"
AI_THRESHOLD = 0.75

# Strategy Parameters (V13 Supreme Apex)
BASE_TARGET_PCT = 0.0070
STOP_LOSS_PCT = 2.0        # 2% fixed SL
TRAILING_EFFICIENCY = 0.90 
SQUEEZE_THRESH = 0.00080    # Momentum stall trigger
SQUEEZE_WIDTH_PCT = 0.0005  # Tight 0.05% trail on stall

TIMEZONE = "Asia/Kolkata"
CHECK_INTERVAL_LIVE = 15    # Seconds between scans

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
    "SUNPHARMA": {"Beta": 1.4, "Sector": "Pharma"}
}

UNIQUE_SECTORS = sorted(list(set([v.get('Sector', 'Other') for v in STOCK_META.values()])))

# ================== INITIALIZATION ==================

if not os.path.exists(TOKEN_FILE):
    print(f"FATAL: {TOKEN_FILE} not found. Run kite_login.py first.")
    exit(1)

with open(TOKEN_FILE, "r") as f:
    ACCESS_TOKEN = f.read().strip()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

try:
    print("Connecting to Kite...")
    profile = kite.profile()
    print(f"OK: Logged in as {profile['user_name']} ({profile['user_id']})")
except Exception as e:
    print(f"FATAL: Kite Login Failed: {e}")
    exit(1)

print(f"Loading AI Model: {MODEL_PATH}...")
try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print("OK: Model Loaded successfully.")
except Exception as e:
    print(f"FATAL: Model loading failed: {e}")
    exit(1)

# Features List (Must match Training)
base_features = ['cum_delta','delta','msb','range_filter_dist','ob_dist_pct',
                'vol_ratio','bars_since_msb','atr_ratio','vwap_dist_pct','ma20_dist_pct',
                'rvol','adx','sess_high_dist_pct','sess_low_dist_pct',
                'is_high_vol_bar','delta_zscore','atr_ratio_slope','rvol_slope','adx_slope']
surgical_features = ['cmf', 'bb_percent', 'macd_hist_slope']
features = base_features + ['beta'] + surgical_features + [f'sector_{sec}' for sec in UNIQUE_SECTORS]

# ================== ROBUST API HELPERS ==================

def safe_kite_call(func, *args, **kwargs):
    """Wrapper for general Kite API calls with simple retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1: raise e
            print(f"API Call Warning: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(1)

def place_order_with_retry(side, ticker, quantity, price=None, order_type=KiteConnect.ORDER_TYPE_MARKET):
    """Robust order placement with exponential backoff."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Placing {side} order for {ticker} (Attempt {attempt+1})...")
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=ticker,
                transaction_type=side,
                quantity=quantity,
                product=kite.PRODUCT_MIS,
                order_type=order_type,
                price=price
            )
            print(f"SUCCESS: Order ID {order_id}")
            return order_id
        except Exception as e:
            err_str = str(e).lower()
            # Don't retry if it's a fundamental account/input error
            if "insufficient" in err_str or "invalid" in err_str or "margin" in err_str:
                print(f"PERMANENT ERROR: {e}")
                return None
            
            if attempt == max_retries - 1:
                print(f"FATAL: Order failed after {max_retries} attempts: {e}")
                return None
            
            print(f"RETRYABLE ERROR: {e}. Waiting {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2 # Exponential backoff

# ================== CORE LOGIC ==================

class LivePosition:
    def __init__(self, ticker, entry_price, quantity, initial_stop, initial_target, beta):
        self.ticker = ticker
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = initial_stop
        self.target_price = initial_target
        self.beta = beta
        self.status = "OPEN"
        self.peak_price = entry_price
        self.is_squeezed = False
        self.p_6_ref = None # Price after 6 mins of entry for velocity check
        self.entry_time = datetime.now(pytz.timezone(TIMEZONE))

def add_features(df, stock_symbol):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    try:
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        # Feature engineering logic (same as Paper.py)
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
        avrng = pd.Series(tr, index=df.index).ewm(span=100, adjust=False).mean()
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
        
        meta = STOCK_META.get(stock_symbol.replace('.NS', ''), {})
        df['beta'] = meta.get('Beta', 1.0)
        sec = meta.get('Sector', 'Other')
        for s in UNIQUE_SECTORS: df[f'sector_{s}'] = 1 if s == sec else 0
        return df
    except: return pd.DataFrame()

def get_ai_prediction(df):
    try:
        if len(df) < 50: return 0.0
        latest_data = df.iloc[-1:].copy()
        for f in features:
            if f not in latest_data.columns: latest_data[f] = 0.0
        X = latest_data[features].values
        probs = model.predict_proba(X)
        return float(probs[0][1])
    except: return 0.0

active_pos = None

def scan_and_trade():
    global active_pos
    import yfinance as yf # Keep scanner simple using yfinance for data, but Kite for orders
    
    if active_pos:
        # Manage Exit logic
        try:
            ticker = active_pos.ticker + ".NS"
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if df.empty: return
            
            curr_p = df['Close'].iloc[-1]
            active_pos.peak_price = max(active_pos.peak_price, curr_p)
            elapsed = (datetime.now(pytz.timezone(TIMEZONE)) - active_pos.entry_time).total_seconds() / 60.0
            
            # Record price at 6 mins for velocity check
            if 5.5 < elapsed < 7.0 and active_pos.p_6_ref is None:
                active_pos.p_6_ref = curr_p
                print(f"Recorded 6-min Reference: ₹{curr_p}")

            exit_reason = None
            if curr_p <= active_pos.stop_loss: exit_reason = "STOP LOSS"
            elif curr_p >= active_pos.target_price:
                # Target Hit -> Squeeze Logic
                if active_pos.p_6_ref:
                    velocity = (active_pos.p_6_ref - active_pos.entry_price) / active_pos.entry_price
                    if velocity < SQUEEZE_THRESH:
                        # Stall detected
                        new_sl = active_pos.peak_price * (1 - SQUEEZE_WIDTH_PCT)
                        if not active_pos.is_squeezed:
                            print(f"Momentum Stall! Squeezing SL to ₹{new_sl:.2f}")
                            active_pos.is_squeezed = True
                        if curr_p <= new_sl: exit_reason = "SQUEEZE EXIT"
                    else:
                        # Healthy Trend
                        trail_sl = active_pos.peak_price * (1 - 0.003)
                        if curr_p <= trail_sl: exit_reason = "TRAILING STOP"
                else:
                    # Before 6 mins, use standard target profit
                    exit_reason = "TARGET HIT"

            if exit_reason:
                print(f"EXECUTING {exit_reason} for {active_pos.ticker} at ₹{curr_p}")
                order_id = place_order_with_retry(
                    side=kite.TRANSACTION_TYPE_SELL,
                    ticker=active_pos.ticker,
                    quantity=active_pos.quantity
                )
                if order_id:
                    # Log Sell
                    entry_p = active_pos.entry_price
                    qty = active_pos.quantity
                    pnl = (curr_p - entry_p) * qty
                    logger.log_trade("sell", active_pos.ticker, curr_p, qty, pnl=pnl, reason=exit_reason)
                    active_pos = None
                else:
                    print("CRITICAL: EXIT ORDER FAILED. WILL RETRY IN NEXT SCAN.")
        except Exception as e:
            print(f"Position Management Error: {e}")
        return

    # Scan for Entries
    print("Scanning stocks for entries...")
    tickers = [t + ".NS" for t in STOCK_META.keys()]
    try:
        data = yf.download(tickers, period="1d", interval="1m", group_by='ticker', progress=False)
    except: return

    for sym in tickers:
        df = pd.DataFrame()
        if sym in data.columns.levels[0]: df = data[sym].copy()
        elif sym in data: df = data[sym].copy()
        
        if df.empty or len(df) < 60: continue
        
        pdf = add_features(df, sym)
        prob = get_ai_prediction(pdf)
        
        if prob >= AI_THRESHOLD:
            ticker_raw = sym.replace('.NS', '')
            curr_p = pdf['close'].iloc[-1]
            beta = pdf['beta'].iloc[-1]
            
            # Simple Capital Math
            # Assuming ₹11,700 per trade
            qty = int(11700 / curr_p)
            if qty < 1: continue
            
            print(f"NUCLEAR SIGNAL! {ticker_raw} | Prob: {prob:.1%} | Price: ₹{curr_p}")
            order_id = place_order_with_retry(
                side=kite.TRANSACTION_TYPE_BUY,
                ticker=ticker_raw,
                quantity=qty
            )
            
            if order_id:
                target = curr_p * (1 + BASE_TARGET_PCT * beta)
                sl = curr_p * (1 - STOP_LOSS_PCT/100.0)
                active_pos = LivePosition(ticker_raw, curr_p, qty, sl, target, beta)
                logger.log_trade("buy", ticker_raw, curr_p, qty, reason=f"Confidence: {prob:.1%}")
                print(f"Position Tracked: SL: ₹{sl:.2f}, Target: ₹{target:.2f}")
                break 
            else:
                print(f"ENTRY FAILED for {ticker_raw}. Skipping.")

if __name__ == "__main__":
    print("\n☢️  LIVE TRADER KITE V13 SUPREME ACTIVE")
    print("Concentration: One stock at a time | Threshold: 0.75\n")
    while True:
        try:
            scan_and_trade()
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Loop Error: {e}")
        time.sleep(CHECK_INTERVAL_LIVE)

