import os
import time
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import pytz
import pyotp
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from kiteconnect import KiteConnect
import warnings
from logger import logger
import yfinance as yf

warnings.filterwarnings("ignore")

# ================== CREDENTIALS & API CONFIG ==================
USER_ID = "THS720"
PASSWORD = "Chits@23"
TOTP_SECRET = "E5HRMPZEN4QOOORKIPR5UX66E5SAG4VY"

API_KEY = "vv25p1x1xjh0gpnr"
API_SECRET = "2mj8gklqv9hjf51a3vuf31mm4xgety5f"
TOKEN_FILE = "access_token.txt"
MODEL_PATH = "recovered_model.cbm"
INITIAL_CAPITAL = 11700

# ================== OPTIMAL PARAMETERS (FROM BACKTEST) ==================
AI_THRESHOLD = 0.70
BASE_TARGET_PCT = 0.050          # 5.0%
STOP_LOSS_PCT = 3.0               # 3.0%
SQUEEZE_THRESH = 0.00080
SQUEEZE_WIDTH_PCT = 0.0005
HOLD_BARS_NORMAL = 4             # Core hold period (60 minutes)
MAX_HOLD_BARS = 8                # Absolute max bars before forced exit
MAX_CONCURRENT_TRADES = 20
CHECK_INTERVAL_LIVE = 60         # Check every ~1 minute (15m data updates slowly)

MAX_DAILY_LOSS = -2000
TRADE_LOG_FILE = "v17_trade_log.csv"

TIMEZONE = "Asia/Kolkata"

# Session State
SESSION_PNL = 0.0
IS_Halt = False

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

UNIQUE_SECTORS = ['Auto', 'Banking', 'Cables', 'Cement', 'Conglomerate', 'Diversified', 'Electronics', 'Energy',
                  'Finance', 'Fintech', 'Gas', 'Housing Finance', 'IT', 'Infra', 'Jewellery', 'Logistics',
                  'Metals', 'Mining', 'Pharma', 'Power', 'Railway', 'Railway Finance', 'Retail', 'Tech', 'Telecom']

# ================== INITIALIZATION ==================
def auto_kite_login():
    print("Attempting Automated Kite Login (Headless)...")
    kite_api = KiteConnect(api_key=API_KEY)
    login_url = kite_api.login_url()
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    try:
        pi_driver_path = "/usr/bin/chromedriver"
        if os.path.exists(pi_driver_path):
            service = Service(executable_path=pi_driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            driver = webdriver.Chrome(options=chrome_options)
        
        driver.get(login_url)
        wait = WebDriverWait(driver, 20)
        
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(USER_ID)
        driver.find_element(By.ID, "password").send_keys(PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        
        clean_secret = TOTP_SECRET.replace(" ", "")
        totp = pyotp.TOTP(clean_secret)
        otp_code = totp.now()
        
        pin_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='number'], input[label='TOTP']")))
        pin_input.send_keys(otp_code)
        
        time.sleep(5)
        
        if "request_token=" in driver.current_url:
            parsed_url = urllib.parse.urlparse(driver.current_url)
            request_token = urllib.parse.parse_qs(parsed_url.query)['request_token'][0]
            session = kite_api.generate_session(request_token, api_secret=API_SECRET)
            access_token = session["access_token"]
            
            with open(TOKEN_FILE, "w") as f:
                f.write(access_token)
            print(f"SUCCESS: New Access Token saved to {TOKEN_FILE}")
            driver.quit()
            return access_token
        else:
            print("FAILED: Redirect URL not found.")
            driver.quit()
            return None
    except Exception as e:
        print(f"FAILED: Auto-login error: {e}")
        return None

def initialize_kite():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
    else:
        token = auto_kite_login()

    if not token:
        print("FATAL: Could not acquire access token.")
        exit(1)

    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(token)

    try:
        profile = k.profile()
        print(f"OK: Logged in as {profile['user_name']}")
        return k
    except Exception:
        print("Token expired or invalid. Attempting fresh login...")
        new_token = auto_kite_login()
        if new_token:
            k.set_access_token(new_token)
            return k
        else:
            print("FATAL: Re-login failed.")
            exit(1)

kite = initialize_kite()

print(f"Loading AI Model: {MODEL_PATH}...")
try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print("OK: Model Loaded successfully.")
except Exception as e:
    print(f"FATAL: Model loading failed: {e}")
    exit(1)

base_features = ['cum_delta','delta','msb','range_filter_dist','ob_dist_pct',
                'vol_ratio','bars_since_msb','atr_ratio','vwap_dist_pct','ma20_dist_pct',
                'rvol','adx','sess_high_dist_pct','sess_low_dist_pct',
                'is_high_vol_bar','delta_zscore','atr_ratio_slope','rvol_slope','adx_slope']
surgical_features = ['cmf', 'bb_percent', 'macd_hist_slope']
features = base_features + ['beta'] + surgical_features + [f'sector_{sec}' for sec in UNIQUE_SECTORS]

# ================== CHARGES (DETAILED - KEPT AS ORIGINAL) ==================
def calculate_charges(buy_p, sell_p, qty):
    buy_val = buy_p * qty
    sell_val = sell_p * qty
    turnover = buy_val + sell_val
    
    buy_brokerage = min(20.0, buy_val * 0.0003)
    sell_brokerage = min(20.0, sell_val * 0.0003)
    total_brokerage = buy_brokerage + sell_brokerage
    
    stt = sell_val * 0.00025
    txn_charge = turnover * 0.0000325
    sebi_charge = turnover * 0.000001
    gst = (total_brokerage + txn_charge + sebi_charge) * 0.18
    stamp_duty = buy_val * 0.00003
    
    total_charges = total_brokerage + stt + txn_charge + sebi_charge + gst + stamp_duty
    breakeven_pts = total_charges / qty
    
    return {
        "total": total_charges,
        "breakeven_pts": breakeven_pts
    }

def log_v17_trade(data):
    df = pd.DataFrame([data])
    if not os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, index=False)
    else:
        df.to_csv(TRADE_LOG_FILE, mode='a', header=False, index=False)

def place_order_with_retry(side, ticker, quantity, price=None, order_type=KiteConnect.ORDER_TYPE_MARKET):
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
            if "insufficient" in err_str or "invalid" in err_str or "margin" in err_str:
                print(f"PERMANENT ERROR: {e}")
                return None
            
            if attempt == max_retries - 1:
                print(f"FATAL: Order failed after {max_retries} attempts: {e}")
                return None
            
            print(f"RETRYABLE ERROR: {e}. Waiting {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2

# ================== FEATURE ENGINEERING (SAME AS BACKTEST) ==================
def add_features(df, stock_symbol):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    
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
        probs = model.predict_proba(latest_data[features])
        return float(probs[0][1])
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0

# ================== POSITION CLASS ==================
class LivePosition:
    def __init__(self, ticker, entry_price, quantity, initial_stop, initial_target, beta):
        self.ticker = ticker
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = initial_stop
        self.target_price = initial_target
        self.beta = beta
        self.peak_price = entry_price
        self.bars_held = 0
        self.squeeze_applied = False

active_positions = {}

# ================== CORE TRADING LOOP ==================
def scan_and_trade():
    global active_positions, SESSION_PNL, IS_Halt
    if IS_Halt: return

    now = datetime.now(pytz.timezone(TIMEZONE))
    tickers = [t + ".NS" for t in STOCK_META.keys()]

    # Fetch 15m data for all stocks at once
    try:
        data_15m = yf.download(tickers, period="5d", interval="15m", group_by='ticker', progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Data fetch error: {e}")
        return

    # 1. Manage Existing Positions
    for ticker_raw in list(active_positions.keys()):
        pos = active_positions[ticker_raw]
        try:
            df = data_15m[ticker_raw + ".NS"].dropna()
            if len(df) < 2: continue

            current_bar = df.iloc[-1]
            curr_p = current_bar['Close']
            pos.peak_price = max(pos.peak_price, curr_p)
            pos.bars_held += 1

            exit_reason = None
            exit_price = curr_p

            if curr_p <= pos.stop_loss:
                exit_reason = "SL"
                exit_price = pos.stop_loss
            elif curr_p >= pos.target_price:
                exit_reason = "TARGET"
                exit_price = pos.target_price
            elif pos.bars_held == 6 and not pos.squeeze_applied:  # After exactly 6 bars (~90min)
                move_pct = (curr_p - pos.entry_price) / pos.entry_price
                if move_pct < SQUEEZE_THRESH:
                    new_sl = pos.peak_price * (1 - SQUEEZE_WIDTH_PCT)
                    pos.stop_loss = max(pos.stop_loss, new_sl)
                    pos.squeeze_applied = True
                    print(f"⚠️ SQUEEZE: {ticker_raw} | Tightened SL to {new_sl:.2f}")
            elif pos.bars_held >= MAX_HOLD_BARS:
                exit_reason = "MAX_TIME"
            elif now.hour >= 15 and now.minute >= 15:
                exit_reason = "EOD"

            if exit_reason:
                order_id = place_order_with_retry(side=kite.TRANSACTION_TYPE_SELL, ticker=ticker_raw, quantity=pos.quantity)
                if order_id:
                    c_info = calculate_charges(pos.entry_price, exit_price, pos.quantity)
                    pnl = (exit_price - pos.entry_price) * pos.quantity - c_info['total']
                    SESSION_PNL += pnl

                    log_v17_trade({
                        "time": now, "ticker": ticker_raw, "side": "SELL", "price": exit_price,
                        "qty": pos.quantity, "pnl": pnl, "charges": c_info['total'], "reason": exit_reason
                    })
                    print(f"✅ EXIT {ticker_raw} | {exit_reason} @ {exit_price:.2f} | PnL: ₹{pnl:.2f}")
                    del active_positions[ticker_raw]

                    if SESSION_PNL <= MAX_DAILY_LOSS:
                        print(f"🛑 DAILY LOSS LIMIT HIT: ₹{SESSION_PNL:.2f}")
                        IS_Halt = True
        except Exception as e:
            print(f"Error managing {ticker_raw}: {e}")

    # 2. Scan for New Entries
    if len(active_positions) >= MAX_CONCURRENT_TRADES:
        return
    if now.hour >= 15 and now.minute >= 10:
        return

    for sym in tickers:
        ticker_raw = sym.replace('.NS', '')
        if ticker_raw in active_positions: continue

        try:
            df = data_15m[sym].dropna()
            if len(df) < 50: continue

            pdf = add_features(df, sym)
            if pdf.empty: continue

            prob = get_ai_prediction(pdf)
            if prob < AI_THRESHOLD: continue

            curr_p = df['Close'].iloc[-1]
            qty = int(INITIAL_CAPITAL / curr_p)
            if qty < 1: continue

            print(f"🚀 SIGNAL: {ticker_raw} | Conf: {prob:.1%} | Price: ₹{curr_p:.2f}")

            order_id = place_order_with_retry(side=kite.TRANSACTION_TYPE_BUY, ticker=ticker_raw, quantity=qty)
            if order_id:
                target = curr_p * (1 + BASE_TARGET_PCT)
                sl = curr_p * (1 - STOP_LOSS_PCT / 100.0)

                active_positions[ticker_raw] = LivePosition(ticker_raw, curr_p, qty, sl, target, pdf['beta'].iloc[-1])

                log_v17_trade({
                    "time": now, "ticker": ticker_raw, "side": "BUY", "price": curr_p,
                    "qty": qty, "pnl": 0.0, "reason": f"Signal {prob:.1%}"
                })

                if len(active_positions) >= MAX_CONCURRENT_TRADES:
                    break
        except Exception as e:
            continue

if __name__ == "__main__":
    print("\n" + "="*60)
    print("☢️  V17 ALPHA HUNTER - 15M OPTIMAL MODE (BACKTEST ALIGNED)")
    print(f"Threshold: {AI_THRESHOLD} | Target: {BASE_TARGET_PCT:.1%} | SL: {STOP_LOSS_PCT}% | Max Hold: {MAX_HOLD_BARS} bars")
    print("="*60 + "\n")

    while True:
        try:
            now = datetime.now(pytz.timezone(TIMEZONE))
            status = "HALTED" if IS_Halt else "RUNNING"
            print(f"[{now.strftime('%H:%M:%S')}] {status} | Trades: {len(active_positions)}/{MAX_CONCURRENT_TRADES} | Session PnL: ₹{SESSION_PNL:.2f}", end="\r")
            
            scan_and_trade()
        except KeyboardInterrupt:
            print("\n👋 Shutdown requested...")
            break
        except Exception as e:
            print(f"\n⚠️ Main loop error: {e}")
        
        time.sleep(CHECK_INTERVAL_LIVE)