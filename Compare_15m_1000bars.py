import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

MODEL_PATH = "recovered_model.cbm"
INITIAL_CAPITAL = 11700.0

# BEST 15M CONFIGURATION (from optimization)
AI_THRESHOLD = 0.70
BASE_TARGET_PCT = 0.050  # 5.0%
STOP_LOSS_PCT = 0.030    # 3.0%
SQUEEZE_THRESHOLD = 0.0008
HOLD_TIME = 120  # minutes (4 bars of 15m)
BETA_WEIGHTED = False

# STOCK META
STOCK_META = {
    "TATASTEEL": {"Beta": 1.8}, "ADANIENT": {"Beta": 2.1}, "ADANIPORTS": {"Beta": 1.9},
    "JSWSTEEL": {"Beta": 2.0}, "HINDALCO": {"Beta": 1.7}, "VEDL": {"Beta": 2.3},
    "M&M": {"Beta": 1.7}, "BPCL": {"Beta": 2.1}, "IOC": {"Beta": 2.0},
    "RELIANCE": {"Beta": 1.4}, "SBIN": {"Beta": 1.5}, "HDFCBANK": {"Beta": 1.3},
    "ICICIBANK": {"Beta": 1.4}, "AXISBANK": {"Beta": 1.5}, "BAJFINANCE": {"Beta": 1.6},
    "SHRIRAMFIN": {"Beta": 1.7}, "INDUSINDBK": {"Beta": 1.6}, "EICHERMOT": {"Beta": 1.6},
    "HEROMOTOCO": {"Beta": 1.5}, "BAJAJ-AUTO": {"Beta": 1.4}, "MARUTI": {"Beta": 1.4},
    "LT": {"Beta": 1.5}, "ULTRACEMCO": {"Beta": 1.5}, "GRASIM": {"Beta": 1.4},
    "TITAN": {"Beta": 1.4}, "PAYTM": {"Beta": 2.0}, "NYKAA": {"Beta": 1.8},
    "DELHIVERY": {"Beta": 1.7}, "RVNL": {"Beta": 2.2}, "IRFC": {"Beta": 1.9},
    "HUDCO": {"Beta": 1.7}, "POLYCAB": {"Beta": 1.6}, "DIXON": {"Beta": 1.8},
    "SAIL": {"Beta": 1.8}, "JINDALSTEL": {"Beta": 1.9}, "NMDC": {"Beta": 1.6},
    "COALINDIA": {"Beta": 1.5}, "ONGC": {"Beta": 1.5}, "GAIL": {"Beta": 1.5},
    "POWERGRID": {"Beta": 1.4}, "NTPC": {"Beta": 1.4}, "HINDPETRO": {"Beta": 1.6},
    "BHARTIARTL": {"Beta": 1.4}, "INFY": {"Beta": 1.3}, "TCS": {"Beta": 1.2},
    "WIPRO": {"Beta": 1.3}, "HCLTECH": {"Beta": 1.3}, "SUNPHARMA": {"Beta": 1.4}
}

UNIQUE_SECTORS = ['Auto', 'Banking', 'Cables', 'Cement', 'Conglomerate', 'Diversified', 'Electronics', 'Energy', 'Finance', 'Fintech', 'Gas', 'Housing Finance', 'IT', 'Infra', 'Jewellery', 'Logistics', 'Metals', 'Mining', 'Pharma', 'Power', 'Railway', 'Railway Finance', 'Retail', 'Tech', 'Telecom']
base_features = ['cum_delta','delta','msb','range_filter_dist','ob_dist_pct','vol_ratio','bars_since_msb','atr_ratio','vwap_dist_pct','ma20_dist_pct','rvol','adx','sess_high_dist_pct','sess_low_dist_pct','is_high_vol_bar','delta_zscore','atr_ratio_slope','rvol_slope','adx_slope']
surgical_features = ['cmf', 'bb_percent', 'macd_hist_slope']
features = base_features + ['beta'] + surgical_features + [f'sector_{sec}' for sec in UNIQUE_SECTORS]

def add_features(df, sym):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy(); df.columns = [c.lower() for c in df.columns]
    try:
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        mf = ((c - l) - (h - c)) / (h - l + 1e-8) * v
        df['cmf'] = mf.rolling(20).sum() / v.rolling(20).sum().replace(0,1)
        ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        df['bb_percent'] = (c - (ma20 - 2*std20)) / (4*std20 + 1e-8)
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        df['macd_hist_slope'] = (macd - macd.ewm(span=9, adjust=False).mean()).diff()
        delta = v * (2*c - h - l) / (h - l + 1e-8); df['cum_delta'] = delta.rolling(50).sum(); df['delta'] = delta
        swing_h = h.rolling(20, center=True).max().ffill().shift(1); swing_l = l.rolling(20, center=True).min().ffill().shift(1)
        df['msb'] = np.where(h > swing_h, 1, np.where(l < swing_l, -1, 0)); df['bars_since_msb'] = df['msb'].abs().cumsum()
        tr = np.abs(np.diff(c, prepend=c.iloc[0]))
        avrng = pd.Series(tr, index=df.index).ewm(span=100, adjust=False).mean()
        smrng = avrng.ewm(span=199, adjust=False).mean() * 3
        filt = c.copy().values; c_val = c.values; smrng_val = smrng.values
        for i in range(1, len(df)):
            if c_val[i] > filt[i-1]: filt[i] = max(filt[i-1], c_val[i] - smrng_val[i])
            else: filt[i] = min(filt[i-1], c_val[i] + smrng_val[i])
        df['range_filter_dist'] = (c - filt) / c
        df['ob_dist_pct'] = np.abs(np.where(df['msb'] == 1, c - swing_l, swing_h - c) / c); df['vol_ratio'] = v / v.rolling(20).mean()
        tr_s = pd.Series(tr, index=df.index); df['atr_ratio'] = tr_s.rolling(14).mean() / tr_s.rolling(50).mean().replace(0,1)
        df['vwap'] = (( (h + l + c) / 3 ) * v).rolling(50).sum() / v.rolling(50).sum().replace(0,1); df['vwap_dist_pct'] = (c - df['vwap']) / c
        df['ma20_dist_pct'] = (c - c.rolling(20).mean()) / c.rolling(20).mean(); df['rvol'] = v / v.rolling(5).mean().shift(1).replace(0,1)
        plus_di = 100 * ((h - h.shift(1)).clip(lower=0) / (tr_s + 1e-8)).ewm(span=14, adjust=False).mean()
        minus_di = 100 * ((l.shift(1) - l).clip(lower=0) / (tr_s + 1e-8)).ewm(span=14, adjust=False).mean()
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8); df['adx'] = dx.ewm(span=14, adjust=False).mean()
        df['sess_high_dist_pct'] = (h.rolling(50).max() - c) / c; df['sess_low_dist_pct'] = (c - l.rolling(50).min()) / c
        df['is_high_vol_bar'] = (v > v.rolling(9).mean() * 2).astype(int)
        df['delta_zscore'] = (delta - delta.rolling(9).mean()) / delta.rolling(9).std().replace(0,1)
        df['atr_ratio_slope'] = df['atr_ratio'].diff(); df['rvol_slope'] = df['rvol'].diff(); df['adx_slope'] = df['adx'].diff()
        
        meta = STOCK_META.get(sym.replace('.NS',''), {"Beta": 1.0, "Sector": "Other"})
        df['beta'] = meta.get('Beta', 1.0)
        sec = meta.get('Sector', 'Other')
        for s in UNIQUE_SECTORS: df[f'sector_{s}'] = 1 if s == sec else 0
        return df
    except: return pd.DataFrame()

print("=" * 80)
print("V17 ALPHA - 15M STRATEGY ANALYSIS (1000 BARS)")
print("=" * 80)
print(f"Configuration: TH:{AI_THRESHOLD} | TP:{BASE_TARGET_PCT*100:.1f}% | SL:{STOP_LOSS_PCT*100:.1f}% | Hold:{HOLD_TIME}m")
print()

df_raw = pd.read_csv("nuclear_training_data.csv")  # 15m data
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

model = CatBoostClassifier(logging_level='Silent')
model.load_model(MODEL_PATH)

# Take last 1000 bars per symbol
all_signals = []
for sym in df_raw['symbol'].unique():
    sym_df = df_raw[df_raw['symbol'] == sym].copy().sort_values('datetime').tail(1000)
    if len(sym_df) < 500: continue
    
    sym_df.set_index('datetime', inplace=True)
    df_feat = add_features(sym_df, sym).dropna()
    if df_feat.empty: continue
    
    for f in features: 
        if f not in df_feat.columns: df_feat[f] = 0.0
    
    probs = model.predict_proba(df_feat[features])
    df_feat['prob'] = [p[1] for p in probs]
    
    # Use native 15m bars for outcome calculation
    c_arr = sym_df['close'].values; h_arr = sym_df['high'].values; l_arr = sym_df['low'].values
    idx_lookup = sym_df.index
    
    # Convert HOLD_TIME from minutes to 15m bars
    hold_bars = HOLD_TIME // 15
    
    for ts, row in df_feat[df_feat['prob'] >= AI_THRESHOLD].iterrows():
        try:
            loc = idx_lookup.get_loc(ts)
            obs_limit = min(len(c_arr)-loc-1, hold_bars)
            if obs_limit < 1: continue
            all_signals.append({
                'time': ts, 'entry': row['close'], 'beta': row['beta'],
                'h': h_arr[loc+1:loc+obs_limit+1],
                'l': l_arr[loc+1:loc+obs_limit+1],
                'c': c_arr[loc+1:loc+obs_limit+1]
            })
        except: continue

all_signals.sort(key=lambda x: x['time'])
print(f"Total Signals: {len(all_signals)}")

# Simulate
cash = INITIAL_CAPITAL; active = []; trades = 0; wins = 0
for sig in all_signals:
    active = [a for a in active if a > sig['time']]
    if len(active) >= 20: continue
    
    entry = sig['entry']
    mult = sig['beta'] if BETA_WEIGHTED else 1.0
    tp = entry * (1 + BASE_TARGET_PCT * mult)
    sl = entry * (1 - STOP_LOSS_PCT)
    
    h, l, c = sig['h'], sig['l'], sig['c']
    exit_p = c[-1] if len(c) > 0 else entry
    dur = len(c)
    
    for i in range(len(c)):
        if l[i] <= sl: exit_p = sl; dur = i+1; break
        if h[i] >= tp: exit_p = tp; dur = i+1; break
        if i == 6 and len(c) > 6:
            if (c[i]-entry)/entry < SQUEEZE_THRESHOLD:
                pk = max(h[:i+1])
                sq_sl = pk * (1 - 0.0005)
                if c[i] <= sq_sl: exit_p = c[i]; dur = i+1; break
    
    qty = int(11700 / entry)
    turn = (entry + exit_p) * qty
    tax = (turn * 0.0002) + 40
    pnl = (exit_p - entry) * qty - tax
    
    cash += pnl; trades += 1
    if pnl > 0: wins += 1
    active.append(sig['time'] + timedelta(minutes=dur))

# Calculate metrics
total_roi = ((cash - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
wr = (wins / trades * 100) if trades > 0 else 0

# Calculate trading days (15m bars: ~25 per day)
first_sig = all_signals[0]['time'] if all_signals else None
last_sig = all_signals[-1]['time'] if all_signals else None
trading_days = (last_sig - first_sig).days if first_sig and last_sig else 1
if trading_days == 0: trading_days = 1

avg_daily_roi = total_roi / trading_days

print(f"\n{'='*80}")
print("RESULTS:")
print(f"{'='*80}")
print(f"Total Trades:        {trades}")
print(f"Win Rate:            {wr:.1f}%")
print(f"Total ROI:           {total_roi:.2f}%")
print(f"Trading Period:      {trading_days} days")
print(f"AVG DAILY ROI:       {avg_daily_roi:.2f}%")
print(f"Final Capital:       ₹{cash:,.2f}")
print(f"{'='*80}")

