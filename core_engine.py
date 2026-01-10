import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm
import json
import os

# =============================
# CONFIG
# =============================

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 500
RISK_FREE = 0.05
T_EXPIRY = 30 / 365

MT5_FILES_DIR = r"C:\Users\LENOVO\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\Common"

# =============================
# MT5 INIT
# =============================

if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# =============================
# STATE
# =============================

vol_history = []

# =============================
# HELPERS
# =============================

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def detect_vol_regime(vol_series, current_vol):
    if len(vol_series) < 100:
        return "INSUFFICIENT_DATA"
    if current_vol < vol_series.quantile(0.33):
        return "LOW_VOL"
    elif current_vol < vol_series.quantile(0.66):
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"

def compute_var_es(returns, alpha=0.95):
    r = np.sort(returns)
    idx = int((1 - alpha) * len(r))
    var = r[idx]
    es = r[:idx].mean() if idx > 0 else var
    return var, es

def write_snapshot_to_mt5(snapshot):
    path = os.path.join(MT5_FILES_DIR, "mt5_risk_snapshot.json")

    payload = {
        "time": str(snapshot["time"]),
        "spot": float(snapshot["spot"]),
        "rolling_vol": float(snapshot["rolling_vol"]),
        "garch_vol": float(snapshot["garch_vol"]),
        "regime": snapshot["regime"],
        "call_price": float(snapshot["call_price"]),
        "var": float(snapshot["var"]),
        "es": float(snapshot["es"]),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# =============================
# PUBLIC API
# =============================

def get_live_snapshot():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
    print("MT5 bars received:", 0 if rates is None else len(rates))

    if rates is None or len(rates) < 100:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)

    rolling_vol = df["log_return"].rolling(50).std().iloc[-1] * np.sqrt(252)

    vol_history.append(rolling_vol)
    vol_series = pd.Series(vol_history[-500:])

    returns_pct = df["log_return"] * 100

    try:
        garch = arch_model(
            returns_pct,
            p=1,
            q=1,
            mean="Zero",
            rescale=False
        )
        res = garch.fit(disp="off")
        garch_vol = res.conditional_volatility.iloc[-1] / 100
    except Exception:
        garch_vol = rolling_vol

    S = df["close"].iloc[-1]
    call_price = black_scholes_call(S, S, T_EXPIRY, RISK_FREE, garch_vol)

    var95, es95 = compute_var_es(df["log_return"].iloc[-250:])
    regime = detect_vol_regime(vol_series, rolling_vol)

    snapshot = {
        "time": df.index[-1],
        "spot": S,
        "rolling_vol": rolling_vol,
        "garch_vol": garch_vol,
        "regime": regime,
        "call_price": call_price,
        "var": var95,
        "es": es95,
        "df": df
    }

    write_snapshot_to_mt5(snapshot)
    print(snapshot)
    return snapshot 

