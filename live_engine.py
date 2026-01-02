import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time

from arch import arch_model
from scipy.stats import norm

# =============================
# 1. MT5 INITIALIZATION
# =============================

if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 500
RISK_FREE = 0.05
T_EXPIRY = 30 / 365

print("MT5 CONNECTED")

# =============================
# 2. BLACK-SCHOLES FUNCTION
# =============================

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# =============================
# 3. VOLATILITY REGIME FUNCTION
# =============================

def detect_vol_regime(vol_series, current_vol):
    if len(vol_series) < 100:
        return "INSUFFICIENT_DATA"

    low_th = vol_series.quantile(0.33)
    high_th = vol_series.quantile(0.66)

    if current_vol < low_th:
        return "LOW_VOL"
    elif current_vol < high_th:
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"

# =============================
# 4. STATE INITIALIZATION
# =============================

vol_history = []   # persistent state

print("\n=== LIVE VOLATILITY & PRICING ENGINE STARTED ===\n")

# =============================
# TAIL RISK FUNCTIONS
# =============================

def compute_var_es(returns, alpha=0.95):
    """
    returns: array-like of returns (log returns)
    alpha: confidence level (e.g., 0.95)
    """
    returns = np.sort(returns)

    var_index = int((1 - alpha) * len(returns))
    var = returns[var_index]

    es = returns[:var_index].mean() if var_index > 0 else var

    return var, es

# =============================
# 5. LIVE ENGINE LOOP
# =============================

while True:

    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        BARS
    )

    if rates is None or len(rates) < 100:
        print("Waiting for enough market data...")
        time.sleep(5)
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    # -------------------------
    # RETURNS
    # -------------------------
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)

    # -------------------------
    # TAIL RISK (VaR & ES)
    # -------------------------

    recent_returns = df["log_return"].iloc[-250:]  # ~1 year M5 approx scaled

    var_95, es_95 = compute_var_es(recent_returns, alpha=0.95)

    
    # -------------------------
    # ROLLING VOLATILITY
    # -------------------------
    rolling_vol = (
        df["log_return"]
        .rolling(50)
        .std()
        .iloc[-1]
        * np.sqrt(252)
    )

    # -------------------------
    # UPDATE VOL HISTORY
    # -------------------------
    vol_history.append(rolling_vol)
    vol_series = pd.Series(vol_history[-500:])

    # -------------------------
    # REGIME DETECTION
    # -------------------------
    regime = detect_vol_regime(vol_series, rolling_vol)

    # -------------------------
    # GARCH VOLATILITY
    # -------------------------
    returns_pct = df["log_return"] * 100

    try:
        garch = arch_model(
            returns_pct,
            p=1,
            q=1,
            mean="Zero",
            vol="Garch",
            rescale=False
        )
        res = garch.fit(disp="off")
        garch_vol = res.conditional_volatility.iloc[-1] / 100
    except Exception:
        garch_vol = rolling_vol

    # -------------------------
    # OPTION PRICING
    # -------------------------
    S = df["close"].iloc[-1]
    K = S

    call_price = black_scholes_call(
        S=S,
        K=K,
        T=T_EXPIRY,
        r=RISK_FREE,
        sigma=garch_vol
    )

    # -------------------------
    # OUTPUT
    # -------------------------
    print("======================================")
    print("TIME:", df.index[-1])
    print(f"SPOT PRICE : {S:.2f}")
    print(f"ROLLING VOL: {rolling_vol:.2%}")
    print(f"GARCH VOL  : {garch_vol:.2%}")
    print(f"VOL REGIME : {regime}")
    print(f"CALL PRICE : {call_price:.2f}")


    print("\n--- TAIL RISK (95%) ---")
    print(f"VaR (1-step): {var_95:.4%}")
    print(f"ES  (1-step): {es_95:.4%}")

    
    if regime == "STRESS_VOL":
        print("⚠️  RISK MODE: Volatility Expansion")
    elif regime == "LOW_VOL":
        print("🟢 CALM MODE: Compressed Risk")
    else:
        print("🟡 NORMAL MODE")

    print("======================================\n")

    time.sleep(300)
