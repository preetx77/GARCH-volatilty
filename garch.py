import pandas as pd
import numpy as np
import time

from arch import arch_model
from scipy.stats import norm

# =============================
# 1. LOAD & CLEAN DATA
# =============================

df = pd.read_csv("data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.sort_index()

# Clean price column (Investing.com style)
df["Price"] = df["Price"].astype(str)
df["Close"] = (
    df["Price"]
    .str.replace(",", "", regex=False)
)

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

# =============================
# 2. RETURNS
# =============================

df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna(subset=["log_return"])

print("DATA READY")
print("Rows:", len(df))
print(df.head())

# =============================
# 3. BLACK-SCHOLES FUNCTION
# =============================

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# =============================
# 4. REAL-TIME ENGINE LOOP
# =============================

WINDOW = 20          # rolling window
RISK_FREE = 0.05
T_EXPIRY = 30 / 365  # 30-day option

print("\n=== STARTING REAL-TIME ENGINE ===\n")

for i in range(WINDOW + 5, len(df)):

    live_df = df.iloc[:i]

    # -------- Rolling Volatility --------
    rolling_vol = (
        live_df["log_return"]
        .rolling(WINDOW)
        .std()
        .iloc[-1]
        * np.sqrt(252)
    )

    # -------- GARCH Volatility --------
    returns_pct = live_df["log_return"] * 100

    try:
        garch = arch_model(
            returns_pct,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero"
        )
        res = garch.fit(disp="off")
        garch_vol = res.conditional_volatility.iloc[-1] / 100
    except Exception:
        garch_vol = rolling_vol

    # -------- Option Pricing --------
    S = live_df["Close"].iloc[-1]
    K = S

    call_price = black_scholes_call(
        S=S,
        K=K,
        T=T_EXPIRY,
        r=RISK_FREE,
        sigma=garch_vol
    )

    # -------- OUTPUT --------
    print("===================================")
    print("TIME:", live_df.index[-1])
    print(f"SPOT PRICE     : {S:.2f}")
    print(f"ROLLING VOL    : {rolling_vol:.2%}")
    print(f"GARCH VOL      : {garch_vol:.2%}")
    print(f"CALL PRICE     : {call_price:.2f}")
    print("===================================\n")

    time.sleep(2)

print("ENGINE STOPPED (END OF DATA)")
