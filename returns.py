import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Sort by time (MANDATORY)
df = df.sort_index()

# -----------------------------
# 2. Clean price column (ROBUST)
# -----------------------------
df["Price"] = df["Price"].astype(str)

df["Close"] = (
    df["Price"]
    .str.replace(",", "", regex=False)
)

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# DROP rows where Close is NaN
df = df.dropna(subset=["Close"])

print("Close price check:")
print(df["Close"].head())
print("Close dtype:", df["Close"].dtype)
print("Number of rows:", len(df))

# -----------------------------
# 3. Plot prices
# -----------------------------
plt.figure()
plt.plot(df["Close"])
plt.title("Closing Price")
plt.show()

# -----------------------------
# 4. Compute log returns
# -----------------------------
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna(subset=["log_return"])

print("\nLog return stats:")
print(df["log_return"].describe())

# -----------------------------
# 5. Plot log returns (percent)
# -----------------------------
plt.figure()
plt.plot(df["log_return"] * 100)
plt.title("Log Returns (%)")
plt.show()

# -----------------------------
# 6. Returns distribution
# -----------------------------
plt.figure()
plt.hist(df["log_return"] * 100, bins=50)
plt.title("Distribution of Log Returns (%)")
plt.show()

# -----------------------------
# 7. Rolling volatility
# -----------------------------
window = 20
df["rolling_vol"] = df["log_return"].rolling(window).std() * np.sqrt(252)

def detect_vol_regime(vol_series, current_vol):
    """
    Classify volatility regime using quantiles.
    """
    if len(vol_series.dropna()) < 50:
        return "INSUFFICIENT_DATA"

    low_q = vol_series.quantile(0.33)
    high_q = vol_series.quantile(0.66)

    if current_vol < low_q:
        return "LOW_VOL"
    elif current_vol < high_q:
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"

print("\nRolling vol preview:")
print(df["rolling_vol"].dropna().head())

# -----------------------------
# 8. Volatility regime labeling
# -----------------------------
df["vol_regime"] = None

for i in range(len(df)):
    current_vol = df["rolling_vol"].iloc[i]
    hist_vol = df["rolling_vol"].iloc[:i]

    df.iloc[i, df.columns.get_loc("vol_regime")] = detect_vol_regime(
        hist_vol, current_vol
    )

# -----------------------------
# 9. Plot volatility with regimes
# -----------------------------
plt.figure(figsize=(12, 5))

colors = {
    "LOW_VOL": "green",
    "NORMAL_VOL": "orange",
    "STRESS_VOL": "red",
    "INSUFFICIENT_DATA": "gray"
}

for regime, color in colors.items():
    mask = df["vol_regime"] == regime
    plt.scatter(
        df.index[mask],
        df.loc[mask, "rolling_vol"],
        label=regime,
        color=color,
        s=10
    )

plt.plot(df["rolling_vol"], color="black", alpha=0.4)
plt.title("Rolling Volatility with Regime Classification")
plt.legend()
plt.show()


# -----------------------------
# 8. Plot volatility
# -----------------------------
plt.figure()
plt.plot(df["rolling_vol"])
plt.title("Rolling Volatility (Annualized)")
plt.show()

from arch import arch_model

# -----------------------------
# 9. Prepare returns for GARCH
# -----------------------------
# GARCH expects returns in %
returns_pct = df["log_return"] * 100

# -----------------------------
# 10. Fit GARCH(1,1)
# -----------------------------
garch = arch_model(
    returns_pct,
    vol="Garch",
    p=1,
    q=1,
    mean="Zero",
    dist="normal"
)

garch_result = garch.fit(disp="off")

print("\nGARCH Model Summary:")
print(garch_result.summary())

# -----------------------------
# 11. Conditional volatility
# -----------------------------
df["garch_vol"] = garch_result.conditional_volatility / 100

# -----------------------------
# 12. Plot Rolling vs GARCH Vol
# -----------------------------
plt.figure()
plt.plot(df["rolling_vol"], label="Rolling Vol (20d)")
plt.plot(df["garch_vol"], label="GARCH Vol")
plt.legend()
plt.title("Rolling Volatility vs GARCH Volatility")
plt.show()

from scipy.stats import norm

# -----------------------------
# 13. Black-Scholes Call Option
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )
    return call_price

# -----------------------------
# 14. Option pricing inputs
# -----------------------------
S = df["Close"].iloc[-1]          # current price
K = S                             # ATM option
T = 30 / 365                      # 30 days to expiry
r = 0.05                          # 5% risk-free rate

sigma_garch = df["garch_vol"].iloc[-1]

call_price = black_scholes_call(S, K, T, r, sigma_garch)

print("\nOption Pricing (GARCH Volatility)")
print(f"Spot Price (S): {S:.2f}")
print(f"Volatility (σ): {sigma_garch:.2%}")
print(f"Call Option Price: {call_price:.2f}")

print("\nSensitivity to Volatility:")
for v in [0.5, 0.75, 1.0, 1.25]:
    price = black_scholes_call(S, K, T, r, sigma_garch * v)
    print(f"σ x {v:.2f} → Call Price: {price:.2f}")

# -----------------------------
# 15. Monte Carlo price paths
# -----------------------------
def monte_carlo_paths(S0, T, r, sigma, n_paths=10000, n_steps=252):
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * z
        )

    return paths

# -----------------------------
# 16. Monte Carlo call pricing
# -----------------------------
paths = monte_carlo_paths(
    S0=S,
    T=T,
    r=r,
    sigma=sigma_garch,
    n_paths=20000
)

payoffs = np.maximum(paths[:, -1] - K, 0)
mc_price = np.exp(-r * T) * payoffs.mean()

print("\nMonte Carlo Call Price:")
print(f"MC Price: {mc_price:.2f}")
print(f"Black-Scholes Price: {call_price:.2f}")

# -----------------------------
# 17. Payoff distribution
# -----------------------------
plt.figure()
plt.hist(payoffs, bins=50)
plt.title("Distribution of Option Payoffs")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()

print("\nVolatility Stress Test:")
for shock in [0.75, 1.0, 1.25, 1.5]:
    stressed_paths = monte_carlo_paths(
        S0=S,
        T=T,
        r=r,
        sigma=sigma_garch * shock,
        n_paths=10000
    )

    stressed_payoff = np.maximum(stressed_paths[:, -1] - K, 0)
    stressed_price = np.exp(-r * T) * stressed_payoff.mean()

    print(f"σ x {shock:.2f} → MC Price: {stressed_price:.2f}")

import MetaTrader5 as mt5
from datetime import datetime, timedelta

# -----------------------------
# MT5 INITIALIZATION
# -----------------------------
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
N_BARS = 500  # rolling window

def fetch_live_data():
    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        N_BARS
    )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    return df

def compute_live_volatility(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    # Rolling vol
    rolling_vol = df["log_return"].rolling(50).std() * np.sqrt(252)

    # GARCH
    returns_pct = df["log_return"] * 100
    garch = arch_model(returns_pct, p=1, q=1, mean="Zero")
    res = garch.fit(disp="off")

    garch_vol = res.conditional_volatility.iloc[-1] / 100

    return rolling_vol.iloc[-1], garch_vol


def live_pricing_loop():
    while True:
        df = fetch_live_data()

        S = df["close"].iloc[-1]
        rolling_vol, garch_vol = compute_live_volatility(df)

        call_price = black_scholes_call(
            S=S,
            K=S,
            T=30/365,
            r=0.05,
            sigma=garch_vol
        )

        print(f"""
        ===== LIVE RISK SNAPSHOT =====
        Time: {df.index[-1]}
        Spot: {S:.2f}
        Rolling Vol: {rolling_vol:.2%}
        GARCH Vol: {garch_vol:.2%}
        Fair Call Price: {call_price:.2f}
        """)

        time.sleep(300)  # update every 5 minutes

def real_time_simulation():
    import time

    print("\n=== STARTING REAL-TIME SIMULATION ===")

    for i in range(20, len(df)):
        window_df = df.iloc[:i]

        # recompute rolling vol on expanding window
        rolling_vol_rt = (
            window_df["log_return"]
            .rolling(20)
            .std()
            .iloc[-1]
            * np.sqrt(252)
        )

        # GARCH on expanding window
        returns_pct = window_df["log_return"] * 100
        garch = arch_model(returns_pct, p=1, q=1, mean="Zero")
        res = garch.fit(disp="off")

        garch_vol_rt = res.conditional_volatility.iloc[-1] / 100

        S_rt = window_df["Close"].iloc[-1]

        call_rt = black_scholes_call(
            S=S_rt,
            K=S_rt,
            T=30/365,
            r=0.05,
            sigma=garch_vol_rt
        )

        print(f"""
        ===== LIVE ENGINE UPDATE =====
        Time Index: {window_df.index[-1]}
        Spot: {S_rt:.2f}
        Rolling Vol: {rolling_vol_rt:.2%}
        GARCH Vol: {garch_vol_rt:.2%}
        Call Price: {call_rt:.2f}
        """)

        time.sleep(2)  # simulate live updates

