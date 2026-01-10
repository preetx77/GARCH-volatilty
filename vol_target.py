import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

CSV_PATH = "data.csv"     # your CSV file
ROLLING_WINDOW = 50
TARGET_VOL = 0.08         # 8% annualized target volatility
MAX_LEVERAGE = 1.0
CONFIRM_BARS = 3
INITIAL_CAPITAL = 1.0

# =========================
# LOAD & CLEAN DATA
# =========================

df = pd.read_csv(CSV_PATH)

# normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# handle date/time
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
elif "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
else:
    raise ValueError("No date/time column found")

# handle price column
if "close" in df.columns:
    price_col = "close"
elif "price" in df.columns:
    price_col = "price"
else:
    raise ValueError("No close/price column found")

# clean price (remove commas, convert to float)
df[price_col] = (
    df[price_col]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

df.rename(columns={price_col: "close"}, inplace=True)

df.dropna(subset=["close"], inplace=True)

# =========================
# RETURNS
# =========================

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)

# =========================
# ROLLING VOLATILITY
# =========================

df["rolling_vol"] = (
    df["log_return"]
    .rolling(ROLLING_WINDOW)
    .std()
    * np.sqrt(252)
)

df.dropna(inplace=True)

# =========================
# REGIME DETECTION
# =========================

low_q = df["rolling_vol"].quantile(0.33)
high_q = df["rolling_vol"].quantile(0.66)

def classify_regime(vol):
    if vol < low_q:
        return "LOW_VOL"
    elif vol < high_q:
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"

df["raw_regime"] = df["rolling_vol"].apply(classify_regime)

# =========================
# STRESS CONFIRMATION
# =========================

confirmed = []
counter = 0

for r in df["raw_regime"]:
    if r == "STRESS_VOL":
        counter += 1
    else:
        counter = 0

    if counter >= CONFIRM_BARS:
        confirmed.append("STRESS_VOL")
    else:
        confirmed.append(r)

df["vol_regime"] = confirmed

# =========================
# POSITION SIZING (RISK ENGINE)
# =========================

def position_size(vol, regime):
    if regime == "STRESS_VOL":
        return 0.0
    size = TARGET_VOL / vol
    return min(size, MAX_LEVERAGE)

df["position"] = df.apply(
    lambda x: position_size(x["rolling_vol"], x["vol_regime"]),
    axis=1
)

# =========================
# STRATEGY RETURNS (RISK ONLY)
# =========================

df["strategy_return"] = df["position"] * df["log_return"]
df["equity"] = INITIAL_CAPITAL * np.exp(df["strategy_return"].cumsum())

# =========================
# DIAGNOSTICS
# =========================

print("\n==============================")
print("AVERAGE VOLATILITY BY REGIME")
print("==============================")
print(df.groupby("vol_regime")["rolling_vol"].mean())

print("\n==============================")
print("5% TAIL LOSSES BY REGIME")
print("==============================")
print(df.groupby("vol_regime")["log_return"].quantile(0.05))

# =========================
# PLOTS
# =========================

plt.figure(figsize=(14, 6))
plt.plot(df.index, df["equity"], label="Risk-Controlled Equity")
plt.title("Regime-Aware Volatility Targeting (Hard Stop)")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 4))
colors = {
    "LOW_VOL": "green",
    "NORMAL_VOL": "orange",
    "STRESS_VOL": "red"
}

for reg, col in colors.items():
    subset = df[df["vol_regime"] == reg]
    plt.scatter(
        subset.index,
        subset["rolling_vol"],
        color=col,
        s=10,
        label=reg
    )

plt.title("Volatility Regimes")
plt.ylabel("Rolling Volatility")
plt.legend()
plt.grid(True)
plt.show()
