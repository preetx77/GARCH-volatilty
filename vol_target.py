import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================

CSV_PATH = "data.csv"
ROLLING_WINDOW = 20        # safe default
TARGET_VOL = 0.08
MAX_LEVERAGE = 1.0
CONFIRM_BARS = 3

# ============================
# LOAD & SANITIZE DATA
# ============================

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# Handle time index
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
elif "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
else:
    raise ValueError("No date/time column found")

# Clean price
price_col = "close" if "close" in df.columns else "price"
df[price_col] = (
    df[price_col]
    .astype(str)
    .str.replace(",", "", regex=False)
)
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df.rename(columns={price_col: "close"}, inplace=True)

df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]

print(f"Rows after initial cleaning: {len(df)}")

# ============================
# FILL SMALL GAPS (CRITICAL)
# ============================

df["close"] = df["close"].ffill(limit=2)
df.dropna(subset=["close"], inplace=True)

print(f"Rows after gap handling: {len(df)}")

# ============================
# RETURNS
# ============================

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(subset=["log_return"], inplace=True)

print(f"Rows after returns: {len(df)}")

# ============================
# DYNAMIC ROLLING WINDOW CHECK
# ============================

if len(df) < ROLLING_WINDOW:
    raise RuntimeError(
        f"Not enough data ({len(df)}) for rolling window ({ROLLING_WINDOW})"
    )

# ============================
# ROLLING VOLATILITY
# ============================

df["rolling_vol"] = (
    df["log_return"]
    .rolling(
        window=ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW
    )
    .std()
    * np.sqrt(252)
)

df.dropna(subset=["rolling_vol"], inplace=True)
print(f"Rows after rolling vol: {len(df)}")

# ============================
# REGIME DETECTION
# ============================

low_q = df["rolling_vol"].quantile(0.33)
high_q = df["rolling_vol"].quantile(0.66)

def classify(vol):
    if vol < low_q:
        return "LOW_VOL"
    elif vol < high_q:
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"

df["raw_regime"] = df["rolling_vol"].apply(classify)

# ============================
# CONFIRMATION LOGIC
# ============================

confirmed = []
count = 0

for r in df["raw_regime"]:
    if r == "STRESS_VOL":
        count += 1
    else:
        count = 0

    confirmed.append("STRESS_VOL" if count >= CONFIRM_BARS else r)

df["vol_regime"] = confirmed

print("\nRegime distribution:")
print(df["vol_regime"].value_counts())

# ============================
# VOLATILITY TARGETING
# ============================

def position_size(vol, regime):
    if regime == "STRESS_VOL":
        return 0.0
    return min(TARGET_VOL / vol, MAX_LEVERAGE)

df["position"] = df.apply(
    lambda x: position_size(x["rolling_vol"], x["vol_regime"]),
    axis=1
)

# ============================
# STRATEGY RETURNS
# ============================

df["strategy_return"] = df["position"] * df["log_return"]
df["equity"] = np.exp(df["strategy_return"].cumsum())

# ============================
# DIAGNOSTICS
# ============================

print("\n==============================")
print("AVERAGE VOLATILITY BY REGIME")
print("==============================")
print(df.groupby("vol_regime")["rolling_vol"].mean())

print("\n==============================")
print("5% TAIL LOSSES BY REGIME")
print("==============================")
print(df.groupby("vol_regime")["log_return"].quantile(0.05))

# ============================
# PLOTS (ONLY IF VALID)
# ============================

if df.empty:
    raise RuntimeError("No valid data left to plot")

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["equity"], label="Risk-Controlled Equity")
plt.title("Regime-Aware Volatility Targeting (Hard Stop)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 4))
colors = {"LOW_VOL": "green", "NORMAL_VOL": "orange", "STRESS_VOL": "red"}

for reg, col in colors.items():
    sub = df[df["vol_regime"] == reg]
    plt.scatter(sub.index, sub["rolling_vol"], s=10, color=col, label=reg)

plt.title("Volatility Regimes")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.grid(True)
plt.show()
