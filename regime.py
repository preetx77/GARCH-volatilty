import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. LOAD DATA
# =============================

# Expecting columns like: Date, Close (or Price)
FILE_PATH = "data.csv"

df = pd.read_csv(FILE_PATH)

# Clean column names
df.columns = [c.strip().lower() for c in df.columns]

# Handle date
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

# Use close or price
if "close" in df.columns:
    price_col = "close"
elif "price" in df.columns:
    price_col = "price"
else:
    raise ValueError("CSV must contain 'Close' or 'Price' column")

df = df[[price_col]].dropna()
df.rename(columns={price_col: "price"}, inplace=True)
# -----------------------------
# Ensure price is numeric
# -----------------------------
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df.dropna(inplace=True)



# =============================
# 2. LOG RETURNS
# =============================

df["log_return"] = np.log(df["price"] / df["price"].shift(1))
df.dropna(inplace=True)

# =============================
# 3. ROLLING VOLATILITY
# =============================

WINDOW = 20  # ~1 month
df["rolling_vol"] = (
    df["log_return"]
    .rolling(WINDOW)
    .std()
    * np.sqrt(252)
)

df.dropna(inplace=True)

# =============================
# 4. REGIME DETECTION (QUANTILE-BASED)
# =============================

def detect_vol_regime(vol_series, current_vol):
    if len(vol_series) < 50:
        return "INSUFFICIENT_DATA"

    low_q = vol_series.quantile(0.33)
    high_q = vol_series.quantile(0.66)

    if current_vol < low_q:
        return "LOW_VOL"
    elif current_vol < high_q:
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"


df["vol_regime"] = None

for i in range(len(df)):
    hist_vol = df["rolling_vol"].iloc[:i]
    curr_vol = df["rolling_vol"].iloc[i]

    df.iloc[i, df.columns.get_loc("vol_regime")] = (
        detect_vol_regime(hist_vol, curr_vol)
    )

# =============================
# 5. REGIME STATISTICS (PROOF)
# =============================

print("\n==============================")
print("AVERAGE VOLATILITY BY REGIME")
print("==============================")
print(df.groupby("vol_regime")["rolling_vol"].mean())

print("\n==============================")
print("VOLATILITY PERSISTENCE (LAG-1)")
print("==============================")

for regime in df["vol_regime"].unique():
    subset = df[df["vol_regime"] == regime]["rolling_vol"]
    if len(subset) > 2:
        print(f"{regime}: {subset.autocorr(lag=1):.3f}")

print("\n==============================")
print("5% TAIL LOSSES BY REGIME")
print("==============================")
print(df.groupby("vol_regime")["log_return"].quantile(0.05))

# =============================
# 6. VISUALIZATION
# =============================

plt.figure(figsize=(14, 6))

colors = {
    "LOW_VOL": "green",
    "NORMAL_VOL": "orange",
    "STRESS_VOL": "red",
    "INSUFFICIENT_DATA": "gray",
}

for regime, color in colors.items():
    mask = df["vol_regime"] == regime
    plt.scatter(
        df.index[mask],
        df.loc[mask, "rolling_vol"],
        label=regime,
        color=color,
        s=12,
        alpha=0.7,
    )

plt.plot(df.index, df["rolling_vol"], color="black", alpha=0.4)
plt.title("Rolling Volatility with Regime Classification")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.tight_layout()
plt.show()

df_valid = df[df["vol_regime"] != "INSUFFICIENT_DATA"]

from arch import arch_model

print("\n==============================")
print("GARCH PERSISTENCE BY REGIME")
print("==============================")

garch_results = {}

for regime in ["LOW_VOL", "NORMAL_VOL", "STRESS_VOL"]:
    subset = df_valid[df_valid["vol_regime"] == regime]["log_return"]

    # Need enough data to fit GARCH
    if len(subset) < 100:
        print(f"{regime}: Not enough data")
        continue

    # GARCH expects percentage returns
    returns_pct = subset * 100

    model = arch_model(
        returns_pct,
        p=1,
        q=1,
        mean="Zero",
        vol="Garch",
        rescale=False
    )
# alpha modeling
    
    res = model.fit(disp="off")

    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    persistence = alpha + beta

    garch_results[regime] = persistence

    print(
        f"{regime}: "
        f"alpha={alpha:.3f}, beta={beta:.3f}, "
        f"alpha+beta={persistence:.3f}"
    )
