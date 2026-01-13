import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# ============================
# CONFIG
# ============================

CSV_PATH = "data.csv"
INITIAL_CAPITAL = 100000

# GARCH config
GARCH_P = 1
GARCH_Q = 1

# Risk config
TARGET_VOL = 0.08
MAX_LEVERAGE = 1.0

# Stress logic
CONFIRM_BARS = 3
STEP_MULTIPLIERS = {
    0: 1.0,
    1: 0.6,
    2: 0.3,
    3: 0.0
}

# Costs
TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 2

# Train/Test
TRAIN_SPLIT = 0.70

# Signal parameters
MOMENTUM_WINDOW = 60
TREND_FAST = 20
TREND_SLOW = 50
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# ============================
# LOAD & CLEAN DATA
# ============================

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
elif "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
else:
    raise ValueError("No datetime column found")

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
df["close"] = df["close"].ffill(limit=2)
df.dropna(subset=["close"], inplace=True)

print(f"Rows after cleaning: {len(df)}")

# ============================
# RETURNS
# ============================

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["simple_return"] = df["close"].pct_change()
df.dropna(inplace=True)

# ============================
# GARCH VOLATILITY (CORE CHANGE)
# ============================

returns_pct = df["log_return"] * 100

garch = arch_model(
    returns_pct,
    p=GARCH_P,
    q=GARCH_Q,
    mean="Zero",
    vol="Garch",
    rescale=False
)

res = garch.fit(disp="off")
df["garch_vol"] = res.conditional_volatility / 100  # back to decimal

# ============================
# SIGNAL LAYER
# ============================

# Momentum
df["momentum"] = df["close"] / df["close"].shift(MOMENTUM_WINDOW) - 1

# Trend
df["ma_fast"] = df["close"].rolling(TREND_FAST).mean()
df["ma_slow"] = df["close"].rolling(TREND_SLOW).mean()
df["trend_signal"] = np.where(df["ma_fast"] > df["ma_slow"], 1, -1)

# RSI
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(RSI_WINDOW).mean()
loss = -delta.where(delta < 0, 0).rolling(RSI_WINDOW).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

df["rsi_signal"] = np.where(
    df["rsi"] < RSI_OVERSOLD, 1,
    np.where(df["rsi"] > RSI_OVERBOUGHT, -1, 0)
)

df["signal"] = df["trend_signal"] * (1 + df["momentum"].fillna(0))
df["signal"] = np.clip(df["signal"], -1, 1)

df.loc[(df["signal"] > 0) & (df["rsi"] > RSI_OVERBOUGHT), "signal"] = 0
df.loc[(df["signal"] < 0) & (df["rsi"] < RSI_OVERSOLD), "signal"] = 0

df.dropna(subset=["signal", "garch_vol"], inplace=True)

# ============================
# TRAIN / TEST SPLIT
# ============================

split_idx = int(len(df) * TRAIN_SPLIT)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# ============================
# REGIME DETECTION (GARCH VOL)
# ============================

low_q = train_df["garch_vol"].quantile(0.33)
high_q = train_df["garch_vol"].quantile(0.66)

def classify_regime(vol):
    if vol < low_q:
        return "LOW_VOL"
    elif vol < high_q:
        return "NORMAL_VOL"
    else:
        return "STRESS_VOL"

def apply_regime_logic(data):
    data["vol_regime"] = data["garch_vol"].apply(classify_regime)

    persistence = []
    count = 0
    for r in data["vol_regime"]:
        if r == "STRESS_VOL":
            count += 1
        else:
            count = 0
        persistence.append(count)

    data["stress_persistence"] = persistence

    def step_mult(p):
        if p >= CONFIRM_BARS:
            return STEP_MULTIPLIERS[3]
        return STEP_MULTIPLIERS.get(p, STEP_MULTIPLIERS[3])

    data["risk_multiplier"] = data["stress_persistence"].apply(step_mult)
    return data

train_df = apply_regime_logic(train_df)
test_df = apply_regime_logic(test_df)

# ============================
# POSITION SIZING (FINAL)
# ============================

def position_size(signal, vol, mult):
    if mult == 0.0:
        return 0.0
    size = (TARGET_VOL / vol) * mult
    size = min(size, MAX_LEVERAGE)
    return size * signal

for d in [train_df, test_df]:
    d["position"] = d.apply(
        lambda x: position_size(x["signal"], x["garch_vol"], x["risk_multiplier"]),
        axis=1
    )

# ============================
# BACKTEST WITH COSTS
# ============================

def run_backtest(data):
    data = data.copy()
    data["pos_change"] = data["position"].diff().abs()
    data["cost"] = data["pos_change"] * (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000

    data["gross_return"] = data["position"].shift(1) * data["log_return"]
    data["net_return"] = data["gross_return"] - data["cost"]

    data["equity"] = INITIAL_CAPITAL * np.exp(data["net_return"].cumsum())
    return data

train_res = run_backtest(train_df)
test_res = run_backtest(test_df)
full = pd.concat([train_res, test_res])

# ============================
# SUMMARY
# ============================

def summarize(data, name):
    years = (data.index[-1] - data.index[0]).days / 365.25
    ann_ret = ((data["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    ann_vol = data["net_return"].std() * np.sqrt(252) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    dd = (data["equity"] / data["equity"].cummax() - 1) * 100

    print(f"\n{name}")
    print(f"Annual Return: {ann_ret:.2f}%")
    print(f"Annual Vol: {ann_vol:.2f}%")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Max Drawdown: {dd.min():.2f}%")

summarize(train_res, "TRAIN")
summarize(test_res, "TEST")

# ============================
# PLOTS
# ============================

plt.figure(figsize=(14, 5))
plt.plot(full.index, full["equity"], label="GARCH Stress Survival Equity")
plt.title("GARCH-Based Regime-Aware Stress Survival Engine")
plt.legend()
plt.grid(True)
plt.show()
