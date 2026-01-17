import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================
# CONFIG
# ============================

CSV_PATH = "data.csv"
INITIAL_CAPITAL = 100000

# Volatility parameters
VOL_WINDOW = 20
VOL_LONG_WINDOW = 60
TARGET_VOL = 0.12  # 12% target (more realistic for active strategy)
MAX_LEVERAGE = 2.0
MIN_POSITION = 0.1  # Never go below 10% exposure

# Transaction costs
TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 2

# Train/test split
TRAIN_SPLIT = 0.70

# Advanced regime detection
REGIME_LOOKBACK = 252  # 1 year for regime calibration
STRESS_THRESHOLD = 1.5  # Volatility multiplier for stress
RECOVERY_HALFLIFE = 10  # Days to recover 50% after stress

# Multi-factor signal parameters
MOMENTUM_FAST = 20
MOMENTUM_SLOW = 60
MOMENTUM_LONG = 120

TREND_FAST = 10
TREND_MEDIUM = 30
TREND_SLOW = 100

RSI_WINDOW = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Mean reversion
ZSCORE_WINDOW = 20
ZSCORE_ENTRY = 2.0

# Carry/roll yield (for futures/ETFs with contango/backwardation)
CARRY_WINDOW = 20

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
    raise ValueError("No date/time column found")

if "close" in df.columns:
    price_col = "close"
elif "price" in df.columns:
    price_col = "price"
else:
    raise ValueError("No close/price column found")

df[price_col] = df[price_col].astype(str).str.replace(",", "", regex=False)
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df.rename(columns={price_col: "close"}, inplace=True)

df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]
df["close"] = df["close"].ffill(limit=2)
df.dropna(subset=["close"], inplace=True)

print(f"Total rows: {len(df)}")

# ============================
# RETURNS & VOLATILITY
# ============================

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["simple_return"] = df["close"].pct_change()

# Short and long-term realized volatility
df["vol_short"] = df["log_return"].rolling(window=VOL_WINDOW).std() * np.sqrt(252)
df["vol_long"] = df["log_return"].rolling(window=VOL_LONG_WINDOW).std() * np.sqrt(252)
df["vol_ratio"] = df["vol_short"] / df["vol_long"]

# Volatility z-score (how unusual is current vol?)
df["vol_zscore"] = (
    (df["vol_short"] - df["vol_short"].rolling(REGIME_LOOKBACK).mean()) / 
    df["vol_short"].rolling(REGIME_LOOKBACK).std()
)

df.dropna(subset=["log_return", "vol_short", "vol_long"], inplace=True)

# ============================
# ADVANCED REGIME DETECTION
# ============================

def adaptive_regime(row):
    """Dynamic regime based on vol z-score and recent vol changes"""
    zscore = row["vol_zscore"]
    ratio = row["vol_ratio"]
    
    # Extreme stress: vol exploding AND already elevated
    if zscore > 2.5 or (zscore > 1.5 and ratio > 1.3):
        return "EXTREME_STRESS"
    # Elevated stress: vol rising significantly
    elif zscore > 1.0 or ratio > 1.2:
        return "ELEVATED_STRESS"
    # Normal: vol in historical range
    elif zscore > -0.5:
        return "NORMAL"
    # Low vol: compressed volatility (often precedes moves)
    else:
        return "LOW_VOL"

df["regime"] = df.apply(adaptive_regime, axis=1)

# Regime persistence (for recovery logic)
regime_changes = (df["regime"] != df["regime"].shift(1)).astype(int)
df["regime_duration"] = regime_changes.groupby(
    regime_changes.cumsum()
).cumcount() + 1

print("\nRegime distribution:")
print(df["regime"].value_counts())

# ============================
# MULTI-FACTOR ALPHA SIGNALS
# ============================

# 1. MOMENTUM CASCADE (multiple timeframes)
df["mom_fast"] = df["close"] / df["close"].shift(MOMENTUM_FAST) - 1
df["mom_slow"] = df["close"] / df["close"].shift(MOMENTUM_SLOW) - 1
df["mom_long"] = df["close"] / df["close"].shift(MOMENTUM_LONG) - 1

# Weighted momentum score
df["momentum_score"] = (
    0.5 * np.sign(df["mom_fast"]) + 
    0.3 * np.sign(df["mom_slow"]) + 
    0.2 * np.sign(df["mom_long"])
)

# 2. TREND STRENGTH (multi-timeframe MA)
df["ma_fast"] = df["close"].rolling(TREND_FAST).mean()
df["ma_medium"] = df["close"].rolling(TREND_MEDIUM).mean()
df["ma_slow"] = df["close"].rolling(TREND_SLOW).mean()

# Trend alignment score
trend_alignment = (
    (df["ma_fast"] > df["ma_medium"]).astype(int) +
    (df["ma_medium"] > df["ma_slow"]).astype(int) +
    (df["close"] > df["ma_fast"]).astype(int)
)
df["trend_score"] = (trend_alignment - 1.5) / 1.5  # Normalize to [-1, 1]

# 3. MEAN REVERSION (for range-bound markets)
df["zscore"] = (
    (df["close"] - df["close"].rolling(ZSCORE_WINDOW).mean()) / 
    df["close"].rolling(ZSCORE_WINDOW).std()
)
df["mean_reversion_score"] = -np.tanh(df["zscore"] / 2)  # Fade extremes

# 4. RSI (momentum confirmation)
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(RSI_WINDOW).mean()
loss = -delta.where(delta < 0, 0).rolling(RSI_WINDOW).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))
df["rsi_score"] = (df["rsi"] - 50) / 50  # Normalize to [-1, 1]

# 5. VOLATILITY REGIME SIGNAL
# In low vol: fade moves (mean reversion)
# In high vol: follow momentum (trend)
df["vol_regime_signal"] = np.where(
    df["regime"] == "LOW_VOL",
    df["mean_reversion_score"],
    df["momentum_score"]
)

# ============================
# COMBINED SIGNAL
# ============================

def adaptive_signal(row):
    """Regime-aware signal combination"""
    regime = row["regime"]
    
    if regime == "EXTREME_STRESS":
        # Pure defense: only strong trends
        return 0.3 * row["trend_score"]
    
    elif regime == "ELEVATED_STRESS":
        # Cautious trending
        signal = 0.5 * row["momentum_score"] + 0.5 * row["trend_score"]
        return 0.6 * signal
    
    elif regime == "NORMAL":
        # Balanced multi-factor
        signal = (
            0.35 * row["momentum_score"] +
            0.35 * row["trend_score"] +
            0.15 * row["mean_reversion_score"] +
            0.15 * row["rsi_score"]
        )
        return signal
    
    else:  # LOW_VOL
        # Aggressive: exploit low vol with mean reversion + momentum
        signal = (
            0.4 * row["momentum_score"] +
            0.3 * row["mean_reversion_score"] +
            0.3 * row["trend_score"]
        )
        return signal * 1.2  # Scale up in low vol

df["raw_signal"] = df.apply(adaptive_signal, axis=1)

# Signal smoothing (reduce whipsaw)
df["signal"] = df["raw_signal"].rolling(window=3, center=False).mean()
df["signal"] = np.clip(df["signal"], -1, 1)

# ============================
# TRAIN/TEST SPLIT
# ============================

split_idx = int(len(df) * TRAIN_SPLIT)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"\nTrain: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} rows)")
print(f"Test: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} rows)")

# ============================
# ADAPTIVE POSITION SIZING
# ============================

def advanced_position_size(row, prev_position):
    """
    Adaptive position sizing with:
    - Vol targeting
    - Regime-based scaling
    - Smooth recovery from stress
    - Never fully flat
    """
    signal = row["signal"]
    vol = row["vol_short"]
    regime = row["regime"]
    regime_duration = row["regime_duration"]
    
    # Base size from vol targeting
    if vol > 0:
        base_size = TARGET_VOL / vol
    else:
        base_size = 1.0
    
    base_size = min(base_size, MAX_LEVERAGE)
    
    # Regime-based risk multiplier with SMOOTH RECOVERY
    if regime == "EXTREME_STRESS":
        # Start at 25%, recover exponentially
        recovery_factor = 1 - np.exp(-regime_duration / RECOVERY_HALFLIFE)
        risk_mult = 0.25 + 0.25 * recovery_factor
    
    elif regime == "ELEVATED_STRESS":
        # Start at 50%, faster recovery
        recovery_factor = 1 - np.exp(-regime_duration / (RECOVERY_HALFLIFE / 2))
        risk_mult = 0.5 + 0.3 * recovery_factor
    
    elif regime == "NORMAL":
        risk_mult = 1.0
    
    else:  # LOW_VOL
        # Scale up in low vol (but capped)
        risk_mult = 1.2
    
    # Apply signal direction
    target_position = base_size * risk_mult * signal
    
    # Enforce minimum exposure (never fully flat)
    if abs(target_position) < MIN_POSITION and signal != 0:
        target_position = MIN_POSITION * np.sign(signal)
    
    # Smooth position changes (reduce turnover)
    smooth_position = 0.7 * target_position + 0.3 * prev_position
    
    return smooth_position

# Apply position sizing
for dataset in [train_df, test_df]:
    positions = [0]
    for idx in range(1, len(dataset)):
        row = dataset.iloc[idx]
        prev_pos = positions[-1]
        new_pos = advanced_position_size(row, prev_pos)
        positions.append(new_pos)
    
    dataset["position"] = positions

# ============================
# BACKTEST ENGINE
# ============================

def advanced_backtest(data, name):
    data = data.copy()
    
    # Transaction costs
    data["position_change"] = data["position"].diff().abs()
    data["transaction_cost"] = (
        data["position_change"] * (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000
    )
    
    # Strategy returns
    data["gross_return"] = data["position"].shift(1) * data["log_return"]
    data["net_return"] = data["gross_return"] - data["transaction_cost"]
    
    # Equity curves
    data["strategy_equity"] = INITIAL_CAPITAL * np.exp(data["net_return"].cumsum())
    data["bh_equity"] = INITIAL_CAPITAL * (1 + data["simple_return"]).cumprod()
    
    # Performance metrics
    total_days = (data.index[-1] - data.index[0]).days
    years = total_days / 365.25
    
    # Strategy metrics
    total_return = (data["strategy_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    ann_return = ((data["strategy_equity"].iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    ann_vol = data["net_return"].std() * np.sqrt(252) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Downside deviation (for Sortino)
    negative_returns = data["net_return"][data["net_return"] < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0
    
    # Drawdown
    cummax = data["strategy_equity"].cummax()
    drawdown = (data["strategy_equity"] - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate
    wins = (data["net_return"] > 0).sum()
    total_trades = len(data[data["net_return"] != 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Average win/loss
    avg_win = data[data["net_return"] > 0]["net_return"].mean() if wins > 0 else 0
    avg_loss = data[data["net_return"] < 0]["net_return"].mean() if wins < total_trades else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Buy & Hold
    bh_return = (data["bh_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    bh_ann_return = ((data["bh_equity"].iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    bh_vol = data["simple_return"].std() * np.sqrt(252) * 100
    bh_sharpe = bh_ann_return / bh_vol if bh_vol > 0 else 0
    
    bh_cummax = data["bh_equity"].cummax()
    bh_drawdown = (data["bh_equity"] - bh_cummax) / bh_cummax
    bh_max_dd = bh_drawdown.min() * 100
    
    # Statistical significance (t-test vs zero)
    t_stat, p_value = stats.ttest_1samp(data["net_return"].dropna(), 0)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()} ({years:.2f} years)")
    
    print(f"\n{' STRATEGY METRICS ':-^60}")
    print(f"Total Return:        {total_return:>8.2f}%")
    print(f"Annualized Return:   {ann_return:>8.2f}%")
    print(f"Annualized Vol:      {ann_vol:>8.2f}%")
    print(f"Sharpe Ratio:        {sharpe:>8.3f}")
    print(f"Sortino Ratio:       {sortino:>8.3f}")
    print(f"Max Drawdown:        {max_dd:>8.2f}%")
    print(f"Calmar Ratio:        {calmar:>8.3f}")
    print(f"Win Rate:            {win_rate:>8.2f}%")
    print(f"Profit Factor:       {profit_factor:>8.3f}")
    print(f"Avg Turnover/Day:    {data['position_change'].mean():>8.4f}")
    
    print(f"\n{' BUY & HOLD ':-^60}")
    print(f"Annualized Return:   {bh_ann_return:>8.2f}%")
    print(f"Annualized Vol:      {bh_vol:>8.2f}%")
    print(f"Sharpe Ratio:        {bh_sharpe:>8.3f}")
    print(f"Max Drawdown:        {bh_max_dd:>8.2f}%")
    
    print(f"\n{' ALPHA GENERATION ':-^60}")
    print(f"Excess Return:       {ann_return - bh_ann_return:>8.2f}%")
    print(f"Information Ratio:   {(ann_return - bh_ann_return) / (data['net_return'] - data['simple_return']).std() / np.sqrt(252) / 100:>8.3f}")
    print(f"Sharpe Improvement:  {sharpe - bh_sharpe:>8.3f}")
    print(f"Statistical Sig:     p={p_value:.4f} {'✓ SIGNIFICANT' if p_value < 0.05 else '✗ NOT SIGNIFICANT'}")
    
    return data

train_results = advanced_backtest(train_df, "TRAIN SET")
test_results = advanced_backtest(test_df, "TEST SET (OUT-OF-SAMPLE)")

# ============================
# REGIME PERFORMANCE BREAKDOWN
# ============================

print(f"\n{'='*60}")
print("PERFORMANCE BY REGIME (Test Set)")
print(f"{'='*60}")

for regime in ["EXTREME_STRESS", "ELEVATED_STRESS", "NORMAL", "LOW_VOL"]:
    regime_data = test_results[test_results["regime"] == regime]
    if len(regime_data) > 0:
        regime_ret = regime_data["net_return"].mean() * 252 * 100
        regime_vol = regime_data["net_return"].std() * np.sqrt(252) * 100
        regime_sharpe = regime_ret / regime_vol if regime_vol > 0 else 0
        regime_days = len(regime_data)
        
        print(f"\n{regime}:")
        print(f"  Days:             {regime_days}")
        print(f"  Ann. Return:      {regime_ret:.2f}%")
        print(f"  Ann. Vol:         {regime_vol:.2f}%")
        print(f"  Sharpe:           {regime_sharpe:.3f}")

# ============================
# VISUALIZATION
# ============================

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

# 1. Main equity curves
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(train_results.index, train_results["strategy_equity"], 
         label="Strategy (Train)", color="blue", linewidth=2)
ax1.plot(test_results.index, test_results["strategy_equity"], 
         label="Strategy (Test)", color="darkblue", linewidth=2, linestyle="--")
ax1.plot(df.index, pd.concat([train_results["bh_equity"], test_results["bh_equity"]]), 
         label="Buy & Hold", color="gray", alpha=0.6, linewidth=1.5)
ax1.axvline(test_results.index[0], color="red", linestyle=":", alpha=0.7, 
            linewidth=2, label="Train/Test Split")
ax1.set_title("Adaptive Multi-Factor Strategy vs Buy & Hold", fontsize=14, fontweight="bold")
ax1.set_ylabel("Equity ($)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# 2. Drawdowns
ax2 = fig.add_subplot(gs[1, :])
full_results = pd.concat([train_results, test_results])
cummax_strat = full_results["strategy_equity"].cummax()
dd_strat = (full_results["strategy_equity"] - cummax_strat) / cummax_strat * 100
cummax_bh = pd.concat([train_results["bh_equity"], test_results["bh_equity"]]).cummax()
dd_bh = (pd.concat([train_results["bh_equity"], test_results["bh_equity"]]) - cummax_bh) / cummax_bh * 100

ax2.fill_between(full_results.index, dd_strat, 0, alpha=0.4, color="blue", label="Strategy")
ax2.fill_between(full_results.index, dd_bh, 0, alpha=0.4, color="gray", label="Buy & Hold")
ax2.axvline(test_results.index[0], color="red", linestyle=":", alpha=0.7, linewidth=2)
ax2.set_title("Drawdown Comparison", fontsize=12, fontweight="bold")
ax2.set_ylabel("Drawdown (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Regime coloring
ax3 = fig.add_subplot(gs[2, :])
regime_colors = {
    "EXTREME_STRESS": "darkred",
    "ELEVATED_STRESS": "orange",
    "NORMAL": "green",
    "LOW_VOL": "blue"
}
for regime, color in regime_colors.items():
    sub = full_results[full_results["regime"] == regime]
    ax3.scatter(sub.index, sub["vol_short"], s=8, color=color, label=regime, alpha=0.6)
ax3.axvline(test_results.index[0], color="red", linestyle=":", alpha=0.7, linewidth=2)
ax3.set_title("Volatility Regimes (Adaptive)", fontsize=12, fontweight="bold")
ax3.set_ylabel("Realized Vol")
ax3.legend(loc="upper left", ncol=4)
ax3.grid(True, alpha=0.3)

# 4. Position sizing over time
ax4 = fig.add_subplot(gs[3, :])
ax4.plot(full_results.index, full_results["position"], color="purple", linewidth=1, alpha=0.8)
ax4.axvline(test_results.index[0], color="red", linestyle=":", alpha=0.7, linewidth=2)
ax4.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax4.axhline(MIN_POSITION, color="green", linestyle="--", linewidth=0.5, alpha=0.5, label=f"Min Position ({MIN_POSITION})")
ax4.axhline(-MIN_POSITION, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
ax4.set_title("Dynamic Position Sizing (Signal × Vol Target × Regime Adjustment)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Position Size")
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Rolling Sharpe (left)
ax5 = fig.add_subplot(gs[4, 0])
rolling_sharpe = (
    full_results["net_return"].rolling(window=60).mean() / 
    full_results["net_return"].rolling(window=60).std() * np.sqrt(252)
)
ax5.plot(full_results.index, rolling_sharpe, color="teal", linewidth=1.5)
ax5.axvline(test_results.index[0], color="red", linestyle=":", alpha=0.7, linewidth=2)
ax5.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax5.set_title("Rolling 60-Day Sharpe Ratio", fontsize=11, fontweight="bold")
ax5.set_ylabel("Sharpe")
ax5.grid(True, alpha=0.3)

# 6. Signal distribution (right)
ax6 = fig.add_subplot(gs[4, 1])
ax6.hist(test_results["signal"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
ax6.set_title("Signal Distribution (Test Set)", fontsize=11, fontweight="bold")
ax6.set_xlabel("Signal Value")
ax6.set_ylabel("Frequency")
ax6.grid(True, alpha=0.3, axis="y")

plt.suptitle("Advanced Multi-Factor Regime-Adaptive Trading System", 
             fontsize=16, fontweight="bold", y=0.995)

plt.show()

print("\n" + "="*60)
print("ADVANCED ANALYSIS COMPLETE")
print("="*60)