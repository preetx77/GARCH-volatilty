# GARCH Volatility Regime Analysis

This project focuses on **volatility regime analysis** using financial time series data, with a particular emphasis on **GARCH-based modeling** to understand how market risk evolves across different volatility states.

Rather than predicting price direction, the project studies **risk behavior**, **volatility clustering**, and **tail behavior** under different regimes.

---

## Project Objective

- Analyze how volatility behaves across different market regimes
- Classify markets into **Low**, **Normal**, and **Stress** volatility states
- Measure how risk, persistence, and tail losses change across regimes
- Apply **GARCH(1,1)** models within each regime to quantify volatility dynamics
- Demonstrate why a single volatility model can underestimate risk

---

## Dataset

- Instrument: **XAUUSD (Gold)**
- Data Type: Historical price data
- Frequency: Depends on data source (daily / intraday)
- Input Fields:
  - Timestamp
  - Open / High / Low / Close (OHLC)

Gold is used because it is **highly sensitive to volatility shocks** and displays clear regime transitions.

---

## Repository Structure & Explanation

### 1. Data Loading & Preprocessing

**Purpose**
- Load raw historical price data
- Clean missing or inconsistent values
- Prepare data for volatility analysis

**What this part does**
- Reads price data into pandas DataFrames
- Computes **log returns**
- Removes NaNs introduced by return calculations
- Ensures time series consistency

**Why it matters**
Volatility modeling is extremely sensitive to data quality. Clean returns are the foundation of all further analysis.

---

### 2. Rolling Volatility Computation

**Purpose**
- Measure short-term volatility behavior

**What this part does**
- Computes rolling standard deviation of log returns
- Uses a fixed rolling window (e.g., 20 periods)
- Produces a time-varying volatility series

**Why it matters**
Rolling volatility provides a **real-time view of risk buildup** and helps identify regime boundaries.

---

### 3. Volatility Regime Classification

**Purpose**
- Segment the market into distinct volatility states

**What this part does**
- Uses volatility **quantiles** to classify regimes:
  - Low Volatility Regime
  - Normal Volatility Regime
  - Stress / High Volatility Regime
- Assigns a regime label to each time period

**Why it matters**
Markets do not behave uniformly. Risk, drawdowns, and persistence differ significantly across regimes.

---

### 4. Regime-Based Return & Risk Analysis

**Purpose**
- Compare risk characteristics across regimes

**What this part does**
- Computes return distributions for each regime
- Analyzes:
  - Mean returns
  - Volatility levels
  - Tail losses (extreme negative returns)
- Highlights how stress regimes amplify downside risk

**Why it matters**
This shows why treating all periods with a single risk model leads to **systematic underestimation of risk**.

---

### 5. Volatility Persistence Analysis

**Purpose**
- Measure how long volatility shocks last

**What this part does**
- Studies autocorrelation and clustering of volatility
- Identifies persistence differences between regimes
- Shows that volatility decays slower in stress regimes

**Why it matters**
Persistence determines how long elevated risk stays in the system, directly impacting position sizing and risk control.

---

### 6. GARCH(1,1) Modeling by Regime

**Purpose**
- Quantify volatility dynamics mathematically

**What this part does**
- Fits separate **GARCH(1,1)** models for each volatility regime
- Extracts:
  - Alpha (reaction to new shocks)
  - Beta (volatility persistence)
- Compares parameter stability across regimes

**Why it matters**
GARCH parameters vary significantly by regime, proving that **one global GARCH model is structurally flawed**.

---

### 7. Results & Key Observations

**Main findings**
- Stress regimes exhibit:
  - Much higher tail risk
  - Stronger volatility clustering
  - Slower mean reversion
- Low-volatility regimes show:
  - Faster decay of shocks
  - Lower downside risk
- Volatility is **regime-dependent**, not stationary

---

## Technologies Used

- Python
- pandas
- numpy
- matplotlib / seaborn
- arch (for GARCH modeling)
- scipy / statsmodels

---

## How to Run the Project

1. Clone the repository
git clone https://github.com/preetx77/GARCH-volatilty.git

Install dependencies
pip install -r requirements.txt
Run analysis scripts in sequence:

Data preprocessing
Volatility computation
Regime classification
GARCH modeling ```
