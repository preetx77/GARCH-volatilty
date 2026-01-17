import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
START_DATE = datetime(2022, 1, 1, tzinfo=pytz.UTC)
END_DATE = datetime.now(pytz.UTC)

# Initialize MT5
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# Fetch data
rates = mt5.copy_rates_range(
    SYMBOL,
    TIMEFRAME,
    START_DATE,
    END_DATE
)

mt5.shutdown()

if rates is None or len(rates) == 0:
    raise RuntimeError("No data fetched from MT5")

# Convert to DataFrame
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df.set_index("time", inplace=True)

# Keep only what we need
df = df[["open", "high", "low", "close", "tick_volume"]]

print(df.head())
print(df.tail())
print(df.index.inferred_freq)



