"""Generate synthetic cryptocurrency OHLCV data for pipeline testing."""

import numpy as np
import pandas as pd

np.random.seed(42)

n = 10000
base_price = 30000.0
returns = np.random.normal(loc=0.0001, scale=0.02, size=n)
close = base_price * np.cumprod(1 + returns)

high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n)
volume = np.random.uniform(100, 10000, n).round(2)

df = pd.DataFrame({
    "open": open_price.round(2),
    "high": high.round(2),
    "low": low.round(2),
    "close": close.round(2),
    "volume": volume,
})

df.to_csv("data.csv", index=False)
print(f"Generated data.csv with {len(df)} rows")
