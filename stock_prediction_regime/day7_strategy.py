import yfinance as yf
import pandas as pd
import numpy as np

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

data = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2025-01-01"
)

data.columns = data.columns.get_level_values(0)

data["Return"] = data["Close"].pct_change()

data["Volatility"] = (
    data["Return"]
    .rolling(20)
    .std()
)

data["Momentum"] = (
    data["Close"]
    - data["Close"].shift(10)
)

data["MA20"] = (
    data["Close"]
    .rolling(20)
    .mean()
)

data["MA50"] = (
    data["Close"]
    .rolling(50)
    .mean()
)

data.dropna(inplace=True)

features = data[
    [
        "Return",
        "Volatility",
        "Momentum"
    ]
]


scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    features
)

model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=100,
    random_state=42
)

model.fit(
    scaled_features
)

states = model.predict(
    scaled_features
)

data["Regime"] = states

data["Signal"] = 0

data.loc[
    data["Regime"].isin([0,1]),
    "Signal"
] = 1

data["Market_Return"] = data["Close"].pct_change()

data["Strategy_Return"] = (
    data["Signal"].shift(1)
    * data["Market_Return"]
)

data["Market_Cumulative"] = (
    1 + data["Market_Return"]
).cumprod()

data["Strategy_Cumulative"] = (
    1 + data["Strategy_Return"]
).cumprod()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(
    data.index,
    data["Market_Cumulative"],
    label="Buy & Hold"
)

plt.plot(
    data.index,
    data["Strategy_Cumulative"],
    label="HMM Strategy"
)

plt.title("Strategy vs Buy & Hold")

plt.legend()

plt.show()

market_return = (
    data["Market_Cumulative"].iloc[-1] - 1
) * 100

strategy_return = (
    data["Strategy_Cumulative"].iloc[-1] - 1
) * 100

print("\nMarket Return:")
print(round(market_return,2), "%")

print("\nStrategy Return:")
print(round(strategy_return,2), "%")

print("\nDays Invested:")

print(
    data["Signal"]
    .value_counts()
)