import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

scaled_features = scaler.fit_transform(features)

model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=100,
    random_state=42
)

model.fit(scaled_features)

states = model.predict(scaled_features)

data["Regime"] = states

data["Signal"] = 0

data.loc[
    data["Regime"].isin([0, 1]),
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

market_return = (
    data["Market_Cumulative"].iloc[-1] - 1
) * 100

strategy_return = (
    data["Strategy_Cumulative"].iloc[-1] - 1
) * 100

print("\nMarket Return:")
print(round(market_return, 2), "%")

print("\nStrategy Return:")
print(round(strategy_return, 2), "%")

years = len(data) / 252

cagr_market = (
    data["Market_Cumulative"].iloc[-1]
    ** (1 / years)
    - 1
)

cagr_strategy = (
    data["Strategy_Cumulative"].iloc[-1]
    ** (1 / years)
    - 1
)

print("\nCAGR")

print(
    "Market:",
    round(cagr_market * 100, 2),
    "%"
)

print(
    "Strategy:",
    round(cagr_strategy * 100, 2),
    "%"
)

market_vol = (
    data["Market_Return"]
    .std()
    * np.sqrt(252)
)

strategy_vol = (
    data["Strategy_Return"]
    .std()
    * np.sqrt(252)
)

print("\nAnnual Volatility")

print(
    "Market:",
    round(market_vol * 100, 2),
    "%"
)

print(
    "Strategy:",
    round(strategy_vol * 100, 2),
    "%"
)

sharpe_market = (
    cagr_market
    / market_vol
)

sharpe_strategy = (
    cagr_strategy
    / strategy_vol
)

print("\nSharpe Ratio")

print(
    "Market:",
    round(sharpe_market, 2)
)

print(
    "Strategy:",
    round(sharpe_strategy, 2)
)

def max_drawdown(cumulative):

    rolling_max = cumulative.cummax()

    drawdown = (
        cumulative
        - rolling_max
    ) / rolling_max

    return drawdown.min()

mdd_market = max_drawdown(
    data["Market_Cumulative"]
)

mdd_strategy = max_drawdown(
    data["Strategy_Cumulative"]
)

print("\nMaximum Drawdown")

print(
    "Market:",
    round(
        mdd_market * 100,
        2
    ),
    "%"
)

print(
    "Strategy:",
    round(
        mdd_strategy * 100,
        2
    ),
    "%"
)

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