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

print(
    data.groupby("Regime")[
        [
            "Return",
            "Volatility",
            "Momentum"
        ]
    ].mean()
)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

for regime in range(3):

    subset = data[data["Regime"] == regime]

    plt.scatter(
        subset.index,
        subset["Close"],
        s=5,
        label=f"Regime {regime}"
    )

plt.title("NIFTY Market Regimes Detected by HMM")

plt.xlabel("Date")
plt.ylabel("NIFTY Close")

plt.legend()

plt.show()

print("\nRegime Counts:\n")

print(
    data["Regime"]
    .value_counts()
)

plt.figure(figsize=(12,6))

for regime in range(3):

    subset = data[data["Regime"] == regime]

    plt.scatter(
        subset.index,
        subset["Volatility"],
        s=5,
        label=f"Regime {regime}"
    )

plt.title("Volatility Across Regimes")

plt.legend()

plt.show()

print("\nRegime Summary")

print(
    data.groupby("Regime")[
        [
            "Return",
            "Volatility",
            "Momentum"
        ]
    ].agg(
        ["mean", "std"]
    )
)

data.to_csv(
    "nifty_regimes.csv"
)

print(
    "Regime dataset saved!"
)

