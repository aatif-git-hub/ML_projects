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
    data[
        [
            "Return",
            "Volatility",
            "Momentum",
            "Regime"
        ]
    ].head(20)
)

print(
    data["Regime"].value_counts()
)

print("\nTransition Matrix:\n")
print(model.transmat_)