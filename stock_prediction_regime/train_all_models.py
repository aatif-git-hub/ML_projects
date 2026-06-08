import yfinance as yf
import pandas as pd
import joblib
import os

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

stocks = {
    "nifty": "^NSEI",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infy": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS"
}

for name, ticker in stocks.items():

    print(f"\nTraining {name.upper()} ...")

    data = yf.download(
        ticker,
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

    model.fit(scaled_features)

    joblib.dump(
        model,
        f"models/{name}_hmm.pkl"
    )

    joblib.dump(
        scaler,
        f"models/{name}_scaler.pkl"
    )

    print(f"{name.upper()} saved!")

print("\nAll Models Trained Successfully!")