import yfinance as yf
import pandas as pd

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

print(
    data[
        [
            "Close",
            "Return",
            "Volatility",
            "Momentum",
            "MA20",
            "MA50"
        ]
    ].head()
)

data.to_csv("nifty_features.csv")

print("Features saved successfully!")


print(data.shape)

data[
    [
        "Return",
        "Volatility",
        "Momentum",
        "MA20",
        "MA50"
    ]
].head()

print("\nFeature Statistics:\n")

print(
    data[
        [
            "Return",
            "Volatility",
            "Momentum"
        ]
    ].describe()
)