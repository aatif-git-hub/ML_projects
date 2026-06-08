import yfinance as yf
import pandas as pd
import joblib

model = joblib.load("hmm_model.pkl")
scaler = joblib.load("scaler.pkl")

data = yf.download(
    "^NSEI",
    period="1y"
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

scaled_features = scaler.transform(features)

states = model.predict(scaled_features)

current_regime = states[-1]

latest_close = data["Close"].iloc[-1]

print("\nLatest Data Date:")
print(data.index[-1].date())

print("\nLatest NIFTY Close:")
print(round(latest_close, 2))

print("\nLast 5 Predicted Regimes:")
print(states[-5:])

print("\nCurrent Regime:")

if current_regime == 0:

    print("Bullish Trending")

    recommendation = "BUY"

elif current_regime == 1:

    print("Neutral / Sideways")

    recommendation = "HOLD"

else:

    print("Volatile Bearish")

    recommendation = "STAY OUT"

print("\nRecommendation:")
print(recommendation)