import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

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

plt.figure(figsize=(12,6))

plt.plot(data.index, data["Close"])

plt.title("NIFTY 50 Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(True)

plt.show()

plt.figure(figsize=(10,6))

plt.hist(
    data["Return"],
    bins=50
)

plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(12,6))

plt.plot(
    data.index,
    data["Volatility"]
)

plt.title("Market Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")

plt.grid(True)

plt.show()

plt.figure(figsize=(12,6))

plt.plot(
    data.index,
    data["MA20"],
    label="MA20"
)

plt.plot(
    data.index,
    data["MA50"],
    label="MA50"
)

plt.legend()

plt.title("MA20 vs MA50")

plt.show()

plt.figure(figsize=(12,6))

plt.plot(
    data.index,
    data["Momentum"]
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Momentum")

plt.show()

print(
    data[
        [
            "Return",
            "Volatility",
            "Momentum",
            "MA20",
            "MA50"
        ]
    ].corr()
)

import seaborn as sns
import matplotlib.pyplot as plt

corr = data[
    [
        "Return",
        "Volatility",
        "Momentum",
        "MA20",
        "MA50"
    ]
].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True
)

plt.title("Feature Correlation Matrix")

plt.show()