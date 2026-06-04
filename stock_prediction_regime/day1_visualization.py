import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2025-01-01"
)

print(data.head())

print(data.columns)

plt.figure(figsize=(12,6))

plt.plot(data.index, data["Close"])

plt.title("NIFTY 50 Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.grid(True)

plt.show()