import yfinance as yf

data = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2025-01-01"
)

print(data.head())

print("\nDATA INFO")
print(data.info())

print("\nSTATISTICS")
print(data.describe())

data.to_csv("nifty50_data.csv")

print("Dataset saved successfully!")