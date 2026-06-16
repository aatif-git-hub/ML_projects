import feedparser
import joblib
import pandas as pd
import yfinance as yf

from transformers import pipeline

model = joblib.load("models/infy_hmm.pkl")
scaler = joblib.load("models/infy_scaler.pkl")

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert"
)

data = yf.download(
    "INFY.NS",
    period="2y",
    auto_adjust=True
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
    - data["Close"].shift(20)
)

data = data.dropna()

features = data[
    [
        "Return",
        "Volatility",
        "Momentum"
    ]
]

scaled_features = scaler.transform(features)

regimes = model.predict(
    scaled_features
)

current_regime = regimes[-1]

regime_names = {
    0: "Bullish",
    1: "Neutral",
    2: "Bearish"
}

regime = regime_names.get(
    current_regime,
    "Unknown"
)

rss_url = (
    "https://news.google.com/rss/search?"
    "q=Infosys&hl=en-IN&gl=IN&ceid=IN:en"
)

feed = feedparser.parse(rss_url)

headlines = []

for article in feed.entries[:25]:
    headlines.append(article.title)

results = classifier(headlines)

scores = []

for result in results:

    label = result["label"]
    confidence = result["score"]

    if label == "positive":
        score = confidence

    elif label == "negative":
        score = -confidence

    else:
        score = 0

    scores.append(score)

overall_sentiment = (
    sum(scores)
    / len(scores)
)

if overall_sentiment > 0.25:
    sentiment = "Positive"

elif overall_sentiment < -0.25:
    sentiment = "Negative"

else:
    sentiment = "Neutral"

signal_table = {

    ("Bullish", "Positive"):
        "STRONG BUY",

    ("Bullish", "Neutral"):
        "BUY",

    ("Bullish", "Negative"):
        "HOLD",

    ("Neutral", "Positive"):
        "WATCH",

    ("Neutral", "Neutral"):
        "HOLD",

    ("Neutral", "Negative"):
        "AVOID",

    ("Bearish", "Positive"):
        "HOLD",

    ("Bearish", "Neutral"):
        "SELL",

    ("Bearish", "Negative"):
        "STRONG SELL"
}

final_signal = signal_table[
    (regime, sentiment)
]

print("\n" + "=" * 50)

print("ASSET")
print("Infosys")

print("\nCURRENT REGIME")
print(regime)

print("\nOVERALL SENTIMENT SCORE")
print(round(overall_sentiment, 4))

print("\nNEWS SENTIMENT")
print(sentiment)

print("\nFINAL SIGNAL")
print(final_signal)

print("\n" + "=" * 50)

print("\nTOP 5 HEADLINES\n")

for article in headlines[:5]:
    print(article)
    print()