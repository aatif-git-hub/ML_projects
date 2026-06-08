import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AI Market Regime Dashboard",
    layout="wide"
)

stock_options = {
    "NIFTY 50": {
        "ticker": "^NSEI",
        "model": "models/nifty_hmm.pkl",
        "scaler": "models/nifty_scaler.pkl"
    },

    "Reliance": {
        "ticker": "RELIANCE.NS",
        "model": "models/reliance_hmm.pkl",
        "scaler": "models/reliance_scaler.pkl"
    },

    "TCS": {
        "ticker": "TCS.NS",
        "model": "models/tcs_hmm.pkl",
        "scaler": "models/tcs_scaler.pkl"
    },

    "Infosys": {
        "ticker": "INFY.NS",
        "model": "models/infy_hmm.pkl",
        "scaler": "models/infy_scaler.pkl"
    },

    "HDFC Bank": {
        "ticker": "HDFCBANK.NS",
        "model": "models/hdfc_hmm.pkl",
        "scaler": "models/hdfc_scaler.pkl"
    },

    "ICICI Bank": {
        "ticker": "ICICIBANK.NS",
        "model": "models/icici_hmm.pkl",
        "scaler": "models/icici_scaler.pkl"
    }
}

st.sidebar.title("Settings")

selected_stock = st.sidebar.selectbox(
    "Choose Asset",
    list(stock_options.keys())
)

ticker = stock_options[selected_stock]["ticker"]

model_path = stock_options[selected_stock]["model"]

scaler_path = stock_options[selected_stock]["scaler"]

model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

st.sidebar.markdown("---")

st.sidebar.header("Project Information")

st.sidebar.write("""
Hidden Markov Model

Features:
- Return
- Volatility
- Momentum

Outputs:
- Bullish
- Neutral
- Bearish
""")

st.title(
    f"📈 AI Market Regime Dashboard - {selected_stock}"
)

data = yf.download(
    ticker,
    period="10y"
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

data["Regime"] = states

current_regime = states[-1]

latest_close = float(
    data["Close"].iloc[-1]
)

if current_regime == 0:

    regime_name = "🟢 Bullish"

    recommendation = "BUY"

elif current_regime == 1:

    regime_name = "🟡 Neutral"

    recommendation = "HOLD"

else:

    regime_name = "🔴 Bearish"

    recommendation = "STAY OUT"

data["Signal"] = 0

data.loc[
    data["Regime"].isin([0, 1]),
    "Signal"
] = 1

data["Market_Return"] = (
    data["Close"]
    .pct_change()
)

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

years = len(data) / 252

cagr = (
    data["Strategy_Cumulative"].iloc[-1]
    ** (1 / years)
    - 1
)

volatility = (
    data["Strategy_Return"]
    .std()
    * np.sqrt(252)
)

sharpe_ratio = cagr / volatility

def max_drawdown(cumulative):

    rolling_max = cumulative.cummax()

    drawdown = (
        cumulative
        - rolling_max
    ) / rolling_max

    return drawdown.min()

mdd = max_drawdown(
    data["Strategy_Cumulative"]
)

regime_labels = {
    0: "Bullish",
    1: "Neutral",
    2: "Bearish"
}

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Latest Close",
        round(latest_close, 2)
    )

with col2:

    st.metric(
        "Current Regime",
        regime_name
    )

with col3:

    st.metric(
        "Recommendation",
        recommendation
    )

st.divider()

m1, m2 = st.columns(2)

with m1:

    st.metric(
        "Sharpe Ratio",
        round(sharpe_ratio, 2)
    )

with m2:

    st.metric(
        "Maximum Drawdown",
        f"{round(mdd * 100,2)}%"
    )

st.divider()

st.subheader("📊 Price Chart")

st.line_chart(
    data["Close"]
)

st.subheader("📈 Buy & Hold vs HMM Strategy")

comparison = pd.DataFrame({
    "Buy & Hold":
        data["Market_Cumulative"],

    "HMM Strategy":
        data["Strategy_Cumulative"]
})

st.line_chart(comparison)

st.subheader("📋 Regime Distribution")

regime_counts = (
    data["Regime"]
    .value_counts()
    .sort_index()
)

regime_chart = pd.DataFrame({
    "Regime":
        [regime_labels[i]
         for i in regime_counts.index],

    "Days":
        regime_counts.values
})

regime_chart = regime_chart.set_index(
    "Regime"
)

st.bar_chart(regime_chart)

st.subheader("📊 Market Statistics")

s1, s2, s3 = st.columns(3)

with s1:

    st.metric(
        "Average Return",
        f"{round(data['Return'].mean()*100,2)}%"
    )

with s2:

    st.metric(
        "Average Volatility",
        f"{round(data['Volatility'].mean()*100,2)}%"
    )

with s3:

    st.metric(
        "Average Momentum",
        round(
            data["Momentum"].mean(),
            2
        )
    )

st.subheader(
    "🕒 Last 10 Predicted Regimes"
)

recent = pd.DataFrame({
    "Date":
        data.index[-10:],

    "Regime":
        [
            regime_labels[i]
            for i in states[-10:]
        ]
})

st.dataframe(
    recent,
    use_container_width=True
)

csv = data.to_csv().encode(
    "utf-8"
)

st.subheader("📌 Today's Market Opportunities")

summary = []

for stock_name, info in stock_options.items():

    temp_model = joblib.load(info["model"])
    temp_scaler = joblib.load(info["scaler"])

    temp = yf.download(
        info["ticker"],
        period="6mo"
    )

    temp.columns = temp.columns.get_level_values(0)

    temp["Return"] = temp["Close"].pct_change()

    temp["Volatility"] = (
        temp["Return"]
        .rolling(20)
        .std()
    )

    temp["Momentum"] = (
        temp["Close"]
        - temp["Close"].shift(10)
    )

    temp.dropna(inplace=True)

    X = temp[
        [
            "Return",
            "Volatility",
            "Momentum"
        ]
    ]

    X_scaled = temp_scaler.transform(X)

    latest_regime = temp_model.predict(X_scaled)[-1]

    if latest_regime == 0:
        action = "BUY"

    elif latest_regime == 1:
        action = "HOLD"

    else:
        action = "SELL"

    summary.append(
        [
            stock_name,
            latest_regime,
            action
        ]
    )

opportunities = pd.DataFrame(
    summary,
    columns=[
        "Stock",
        "Regime",
        "Action"
    ]
)

st.dataframe(
    opportunities,
    use_container_width=True
)

st.download_button(
    "📥 Download Analysis",
    csv,
    file_name=
    f"{selected_stock}_analysis.csv",
    mime="text/csv"
)