import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def generate_house_data(n_samples=200):
    np.random.seed(50)

    size = np.random.normal(1400, 250, n_samples)
    price = size * 50 + np.random.normal(0, 5000, n_samples)

    return pd.DataFrame({
        'size': size,
        'price': price
    })


def train_model():
    df = generate_house_data()

    X = df[['size']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model


def main():
    st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

    st.title("🏠 House Price Prediction")
    st.write("Enter the house size to predict its price.")

    model = train_model()

    st.subheader("Model Performance")
   

    size = st.number_input(
        "House Size (sq ft)",
        min_value=500,
        max_value=3000,
        value=1500,
        step=50
    )

    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])[0]

        st.success(
            f"Predicted House Price: ${predicted_price:,.2f}"
        )

        df = generate_house_data()

        fig = px.scatter(
            df,
            x='size',
            y='price',
            title='House Size vs Price',
            labels={
                'size': 'House Size (sq ft)',
                'price': 'Price ($)'
            }
        )

        fig.add_scatter(
            x=[size],
            y=[predicted_price],
            mode='markers',
            marker=dict(
                color='red',
                size=15
            ),
            name='Prediction'
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()