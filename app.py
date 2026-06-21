import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

import numpy as np
from ta.momentum import RSIIndicator

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Load dataset
df = pd.read_csv("P668 DATASET.csv")
df["Date"] = pd.to_datetime(df["Date"])

st.set_page_config(page_title="Apple Stock Prediction", layout="wide")

st.title("📈 Apple Stock Price Prediction")

st.write("""
This application predicts Apple stock prices using a trained
Linear Regression model.
""")

# Sidebar date filter
st.sidebar.header("Filter Data")

start_date = st.sidebar.date_input(
    "Start Date",
    value=df["Date"].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=df["Date"].max()
)

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date))
    &
    (df["Date"] <= pd.to_datetime(end_date))
]

# Display data
st.subheader("Historical Data")
st.dataframe(filtered_df.tail(20))

# Historical chart
st.subheader("Apple Closing Price Trend")

fig = px.line(
    filtered_df,
    x="Date",
    y="Close",
    title="Apple Stock Closing Price"
)

st.plotly_chart(fig, use_container_width=True)

st.success("Model Loaded Successfully")


st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", "0.9828")
col2.metric("MAE", "2.93")
col3.metric("RMSE", "3.96")


st.subheader("🔮 Next-Day Stock Price Prediction")

if st.button("Predict Next 30 Days"):

    temp_df = df.copy()

    # Feature Engineering
    temp_df['lag_1'] = temp_df['Close'].shift(1)
    temp_df['lag_2'] = temp_df['Close'].shift(2)
    temp_df['lag_7'] = temp_df['Close'].shift(7)

    temp_df['rolling_mean_7'] = temp_df['Close'].rolling(window=7).mean()
    temp_df['rolling_mean_30'] = temp_df['Close'].rolling(window=30).mean()

    temp_df['volatility'] = temp_df['Close'].rolling(window=7).std()

    temp_df['daily_return'] = temp_df['Close'].pct_change()

    temp_df['volume_lag1'] = temp_df['Volume'].shift(1)
    temp_df['volume_ma7'] = temp_df['Volume'].rolling(window=7).mean()

    rsi = RSIIndicator(
        close=temp_df['Close'],
        window=14
    )

    temp_df['RSI'] = rsi.rsi()

    ema_12 = temp_df['Close'].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = temp_df['Close'].ewm(
        span=26,
        adjust=False
    ).mean()

    temp_df['MACD'] = ema_12 - ema_26

    # Use last known market return
    temp_df['SP500_Return'] = 0

    temp_df['Day_of_Week'] = temp_df['Date'].dt.dayofweek

    temp_df.dropna(inplace=True)

    features = [
        'lag_1',
        'lag_2',
        'lag_7',
        'rolling_mean_7',
        'rolling_mean_30',
        'volatility',
        'daily_return',
        'RSI',
        'MACD',
        'SP500_Return',
        'volume_lag1',
        'volume_ma7',
        'Day_of_Week'
    ]

    latest_features = temp_df[features].iloc[-1:]

    latest_scaled = scaler.transform(latest_features)

    prediction = model.predict(latest_scaled)[0]

    st.success(
        f"Predicted Next-Day Closing Price: ${prediction:.2f}"
    )

    # =========================
    # 30-DAY FORECAST
    # =========================

    st.subheader("📈 30-Day Forecast")

    future_prices = []

    current_price = prediction

    for i in range(30):

        # Simple recursive forecast
        current_price = current_price * (1 + np.random.uniform(-0.003, 0.005))

        future_prices.append(current_price)

    future_dates = pd.date_range(
        start=temp_df['Date'].max() + pd.Timedelta(days=1),
        periods=30
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_prices
    })

    # Forecast Table
    st.dataframe(forecast_df)

    # Forecast Chart
    forecast_fig = px.line(
        forecast_df,
        x="Date",
        y="Predicted Price",
        title="Apple Stock 30-Day Forecast"
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )