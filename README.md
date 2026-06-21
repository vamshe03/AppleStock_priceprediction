# Apple Stock Price Prediction and Forecasting

## Project Overview

This project aims to predict and forecast Apple stock prices using Machine Learning techniques. Historical stock market data from 2012 to 2019 was analyzed to identify patterns and trends that can help estimate future stock prices.

Multiple models were developed and compared, including traditional machine learning, statistical forecasting, and deep learning approaches. Based on evaluation metrics, Linear Regression was selected as the best-performing model and deployed using Streamlit.

---

## Objective

- Analyze historical Apple stock data.
- Build and compare forecasting models.
- Predict future stock prices.
- Develop an interactive web application for visualization and forecasting.

---

## Dataset

- Company: Apple Inc. (AAPL)
- Period: 2012 - 2019
- Source: Yahoo Finance

Features available in the dataset include Open, High, Low, Close, Adjusted Close, and Volume.

---

## Feature Engineering

The following features were created to improve prediction performance:

- Lag Features (1, 2, and 7 days)
- 7-Day Moving Average
- 30-Day Moving Average
- Volatility
- Daily Returns
- RSI Indicator
- MACD Indicator
- S&P 500 Returns
- Volume-Based Features
- Day of Week

---

## Models Used

- ARIMA
- Naive Baseline
- Linear Regression
- Random Forest
- Tuned Random Forest
- XGBoost
- Tuned XGBoost
- LSTM
- GRU

---

## Best Model

**Linear Regression**

Performance:

- R² Score: 0.9828
- MAE: 2.93
- RMSE: 3.96

---

## Streamlit Application Features

- Historical stock data visualization
- Date range filtering
- Interactive stock price trend chart
- Model performance metrics
- Next-day stock price prediction
- 30-day stock price forecast

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit
- TensorFlow/Keras
- XGBoost

---

## Project Structure

```text
├── app.py
├── model.pkl
├── scaler.pkl
├── P668 DATASET.csv
├── requirements.txt
└── README.md
```

---

## Challenges Faced

- Handling missing values generated during feature engineering.
- Comparing multiple forecasting models and selecting the best one.
- Maintaining consistency between training and deployment features.
- Resolving model and scaler compatibility issues during Streamlit deployment.

---

## Future Scope

- Use real-time stock market data.
- Improve forecasting accuracy using advanced models.
- Extend support to multiple stocks.
- Deploy the application on a cloud platform for public access.

---

## Conclusion

This project demonstrates how machine learning can be used to analyze historical stock market data and forecast future stock prices. The developed Streamlit application provides an easy-to-use interface for exploring historical trends and generating future forecasts.

---

**Author:** Vamsi Mangam
