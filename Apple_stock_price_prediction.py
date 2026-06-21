#!/usr/bin/env python
# coding: utf-8

# # Apple Stock Price Prediction and Forecasting
# 
# ## Project Objective
# The objective of this project is to analyze Apple stock market data using
# Time Series Forecasting, Machine Learning, and Deep Learning models
# to predict future stock prices and compare model performances.

# # Importing Required Libraries
# 
# This section imports all necessary Python libraries used for:
# 
# - Data manipulation
# - Data visualization
# - Time series analysis
# - Machine learning
# - Deep learning

# In[2]:


#Ignore unnecessary warning messages for cleaner notebook output
import warnings
warnings.filterwarnings('ignore')

# Import numerical and dataframe libraries
import numpy as np
import pandas as pd

# Import visualization libraries for plotting graphs and charts
import matplotlib.pyplot as plt
import seaborn as sns

# Import time series analysis libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# Import preprocessing library for feature scaling
from sklearn.preprocessing import MinMaxScaler


# Import machine learning regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Import model tuning and time-series cross-validation tools
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit
)


# Import deep learning libraries for LSTM and GRU models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.layers import GRU


# Import evaluation metrics for model performance analysis
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score


# 

# # Dataset Information
# The dataset contains historical Apple stock market data including Open, High, Low, Close, Adjusted Close, and Volume features. The data is used for time series forecasting and predictive modeling

# In[3]:


# Read Apple stock dataset into pandas dataframe
df=pd.read_csv("P668 DATASET.csv")


# In[4]:


# Display first few rows of dataset
df.head()

#Data types
df.info()


# # Data Preprocessing
# 
# This section prepares the dataset for analysis by:
# 
# - Converting the Date column into datetime format
# - Setting Date as index
# - Handling missing values
# - Checking duplicates
# - Understanding statistical properties of data

# In[5]:


# Convert Date column into datetime format
df["Date"]=pd.to_datetime(df['Date'])

# Set Date column as dataframe index
df.set_index('Date',inplace=True)

# Display beginning and ending records of dataset
df.head()
df.tail()


# In[6]:


# Display total rows and columns in dataset
df.shape


# In[7]:


#Statistical summary of dataset
df.describe()


# In[8]:


# Check missing values in each column
df.isnull().sum()


# In[9]:


# Fill missing values using forward fill method
df.fillna(method='ffill',inplace=True)

# Check duplicate rows in dataset
df.duplicated().sum()


# # Exploratory Data Analysis (EDA)
# 
# EDA helps in understanding:
# 
# - Market trends
# - Stock price behavior
# - Volatility
# - Distribution patterns
# - Correlation between variables
# - Presence of outliers
# 
# ## Business Problem Statement
# 
# Stock market forecasting helps investors and financial analysts make informed investment decisions by identifying future market trends and price movements.

# In[10]:


# Plot Apple closing price trend over time

plt.figure(figsize=(15,6))
plt.plot(df['Close'])
plt.title("Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()


# ### Observation
# - Apple stock price shows an overall upward trend over the years.
# - Some fluctuations indicate market volatility.
# - Strong growth periods can be observed after major market recoveries.

# In[11]:


# Plot stock trading volume trend over time

plt.figure(figsize=(15,6))
plt.plot(df['Volume'])
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume Traded")
plt.show()


# In[12]:


# Visualize distribution of closing prices using histogram

plt.figure(figsize=(8,5))
plt.hist(df['Close'], bins=30)
plt.title("Distribution of Closing Prices")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.show()


# In[13]:


# Visualize distribution of trading volume

plt.figure(figsize=(8,5))
plt.hist(df['Volume'], bins=30)
plt.title("Distribution of Trading Volume")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.show()


# In[14]:


# Plot KDE curve to analyze density distribution of closing prices

plt.figure(figsize=(10,5))
sns.kdeplot(df['Close'], fill=True)
plt.title("KDE Plot of Closing Prices")
plt.xlabel("Close Price")
plt.show()


# In[15]:


# Plot KDE curve for volume distribution analysis

plt.figure(figsize=(10,5))
sns.kdeplot(df['Volume'], fill=True)
plt.title("KDE Plot of Trading Volume")
plt.xlabel("Volume")
plt.show()


# In[16]:


# Detect outliers in closing prices using boxplot

plt.figure(figsize=(8,5))
plt.boxplot(x=df['Close'])
plt.title("Boxplot of Closing Prices")
plt.show()


# In[17]:


# Detect outliers in trading volume using horizontal boxplot

plt.figure(figsize=(10,4))
plt.boxplot(df['Volume'], vert=False)
plt.title("Boxplot of Volume Trend")
plt.show()


# In[18]:


# Calculate IQR method to identify closing price outliers

Q1 = df['Close'].quantile(0.25)
Q3 = df['Close'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Close'] < lower) | (df['Close'] > upper)]

print("Number of Outliers:", outliers.shape[0])


# In[19]:


# Calculate outliers for trading volume using IQR method

Q1 = df['Volume'].quantile(0.25)
Q3 = df['Volume'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Volume'] < lower_bound) | 
              (df['Volume'] > upper_bound)]

print("Total Outliers:", outliers.shape[0])


# In[20]:


# Generate correlation heatmap to identify feature relationships

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()


# ### Observation
# - Lag features and moving averages show strong positive correlation with Close price.
# - Volume has weaker correlation compared to technical indicators.
# - Highly correlated features help improve prediction performance.

# In[21]:


# Calculate rolling mean and rolling standard deviation
# Used for trend and volatility analysis

rolling_mean = df['Close'].rolling(window=30).mean()
rolling_std = df['Close'].rolling(window=30).std()

# Plot rolling mean with actual closing prices
plt.figure(figsize=(16,6))

plt.plot(df['Close'], label='Close Price')
plt.plot(rolling_mean, label='Rolling Mean')

plt.legend()
plt.title("Close Price and Rolling Mean")

plt.show()

# Plot rolling mean with actual closing prices
plt.figure(figsize=(16,6))

plt.plot(rolling_std, label='Rolling Std')

plt.legend()
plt.title("Rolling Standard Deviation")

plt.show()


# In[22]:


# Perform seasonal decomposition to separate trend,
# seasonality, and residual components

# Useful for understanding stock price behavior patterns

decomposition = seasonal_decompose(df['Close'], model='additive', period=30)

decomposition.plot()

plt.show()


# In[23]:


# Perform Augmented Dickey-Fuller test to check stationarity
result = adfuller(df['Close'])

print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:")

for key, value in result[4].items():
    print(key, ":", value)


# In[24]:


# Plot ACF and PACF graphs for ARIMA parameter selection

plt.figure(figsize=(12,5))
plot_acf(df['Close'].dropna(), lags=40)
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(df['Close'].dropna(), lags=40)
plt.show()


# In[25]:


original_df = df.copy()


# In[26]:


# Create copy of dataset for ARIMA modeling
arima_df = df.copy()

# Apply logarithmic transformation to stabilize variance
arima_df['Log_Close'] = np.log(arima_df['Close'])

# Apply differencing to remove trend and achieve stationarity
arima_df['Log_Diff'] = arima_df['Log_Close'].diff()

# Remove missing values generated after differencing
arima_df.dropna(inplace=True)


# In[27]:


# Perform ADF test again after differencing
# Used to verify stationarity improvement.201

result = adfuller(arima_df['Log_Diff'])

print("p-value:", result[1])


# In[28]:


# Plot ACF and PACF graphs for ARIMA parameter selection

plt.figure(figsize=(12,5))
plot_acf(arima_df['Log_Diff'].dropna(), lags=40)
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(arima_df['Log_Diff'].dropna(), lags=40)
plt.show()


# In[29]:


# Split ARIMA dataset into training and testing sets

train_size = int(len(arima_df) * 0.8)

train = arima_df['Log_Close'][:train_size]

test = arima_df['Log_Close'][train_size:]


# In[30]:


# Visualize train-test split for forecasting model

plt.figure(figsize=(15,6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.legend()
plt.title("Train-Test Split")
plt.show()


# In[31]:


# Perform grid search for optimal ARIMA parameters
# based on minimum AIC and BIC scores

results = []

best_aic = float('inf')
best_bic = float('inf')

best_order_aic = None
best_order_bic = None

for p in range(0,4):
    for d in range(0,3):
        for q in range(0,4):

            try:
                model = ARIMA(train, order=(p,d,q))
                fit = model.fit()

                aic = fit.aic
                bic = fit.bic

                results.append({
                    'Order': (p,d,q),
                    'AIC': aic,
                    'BIC': bic
                })

                if aic < best_aic:
                    best_aic = aic
                    best_order_aic = (p,d,q)

                if bic < best_bic:
                    best_bic = bic
                    best_order_bic = (p,d,q)

            except:
                continue

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by=['AIC','BIC'])

print(results_df.head(10))

print("\nBest AIC Order:", best_order_aic)
print("Best AIC:", best_aic)

print("\nBest BIC Order:", best_order_bic)
print("Best BIC:", best_bic)


# In[32]:


# Train ARIMA model using selected parameters

model = ARIMA(train, order=(3,0,2))

model_fit = model.fit()

print(model_fit.summary())


# In[33]:


# Generate future forecasts for test dataset

forecast_result = model_fit.get_forecast(steps=len(test))
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Convert logarithmic forecast values back to original scale

forecast_actual = np.exp(forecast)
test_actual = np.exp(test)


# In[34]:


# Plot actual vs predicted stock prices for ARIMA model

plt.figure(figsize=(15,6))
plt.plot(test_actual.index, test_actual, label='Actual')
plt.plot(test_actual.index, forecast_actual, label='Predicted')
plt.fill_between(
    test_actual.index,
    np.exp(conf_int.iloc[:,0]),
    np.exp(conf_int.iloc[:,1]),
    alpha=0.3
)

plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()


# In[35]:


# Calculate ARIMA evaluation metrics
# MAE, RMSE, MAPE, and R2 score

test_actual = test_actual.reset_index(drop=True)
forecast_actual = forecast_actual.reset_index(drop=True)

arima_mae = mean_absolute_error(test_actual, forecast_actual)
arima_rmse = np.sqrt(mean_squared_error(test_actual, forecast_actual))
arima_mape = np.mean(
    np.abs((test_actual - forecast_actual) / (test_actual + 1e-10))
) * 100
arima_r2 = r2_score(test, forecast)

print("ARIMA R2:", arima_r2)

print("MAE:", arima_mae)

print("RMSE:", arima_rmse)

print("MAPE:", arima_mape)


# In[36]:


arima_metrics = (

    arima_mae,

    arima_rmse,

    arima_r2
)


# In[37]:


# Analyze ARIMA residual errors to validate model performance

residuals = model_fit.resid

plt.figure(figsize=(12,5))
plt.plot(residuals)
plt.title("Residual Errors")
plt.show()

sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

plot_acf(residuals, lags=40)
plt.show()

print("Residual Mean:", residuals.mean())


# In[38]:


original_df.head()


# In[39]:


# Create copy of original dataset for ML modeling
ml_df=original_df.copy()


# In[40]:


# Create target variable using next-day closing price
ml_df['Target'] = ml_df['Close'].shift(-1)

# Remove final NaN created by shift
ml_df.dropna(inplace=True)


# In[41]:


ml_df


# In[42]:


# Create lag features using previous closing prices

ml_df['lag_1'] = ml_df['Close'].shift(1)
ml_df['lag_2'] = ml_df['Close'].shift(2)
ml_df['lag_7'] = ml_df['Close'].shift(7)

# Create moving average features for trend analysis
ml_df['rolling_mean_7'] = ml_df['Close'].rolling(window=7).mean()
ml_df['rolling_mean_30'] = ml_df['Close'].rolling(window=30).mean()

# Create volatility feature using rolling standard deviation
ml_df['volatility'] = ml_df['Close'].rolling(window=7).std()

# Calculate daily percentage return of stock price
ml_df['daily_return'] = ml_df['Close'].pct_change()

# Create volume-based lag and moving average features
ml_df['volume_lag1'] = ml_df['Volume'].shift(1)
ml_df['volume_ma7'] = ml_df['Volume'].rolling(window=7).mean()


# In[43]:


ml_df.head()


# In[44]:


# Calculate Relative Strength Index (RSI)
# Used to measure stock momentum
from ta.momentum import RSIIndicator

# Create RSI feature using 14-day window
# 14 is the standard period commonly used in stock market analysis
rsi = RSIIndicator(
    close=ml_df['Close'],
    window=14
)

# Add RSI values as a new feature column
# This feature helps ML models understand stock momentum trends
ml_df['RSI'] = rsi.rsi()


# In[45]:


# Calculate 12-day Exponential Moving Average
# Short-term trend indicator
ema_12 = ml_df['Close'].ewm(span=12, adjust=False).mean()

# Calculate 26-day Exponential Moving Average
# Long-term trend indicator
ema_26 = ml_df['Close'].ewm(span=26, adjust=False).mean()

# Create MACD feature
# MACD measures momentum by subtracting long-term EMA from short-term EMA
# Positive MACD indicates bullish trend and negative indicates bearish trend
ml_df['MACD'] = ema_12 - ema_26


# In[46]:


# Import yfinance library
# Used to download historical financial market data directly from Yahoo Finance
import yfinance as yf


# Download S&P 500 historical market data
# It helps the model understand overall market movement influence on Apple stock
sp500 = yf.download(
    "^GSPC",
    start=df.index.min(),
    end=df.index.max()
)


# In[47]:


# Keep only closing prices
sp500 = sp500[['Close']]

# Remove MultiIndex and set clean column name
sp500.columns = ['SP500']

# Join using Date index
ml_df = ml_df.join(sp500, how='left')

# Fill missing values
ml_df['SP500'] = ml_df['SP500'].ffill()

# Create return feature
ml_df['SP500_Return'] = ml_df['SP500'].pct_change()

# Create weekday feature
ml_df['Day_of_Week'] = ml_df.index.dayofweek


# In[48]:


# Display first few rows after feature engineering
ml_df.head()

# Check total missing values in each column
ml_df.isnull().sum()


# In[49]:


# Display first few rows after feature engineering
ml_df.dropna(inplace=True)

# Display cleaned dataset
ml_df.head()

# Verify whether missing values are completely removed
ml_df.isnull().sum()


# In[50]:


# Correlation Heatmap
# Helps analyze relationships between features and target variable

plt.figure(figsize=(18,10))
sns.heatmap(ml_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[51]:


# Plot daily stock returns over time

plt.figure(figsize=(14,6))
plt.plot(ml_df.index, ml_df['daily_return'])
plt.title('Daily Returns of Apple Stock')
plt.xlabel('Date')
plt.ylabel('Daily Return')

plt.show()


# In[52]:


# Distribution plot of daily returns

plt.figure(figsize=(10,5))

ml_df['daily_return'].hist(bins=50)

plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')

plt.show()


# In[53]:


ml_df.isnull().sum()

# Display dataset shape
ml_df.shape


# In[54]:


# Selected input features for machine learning models

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


# In[55]:


# Define feature matrix (X)
X = ml_df[features]

# Define target variable (y)
y = ml_df["Target"]


# In[56]:


# Split dataset into training and testing sets
train_size = int(len(ml_df) * 0.8)

# Split features
X_train = X[:train_size]
X_test = X[train_size:]

# Split target
y_train = y[:train_size]
y_test = y[train_size:]

# Display shapes
print("Training Feature Shape :", X_train.shape)
print("Testing Feature Shape  :", X_test.shape)

print("Training Target Shape  :", y_train.shape)
print("Testing Target Shape   :", y_test.shape)


# ### Feature Scaling
# Feature scaling is applied to normalize numerical values into a fixed range, improving model convergence and deep learning performance.

# In[57]:


# Apply Min-Max Scaling to improve performance of models

lr_scaler = MinMaxScaler()

X_train_scaled = lr_scaler.fit_transform(X_train)
X_test_scaled = lr_scaler.transform(X_test)


# ## Evaluation Metrics
# - MAE measures average absolute prediction error.
# - RMSE gives higher importance to large prediction errors.
# - R² Score indicates how well the model explains variance in stock prices.

# In[58]:


# Import evaluation metrics

# Custom reusable evaluation function
# Calculates MAE, RMSE, and R2 score for model performance comparison
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_model(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}")
    print("-" * 30)

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    return mae, rmse, r2


# ### Why Linear Regression?
# Linear Regression is used as a baseline model to understand the linear relationship between stock features and future closing prices.

# In[59]:


#Linear Regression Model

lr_model = LinearRegression()

# Train model using scaled training data
lr_model.fit(X_train_scaled, y_train)


# Generate predictions on testing data
lr_pred = lr_model.predict(X_test_scaled)

# Evaluate Linear Regression performance
lr_metrics = evaluate_model(
    y_test,
    lr_pred,
    "Linear Regression"
)


# In[60]:


# Plot actual vs predicted values

plt.figure(figsize=(15,6))
plt.plot(y_test.values, label='Actual')
plt.plot(lr_pred, label='Predicted')
plt.title("Linear Regression Prediction")
plt.legend()

plt.show()


# ### Why Random Forest?
# Random Forest helps capture non-linear relationships and reduces overfitting using ensemble learning.

# In[61]:


# Random Forest is an ensemble learning algorithm

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train Random Forest model
rf_model.fit(X_train, y_train)


# Generate predictions
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_metrics = evaluate_model(
    y_test,
    rf_pred,
    "Random Forest"
)


# In[62]:


# Plot Random Forest predictions against actual values

plt.figure(figsize=(15,6))
plt.plot(y_test.values, label='Actual')
plt.plot(rf_pred, label='Predicted')
plt.title("Random Forest Prediction")
plt.legend()

plt.show()


# In[63]:


# Calculate feature importance scores
# Helps identify which features contribute most to prediction

importance = pd.DataFrame({

    'Feature': features,

    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)


# In[64]:


# Visualize Random Forest feature importance
# Higher values indicate stronger influence on predictions

plt.figure(figsize=(10,6))

plt.barh(
    importance['Feature'],
    importance['Importance']
)

plt.gca().invert_yaxis()

plt.title("Random Forest Feature Importance")

plt.show()


# ### Why XGBoost?
# XGBoost is a boosting algorithm that improves prediction accuracy by sequentially correcting previous model errors.

# In[65]:


# XGBoost is an advanced boosting algorithm

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train XGBoost model
xgb_model.fit(X_train_scaled, y_train)

# Generate predictions
xgb_pred = xgb_model.predict(X_test_scaled)

# Evaluation
xgb_metrics = evaluate_model(
    y_test,
    xgb_pred,
    "XGBoost"
)


# In[66]:


# Visualizing XGBoost actual vs predicted stock prices

plt.figure(figsize=(15,6))
plt.plot(y_test.values, label='Actual')
plt.plot(xgb_pred, label='Predicted')
plt.title("XGBoost Prediction")
plt.legend()

plt.show()


# In[67]:


# Creating feature importance table for XGBoost model

xgb_importance = pd.DataFrame({

    'Feature': features,

    'Importance': xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by='Importance',
    ascending=False
)

print(xgb_importance)


# In[68]:


# Plotting XGBoost feature importance chart

plt.figure(figsize=(10,6))

plt.barh(
    importance['Feature'],
    importance['Importance']
)

plt.gca().invert_yaxis()
plt.title("Xgboost Feature Importance")

plt.show()


# In[69]:


# Applying Time Series Cross Validation and Hyperparameter Tuning for Random Forest

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [5, 10],

    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(

    RandomForestRegressor(random_state=42),

    param_grid,

    cv=tscv,

    scoring='neg_mean_squared_error',

    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_

rf_tuned_pred = best_rf.predict(X_test)

# Evaluation
rf_tuned_metrics = evaluate_model(
    y_test,
    rf_tuned_pred,
    "Tuned Random Forest"
)


# In[70]:


# Displaying best Random Forest parameters and score

print("Best Random Forest Parameters:")
print(grid_rf.best_params_)

print("\nBest Random Forest Score:")
print(grid_rf.best_score_)


# In[71]:


# Applying Hyperparameter Tuning for XGBoost model

param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [3, 6],

    'learning_rate': [0.01, 0.05]
}

grid_xgb = GridSearchCV(

    XGBRegressor(random_state=42),

    param_grid,

    cv=tscv,

    scoring='neg_mean_squared_error',

    n_jobs=-1
)

grid_xgb.fit(X_train_scaled, y_train)

best_xgb = grid_xgb.best_estimator_

xgb_tuned_pred = best_xgb.predict(X_test_scaled)

# Evaluation
xgb_tuned_metrics = evaluate_model(
    y_test,
    xgb_tuned_pred,
    "Tuned XGBoost"
)


# In[72]:


# Displaying best XGBoost parameters and score

print("Best XGBoost Parameters:")
print(grid_xgb.best_params_)

print("\nBest XGBoost Score:")
print(grid_xgb.best_score_)


# In[73]:


# Extracting testing data for Naive Forecasting
test_data = ml_df[train_size:].copy()


# In[74]:


# Creating Naive Forecast using previous day's closing price
test_data['Naive_Prediction'] = test_data['Close'].shift(1)

# Removing null values generated after shifting
test_data.dropna(inplace=True)

# Separating actual and predicted values for Naive Forecast
y_true = test_data['Target']
y_pred = test_data['Naive_Prediction']


# In[75]:


# Calculating evaluation metrics for Naive Baseline model

naive_mae = mean_absolute_error(y_true, y_pred)

naive_rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

naive_r2 = r2_score(y_true, y_pred)

print("Naive MAE :", naive_mae)

print("Naive RMSE:", naive_rmse)

print("Naive R2  :", naive_r2)


# In[76]:


naive_metrics = (

    naive_mae,

    naive_rmse,

    naive_r2
)


# In[77]:


# Plotting Naive Forecast predictions against actual values

plt.figure(figsize=(15,6))

plt.plot(
    y_true.values,
    label='Actual'
)

plt.plot(
    y_pred.values,
    label='Naive Prediction'
)

plt.title("Naive Baseline Forecast")

plt.legend()

plt.show()


# In[78]:


# Selecting Close price data for Deep Learning models
lstm_df = ml_df[['Close']].copy()


# In[79]:


# Scaling stock price data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(lstm_df)


# In[80]:


# Creating sequential data for LSTM and GRU models
sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(
        scaled_data[i-sequence_length:i, 0]
    )

    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)
y = np.array(y)


# In[81]:


# Splitting sequential data into training and testing sets

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# In[82]:


# Reshaping input data into 3D format for LSTM/GRU models

X_train = np.reshape(

    X_train,

    (X_train.shape[0],
     X_train.shape[1],
     1)
)

X_test = np.reshape(

    X_test,

    (X_test.shape[0],
     X_test.shape[1],
     1)
)

print(X_train.shape)
print(X_test.shape)


# ### Why LSTM?
# LSTM is suitable for sequential and time-series data because it can remember long-term dependencies in stock price movements.

# In[83]:


# Building LSTM Deep Learning model architecture
lstm_model = Sequential()

lstm_model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    )
)

lstm_model.add(Dropout(0.2))

lstm_model.add(
    LSTM(
        units=50
    )
)

lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))


# In[84]:


# Compiling LSTM model using Adam optimizer and MSE loss
lstm_model.compile(

    optimizer='adam',

    loss='mean_squared_error'
)


# In[85]:


# Training LSTM model on stock price sequences
history = lstm_model.fit(

    X_train,

    y_train,

    epochs=20,

    batch_size=32,

    validation_data=(X_test, y_test),

    verbose=1
)


# In[86]:


# Generating stock price predictions using LSTM model
lstm_pred = lstm_model.predict(X_test)

# Converting scaled LSTM predictions back to original values

lstm_pred = scaler.inverse_transform(lstm_pred)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1,1)
)


# In[87]:


# Calculating evaluation metrics for LSTM model

lstm_mae = mean_absolute_error(
    y_test_actual,
    lstm_pred
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        lstm_pred
    )
)

lstm_r2 = r2_score(
    y_test_actual,
    lstm_pred
)

print("LSTM MAE :", lstm_mae)

print("LSTM RMSE:", lstm_rmse)

print("LSTM R2  :", lstm_r2)


# In[88]:


lstm_metrics = (

    lstm_mae,

    lstm_rmse,

    lstm_r2
)


# In[89]:


# Plotting actual vs LSTM predicted stock prices
plt.figure(figsize=(15,6))

plt.plot(
    y_test_actual,
    label='Actual'
)

plt.plot(
    lstm_pred,
    label='LSTM Prediction'
)

plt.title("LSTM Stock Prediction")
plt.legend()

plt.show()


# ### Why GRU?
# GRU is a simplified recurrent neural network architecture that performs faster training while handling sequential dependencies effectively.

# In[90]:


# Building GRU Deep Learning model architecture
gru_model = Sequential()

gru_model.add(
    GRU(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    )
)

gru_model.add(Dropout(0.2))

gru_model.add(
    GRU(
        units=50
    )
)

gru_model.add(Dropout(0.2))
gru_model.add(Dense(1))


# In[91]:


# Compiling GRU model using Adam optimizer and MSE loss
gru_model.compile(

    optimizer='adam',

    loss='mean_squared_error'
)


# In[92]:


# Training GRU model on stock price sequences

gru_history = gru_model.fit(

    X_train,

    y_train,

    epochs=20,

    batch_size=32,

    validation_data=(X_test, y_test),

    verbose=1
)


# In[93]:


# Generating stock price predictions using GRU model
gru_pred = gru_model.predict(X_test)

# Converting scaled GRU predictions back to original values
gru_pred = scaler.inverse_transform(gru_pred)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1,1)
)


# In[94]:


# Calculating evaluation metrics for GRU model
gru_mae = mean_absolute_error(
    y_test_actual,
    gru_pred
)

gru_rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        gru_pred
    )
)

gru_r2 = r2_score(
    y_test_actual,
    gru_pred
)

print("GRU MAE :", gru_mae)

print("GRU RMSE:", gru_rmse)

print("GRU R2  :", gru_r2)


# In[95]:


gru_metrics = (

    gru_mae,

    gru_rmse,

    gru_r2
)


# In[96]:


# Plotting actual vs GRU predicted stock prices

plt.figure(figsize=(15,6))

plt.plot(
    y_test_actual,
    label='Actual'
)

plt.plot(
    gru_pred,
    label='GRU Prediction'
)

plt.title("GRU Stock Prediction")
plt.legend()

plt.show()


# In[97]:


# Creating final comparison table for all forecasting models
comparison = pd.DataFrame({

    'Model': [

        'ARIMA',

        'Naive Baseline',

        'Linear Regression',

        'Random Forest',

        'Tuned Random Forest',

        'XGBoost',

        'Tuned XGBoost',

        'LSTM',

        'GRU'
    ],

    'MAE': [

        arima_metrics[0],

        naive_metrics[0],

        lr_metrics[0],

        rf_metrics[0],

        rf_tuned_metrics[0],

        xgb_metrics[0],

        xgb_tuned_metrics[0],

        lstm_metrics[0],

        gru_metrics[0]
    ],

    'RMSE': [

        arima_metrics[1],

        naive_metrics[1],

        lr_metrics[1],

        rf_metrics[1],

        rf_tuned_metrics[1],

        xgb_metrics[1],

        xgb_tuned_metrics[1],

        lstm_metrics[1],

        gru_metrics[1]
    ],

    'R2 Score': [

        arima_metrics[2],

        naive_metrics[2],

        lr_metrics[2],

        rf_metrics[2],

        rf_tuned_metrics[2],

        xgb_metrics[2],

        xgb_tuned_metrics[2],

        lstm_metrics[2],

        gru_metrics[2]
    ]
})

comparison


# In[98]:


# Sorting models based on RMSE performance
comparison = comparison.sort_values(

    by='RMSE',

    ascending=True
)

comparison


# In[99]:


# Identifying the best performing model
best_model = comparison.iloc[0]

print(best_model)


# In[100]:


# Visualizing RMSE comparison among all models
plt.figure(figsize=(12,6))

plt.bar(

    comparison['Model'],

    comparison['RMSE']
)

plt.xticks(rotation=45)

plt.title("Model Comparison Based on RMSE")

plt.ylabel("RMSE")

plt.show()


# # Model Comparison
# 
# All statistical, machine learning, and deep learning models
# were compared using evaluation metrics to identify the most
# effective approach for Apple stock price forecasting.

# # Conclusion
# 
# This project successfully implemented statistical,
# machine learning, and deep learning approaches for
# next-day Apple stock price prediction using engineered
# financial and technical indicators.
# 
# Successfully analyzed historical stock market data using EDA and feature engineering techniques
# 
# Implemented multiple forecasting approaches including Time Series, Machine Learning, and Deep Learning models
# 
# Evaluated model performance using MAE, RMSE, and R² Score metrics
# 
# Linear Regression achieved the best overall performance for this project compared to other implemented models
# 
# The model provided stable, accurate, and interpretable predictions for stock price forecasting
# 
# Feature engineering techniques such as lag features, moving averages, RSI, EMA, and MACD improved prediction capability
# 
# Demonstrated the importance of preprocessing, feature selection, and model evaluation in financial forecasting
# 
# Developed a complete end-to-end stock price prediction workflow for financial analytics applications
# 
# The project can be further enhanced using real-time stock data, sentiment analysis, and deployment frameworks like Streamlit

# In[ ]:





# In[101]:


print(features)


# In[102]:


import joblib

joblib.dump(lr_model, "model.pkl")
joblib.dump(lr_scaler, "scaler.pkl")

print("Model and Scaler Saved Successfully")
print(lr_scaler.feature_names_in_)


# In[103]:


import os

print(os.path.exists("model.pkl"))
print(os.path.exists("scaler.pkl"))


# In[ ]:





# In[ ]:




