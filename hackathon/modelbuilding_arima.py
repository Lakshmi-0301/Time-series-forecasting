from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import requests


API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
STOCK_SYMBOL = "AAPL"  
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&apikey={API_KEY}&outputsize=compact"

response = requests.get(URL)
data = response.json()
df = pd.read_csv("stockdata.csv", index_col=0, parse_dates=True)
df = df.sort_index()

# Fit ARIMA model (p, d, q)
model = ARIMA(df['Close'], order=(5, 1, 2))  # Example order, we'll tune it next
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

#Forecasting Using ARIMA
forecast_steps = 30
forecast = model_fit.get_forecast(steps=forecast_steps)
confidence_intervals = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Observed', color='blue')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='green')
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{STOCK_SYMBOL} Stock Price Forecast (ARIMA)')
plt.legend()
plt.grid()
plt.show()

# Automatically find the best (p, d, q) parameters
auto_model = auto_arima(df['Close'], seasonal=False, trace=True, 
                        error_action='ignore', suppress_warnings=True, stepwise=True)

print(auto_model.summary())


# Fit SARIMA model (p, d, q) x (P, D, Q, S)
sarima_model = SARIMAX(df['Close'], order=(5, 1, 2), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()

# Summary of the model
print(sarima_fit.summary())

#Forecasting Using SARIMA
forecast_steps = 30
forecast = sarima_fit.get_forecast(steps=forecast_steps)
confidence_intervals = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Observed', color='blue')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='green')
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{STOCK_SYMBOL} Stock Price Forecast (SARIMA)')
plt.legend()
plt.grid()
plt.show()

# Test on last 30 data points
test = df['Close'].iloc[-30:]
pred = sarima_fit.predict(start=test.index[0], end=test.index[-1])

# Evaluation metrics
mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = np.mean(np.abs((test - pred) / test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

