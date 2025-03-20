import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


API_KEY = "TNU3H1OWGJZ2RO3P"   
STOCK_SYMBOL = "AAPL"  
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&apikey={API_KEY}&outputsize=compact"

response = requests.get(URL)
data = response.json()
df = pd.read_csv("stockdata.csv", index_col=0, parse_dates=True)
df = df.sort_index()

# Prepare Data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences for LSTM
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

# Define LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=32))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=20, batch_size=32)

last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape((1, sequence_length, 1))

# Predict next 30 days
forecast = []
for _ in range(30):
    predicted_value = model.predict(last_sequence)[0][0]
    forecast.append(predicted_value)
    
    # Update the input sequence
    last_sequence = np.append(last_sequence[:, 1:, :], [[[predicted_value]]], axis=1)

# Inverse transform to original scale
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

#Plot LSTM Forecast
forecast_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Observed', color='blue')
plt.plot(forecast_dates, forecast, label='LSTM Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{STOCK_SYMBOL} Stock Price Forecast (LSTM)')
plt.legend()
plt.grid()
plt.show()

#Evaluate Model Performance

test = df['Close'].iloc[-30:]
pred = forecast[:30, 0]

mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = np.mean(np.abs((test - pred) / test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
