import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 1: Fetch Data from API (Example: Alpha Vantage Stock Prices)
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
STOCK_SYMBOL = "AAPL"  # Example: Apple Stock
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&apikey={API_KEY}&outputsize=compact"

response = requests.get(URL)
data = response.json()

# 🔹 Step 2: Extract and Preprocess Data
if "Time Series (Daily)" in data:
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df = df.astype(float)  # Convert all values to float
    df = df.sort_index()  # Ensure data is sorted by date
else:
    print("Error fetching data:", data)
    exit()

# 🔹 Step 3: Basic Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Closing Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{STOCK_SYMBOL} Stock Price Over Time")
plt.legend()
plt.grid()
plt.show()

# 🔹 Moving Average (Trend Analysis)
df["MA_10"] = df["Close"].rolling(window=10).mean()  # 10-day moving average

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Closing Price", alpha=0.6)
plt.plot(df.index, df["MA_10"], label="10-day Moving Avg", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{STOCK_SYMBOL} Stock Price with 10-day Moving Average")
plt.legend()
plt.grid()
plt.show()

print(df)
df.to_csv("stockdata.csv")
