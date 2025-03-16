import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the time-series data
df = pd.read_csv("stockdata.csv", index_col=0, parse_dates=True)

# Ensure data is sorted by date
df = df.sort_index()

# Step 1: Check for Missing Values
print("Missing Values:\n", df.isnull().sum())

# Step 2: Plot the Time-Series Data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Closing Price", alpha=0.6)
plt.plot(df["Close"].rolling(window=30).mean(), label="30-Day Moving Avg", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price with 30-Day Moving Average")
plt.legend()
plt.grid()
plt.show()

# Step 3: Calculate Average Values for Different Time Windows
df["Daily_Mean"] = df["Close"].resample("D").mean()
df["Weekly_Mean"] = df["Close"].resample("W").mean()
df["Monthly_Mean"] = df["Close"].resample("M").mean()

# Plot Rolling Averages
plt.figure(figsize=(12, 6))
plt.plot(df["Daily_Mean"], label="Daily Mean", alpha=0.5)
plt.plot(df["Weekly_Mean"], label="Weekly Mean", alpha=0.7, color="orange")
plt.plot(df["Monthly_Mean"], label="Monthly Mean", alpha=0.9, color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Daily, Weekly, and Monthly Mean")
plt.legend()
plt.grid()
plt.show()

# Step 4: Create a Heatmap to Identify Patterns
df["Hour"] = df.index.hour
df["Day"] = df.index.day
heatmap_data = df.pivot_table(values="Close", index="Hour", columns="Day", aggfunc="mean")

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=False)
plt.title("Hourly Price Trend Across Days")
plt.xlabel("Day")
plt.ylabel("Hour")
plt.show()

# Step 5: Augmented Dickey-Fuller (ADF) Test for Stationarity
def adf_test(timeseries):
    result = adfuller(timeseries.dropna())
    print("ADF Test Results:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")

    if result[1] <= 0.05:
        print("The data is stationary (reject H0).")
    else:
        print("The data is NOT stationary (fail to reject H0). Consider differencing.")

# Run ADF Test
adf_test(df["Close"])

# Step 6: Plot Autocorrelation & Partial Autocorrelation (ACF/PACF)
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(df["Close"].dropna(), ax=ax[0], lags=30)
plot_pacf(df["Close"].dropna(), ax=ax[1], lags=30)
plt.show()