# Time-series-forecasting

## Key Features:
- **Data Retrieval**: Fetch time-series data using REST API.
- **Exploratory Data Analysis**:
  - Plot the time-series data.
  - Identify the trend, seasonality, and cyclic components in the data.
  - Calculate the average value for each time window (daily, weekly, or monthly).
  - Create a heatmap to identify patterns across different time periods (e.g., hourly, daily).
- **Model Building**:
  - Train ARIMA and SARIMA models.
- **Model Evaluation**:
  - Evaluate models using MAE, RMSE, and MAPE metrics.
  - Generate confidence intervals and provide insights.

## Scope of Work:

### Milestone 1: Data Retrieval
- Fetch time-series data.
- Handle authentication, rate limits, and API errors.

### Milestone 2: Data Cleaning and Preprocessing
- Handle missing values, duplicates, and time zone inconsistencies.
- Convert the timestamps to a consistent format like (YYYY-MM-DD HH:MM:SS).

### Milestone 3: Exploratory Data Analysis

#### Basic EDA:
- Plot the time-series data.
- Identify the trend, seasonality, and cyclic components in the data.
- Calculate the average value for each time window (daily, weekly, or monthly).
- Create a heatmap to identify patterns across different time periods (e.g., hourly, daily).

#### Trends and Stationarity:
- Identify the length of any recurring cycles using autocorrelation.
- Check if the time-series data is stationary using the Augmented Dickey-Fuller (ADF) test.

### Milestone 4: Model Building and Model Evaluation
- Train and test ARIMA, SARIMA, or machine learning models.
- Fine-tune hyperparameters for better accuracy.
- Evaluate model performance using MAE, RMSE, and MAPE.

### Milestone 5: Forecasting
- Predict the next value of a time series for a specific time frame.
- Generate a confidence interval around your forecast values and plot it.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, requests, tensorflow

## How to Run

1) To clone the repository, use the following command:

```bash
git clone https://github.com/username/repository-name.git
```
2) pip install -r requirements.txt
3) Run the code files.
