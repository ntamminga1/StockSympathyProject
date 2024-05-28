# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_paths = [
    'WMT.csv',
    'ANF.csv',
    'KR.csv',
    'COST.csv',
    'CASY.csv',
    'DECK.csv',
    'DKS.csv',
    'HD.csv',
    'LOW.csv',
    'SFM.csv',
    'TJX.csv',
    'TSCO.csv',
    'WSM.csv'
]

all_returns = []

allc2hReturn = []

for filepath in file_paths:
    
    df = pd.read_csv(filepath)
    
    df['Previous_close'] = df['Close'].shift(1)
    
    df['c2cReturn'] = df['Close']/df['Previous_close']-1
    
    df['c2hReturn'] = df['High']/df['Previous_close']-1
    
    c2cReturn_series = df['c2cReturn']
    
    c2hReturn_series = df['c2hReturn']
    
    dataset_name = filepath.replace('.csv', '')
    
    c2cReturn_series.name = dataset_name
    
    c2hReturn_series.name = dataset_name
    
    all_returns.append(c2cReturn_series)
    
    allc2hReturn.append(c2hReturn_series)
    
c2cReturn_df = pd.concat(all_returns, axis=1)

c2hReturn_df = pd.concat(allc2hReturn, axis=1)


# DATASET to Predict Walmart
#Setup the date correctly
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.tz_convert(None)
# Send date into the dataset
c2cReturn_df['Date'] = df['Date']  

c2cReturn_df = c2cReturn_df.dropna()

# train test split
train_size = 0.95
train, test = train_test_split(c2cReturn_df, train_size=train_size, shuffle=False)

# Define the window size for the rolling average
window_size = 90

# Calculate the rolling average for the training data
train['rolling_average'] = train['WMT'].rolling(window=window_size).mean()

# Use the rolling average to forecast the training set (excluding initial NaN values)
train['rolling_average_forecast'] = train['rolling_average']

# Use the last rolling average value to forecast the test set
last_rolling_avg = train['rolling_average'].iloc[-1]
test['rolling_average_forecast'] = last_rolling_avg

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate errors for training set (drop NaN values from rolling average forecast)
train_non_nan = train.dropna(subset=['rolling_average_forecast'])
train_mae = mean_absolute_error(train_non_nan['WMT'], train_non_nan['rolling_average_forecast'])
train_mse = mean_squared_error(train_non_nan['WMT'], train_non_nan['rolling_average_forecast'])
train_mape = mean_absolute_percentage_error(train_non_nan['WMT'], train_non_nan['rolling_average_forecast'])

# Calculate errors for test set
test_mae = mean_absolute_error(test['WMT'], test['rolling_average_forecast'])
test_mse = mean_squared_error(test['WMT'], test['rolling_average_forecast'])
test_mape = mean_absolute_percentage_error(test['WMT'], test['rolling_average_forecast'])


print(f"\nTraining Mean Absolute Error (MAE): {train_mae}")
print(f"Training Mean Squared Error (MSE): {train_mse}")
print(f"Training Mean Absolute Percentage Error (MAPE): {train_mape}%")
print(f"\nTest Mean Absolute Error (MAE): {test_mae}")
print(f"Test Mean Squared Error (MSE): {test_mse}")
print(f"Test Mean Absolute Percentage Error (MAPE): {test_mape}%")

# Visualize the comparison between prediction and true data
plt.figure(figsize=(14, 7))
plt.plot(train['Date'], train['WMT'], label='Training Data')
plt.plot(train['Date'], train['rolling_average_forecast'], label='Training Rolling Average Forecast', linestyle='--')
plt.plot(test['Date'], test['WMT'], label='Test Data')
plt.plot(test['Date'], test['rolling_average_forecast'], label='Test Rolling Average Forecast', linestyle='--')
plt.legend()
plt.xlabel('Date')
plt.ylabel('WMT')
plt.title('Comparison of True Data and Rolling Average Forecast')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.show()
