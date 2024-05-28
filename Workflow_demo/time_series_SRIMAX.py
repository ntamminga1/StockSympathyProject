# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import itertools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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


# DATASET to Predict DECK
#Setup the date correctly
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.tz_convert(None)
# Send date into the dataset
c2cReturn_df['Date'] = df['Date']

# store other companies had high return
Other_stocks = [
    'WMT',
    'ANF',
    'KR',
    'COST',
    'CASY',
    #'DECK',
    'DKS',
    'HD',
    'LOW',
    'SFM',
    'TJX',
    'TSCO',
    'WSM'
]

H_Ret = [0]*len(c2cReturn_df['Date'])
H_Ret = np.array(H_Ret)

for company in Other_stocks:
    H_Ret_t = c2cReturn_df[company]>c2cReturn_df[company].quantile(0.75)
    H_Ret_t = np.array(H_Ret_t.astype(int))
    H_Ret = np.maximum(H_Ret, H_Ret_t)
    
# Walmart has no high return
W_H_Ret = c2cReturn_df['DECK']<c2cReturn_df['DECK'].quantile(0.5)
W_SP = H_Ret*W_H_Ret

c2cReturn_df['SP'] = W_SP
c2cReturn_df.loc[0,'SP'] = pd.NA
c2cReturn_df['SP'] = c2cReturn_df['SP'].shift(1) #shift it down by 1 so easier to input into model   

c2cReturn_df = c2cReturn_df.dropna()

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(12, 6))
plot_acf(c2cReturn_df['DECK'], lags=40, title='Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(12, 6))
plot_pacf(c2cReturn_df['DECK'], lags=40, title='Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

# train test split
train_size = 0.95
train, test = train_test_split(c2cReturn_df, train_size=train_size, shuffle=False)

# # Define the range of values for p, d, q
# p_values = range(0, 2)
# d_values = range(0, 2)
# q_values = range(0, 2)

# # Define the range of values for P, D, Q, and m (seasonal parameters)
# seasonal_p_values = range(0, 2)
# seasonal_d_values = range(0, 2)
# seasonal_q_values = range(0, 2)
# m_values = [10]  

# # Generate all possible combinations of p, d, q, P, D, Q, and m
# parameter_grid = list(itertools.product(p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, m_values))


# best_aic = float("inf")
# best_params = None

# # Iterate over all combinations of p, d, q, P, D, Q, and m
# for params in parameter_grid:
#     try:
#         model = SARIMAX(train['WMT'], order=(params[0], params[1], params[2]), seasonal_order=(params[3], params[4], params[5], params[6]))
#         fit_model = model.fit()
#         aic = fit_model.aic
#         if aic < best_aic:
#             best_aic = aic
#             best_params = params
#     except:
#         continue

# print("Best AIC:", best_aic)
# print("Best Parameters (p, d, q, P, D, Q, m):", best_params)


# Define the order of the ARIMA model (p, d, q)
p = 1  # AR parameter
d = 0  # I(d) parameter (degree of differencing)
q = 0  # MA parameter

# Fit ARIMA model
model = SARIMAX(train['DECK'], order=(p, d, q), seasonal_order=(0,0,0,10), exog=train['SP'])
fit_model = model.fit()

# Use the fitted model to forecast both the training and test sets
train['arima_forecast'] = fit_model.fittedvalues
test['arima_forecast'] = fit_model.forecast(steps=len(test),exog=test['SP'])

# Forecast the training and test sets
# train['sarimax_forecast'] = fit_model.predict(start=train.index[0], end=train.index[-1], exog=train['SP'])
# test['sarimax_forecast'] = fit_model.predict(start=test.index[0], end=test.index[-1], exog=test['SP'])


test = test[:-2] # I don't know why, the last two rows are always nans

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate errors for training set
train_mae = mean_absolute_error(train['DECK'], train['arima_forecast'])
train_mse = mean_squared_error(train['DECK'], train['arima_forecast'])
train_mape = mean_absolute_percentage_error(train['WMT'], train['arima_forecast'])

# Calculate errors for test set
test_mae = mean_absolute_error(test['DECK'], test['arima_forecast'])
test_mse = mean_squared_error(test['DECK'], test['arima_forecast'])
test_mape = mean_absolute_percentage_error(test['WMT'], test['arima_forecast'])

print(f"\nTraining Mean Absolute Error (MAE): {train_mae}")
print(f"Training Mean Squared Error (MSE): {train_mse}")
#print(f"Training Mean Absolute Percentage Error (MAPE): {train_mape}%")
print(f"\nTest Mean Absolute Error (MAE): {test_mae}")
print(f"Test Mean Squared Error (MSE): {test_mse}")
#print(f"Test Mean Absolute Percentage Error (MAPE): {test_mape}%")

# Visualize the comparison between prediction and true data
plt.figure(figsize=(14, 7))
plt.plot(train['Date'], train['DECK'], label='Training Data')
plt.plot(train['Date'], train['arima_forecast'], label='Training ARIMA Forecast', linestyle='--')
plt.plot(test['Date'], test['DECK'], label='Test Data')
plt.plot(test['Date'], test['arima_forecast'], label='Test ARIMA Forecast', linestyle='--')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Comparison of True Data and ARIMA Forecast')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.show()