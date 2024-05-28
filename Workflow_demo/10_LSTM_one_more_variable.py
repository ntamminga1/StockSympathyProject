# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import tensorflow as tf
import random

# Set the random seed for NumPy
np.random.seed(42)

# Set the random seed for Python's built-in random module
random.seed(42)

# Set the random seed for TensorFlow
tf.random.set_seed(42)

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

################################################################################################
df = c2cReturn_df

# train test split
train_size = 0.95
train, test = train_test_split(c2cReturn_df, train_size=train_size, shuffle=False)

# Train-test split
train_size = int(len(df) * 0.95)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# Extract values for the target column and the additional feature
train_values = train_data[['DECK', 'SP']].values
test_values = test_data[['DECK', 'SP']].values

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_values)

# Scale the testing data using the same scaler
scaled_test_data = scaler.transform(test_values)

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        X.append(a)
        y.append(dataset[i + time_step, 0])  # Assuming the target variable is the first column
    return np.array(X), np.array(y)

time_step = 1
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test = create_dataset(scaled_test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=25, epochs=100)

# Predictions for training set
train_predict = model.predict(X_train)
# Predictions for testing set
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(np.hstack((train_predict, np.zeros((train_predict.shape[0], 1)))))
train_predict = train_predict[:,0]

y_train = scaler.inverse_transform(np.hstack((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 1)))))
y_train = y_train[:,0]

test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], 1)))))
test_predict = test_predict[:,0]

y_test = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1)))))
y_test = y_test[:,0]

# Calculate MAPE
train_mae = mean_absolute_error(y_train, train_predict)
test_mae = mean_absolute_error(y_test, test_predict)

print(f'Training MAPE: {train_mae}')
print(f'Testing MAPE: {test_mae}')

# Plotting
plt.figure(figsize=(12, 6))

# Plot original data
plt.plot(df['Date'], df['DECK'], label='Original Data')

# Plot predictions with dates
plt.plot(df['Date'][time_step:train_size-1], train_predict, label='Train Prediction')
plt.plot(df['Date'][train_size+time_step+1:], test_predict, label='Test Prediction')

plt.xlabel('Date')
plt.ylabel('DECK')
plt.legend()
plt.show()
