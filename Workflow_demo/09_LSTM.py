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


# DATASET to Predict DECK
#Setup the date correctly
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.tz_convert(None)
# Send date into the dataset
c2cReturn_df['Date'] = df['Date']


c2cReturn_df = c2cReturn_df.dropna()


###########################################################################
df = c2cReturn_df

# Train-test split
train_size = int(len(df) * 0.95)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# Extract values
train_values = train_data['DECK'].values.reshape(-1, 1)
test_values = test_data['DECK'].values.reshape(-1, 1)

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_values)

# Scale the testing data using the same scaler
scaled_test_data = scaler.transform(test_values)



def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 1
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test = create_dataset(scaled_test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
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
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])


# Calculate MAPE
train_mape = mean_absolute_error(y_train[0], train_predict[:,0])
test_mape = mean_absolute_error(y_test[0], test_predict[:,0])

print(f'Training MAE: {train_mape}')
print(f'Testing MAE: {test_mape}')


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





