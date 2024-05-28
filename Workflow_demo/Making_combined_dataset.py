# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates

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
    'ANF',
    'KR',
    'COST',
    'CASY',
    'DECK',
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
W_H_Ret = c2cReturn_df['WMT']<c2cReturn_df['WMT'].quantile(0.5)
W_SP = H_Ret*W_H_Ret

c2cReturn_df['SP'] = W_SP
c2cReturn_df.loc[0,'SP'] = pd.NA
c2cReturn_df['SP'] = c2cReturn_df['SP'].shift(1) #shift it down by 1 so easier to input into model   

c2cReturn_df = c2cReturn_df.dropna()

# # Compute correlation matrix
# correlation_matrix = c2cReturn_df.corr()

# # Visualize correlation matrix using a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Closing Price Correlation Matrix')
# plt.show()

# Adding the SP indicator
# aa = c2cReturn_df['LOW']<c2cReturn_df['LOW'].quantile(0.5) 
# bb = c2cReturn_df['HD']>c2cReturn_df['HD'].quantile(0.75)
# SP_ind =  (aa * bb).astype(int)
# SP_est = c2cReturn_df; SP_est['SP_ind'] = SP_ind

# SP_est['c2hReturn'] = c2hReturn_df['LOW']
