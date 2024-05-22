# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates

#df = pd.read_csv('KR.csv',index_col=0)
df = pd.read_csv('LOW.csv')

df['Date'] = pd.to_datetime(df['Date'], utc=True)

# Convert 'date' column to datetime with timezone information
df['Date'] = pd.to_datetime(df['Date'])

# Option 1: Convert to timezone-naive datetime (removes timezone information)
df['Date'] = df['Date'].dt.tz_convert(None)

df['Previous_close'] = df['Close'].shift(1) 

df['Return'] = df['High']/df['Previous_close']-1

df['Return'].describe()

df['Return'].plot(kind='hist', title='Histogram of Returns', bins=30, edgecolor='red', color='skyblue',
                  alpha = 0.3)
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Date', y='Return')

plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.gcf().autofmt_xdate()

plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Return Over Time')
plt.grid(True)
plt.show()