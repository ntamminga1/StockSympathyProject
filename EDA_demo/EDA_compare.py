

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates

WMT = pd.read_csv('WMT.csv')
TGT = pd.read_csv('TGT.csv')
KR = pd.read_csv('KR.csv')

close_comb = pd.concat([WMT['Close'], TGT['Close'], KR['Close']], axis=1)
close_comb.columns = ['WMT','TGT','KR']

# Compute correlation matrix
correlation_matrix = close_comb.corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Closing Price Correlation Matrix')
plt.show()
