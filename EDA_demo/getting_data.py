

import yfinance as yf
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

power = yf.Ticker("VSCO")
# df = power.history(start="2022-01-01", end='2024-05-03')
df = power.history(start="2020-01-01", end=None)
df.head()

df.to_csv('VSCO.csv')

#readback = pd.read_csv('WMT.csv',index_col=0)