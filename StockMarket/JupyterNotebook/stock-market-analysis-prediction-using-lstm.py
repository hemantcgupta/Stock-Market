# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:30:37 2024

@author: Hemant
"""


# =============================================================================
# 
# =============================================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def yfDownload(tickerName, period, interval):
    yfTicker = yf.Ticker(tickerName)
    df = yfTicker.history(period=period).dropna().reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = df['Date'].dt.date
    df['Symbol'] = tickerName
    return df
tickerNames = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
df = pd.concat([yfDownload(ticker, '1y', '5m') for ticker in tickerNames]).reset_index(drop=True)
df.tail(10)

# Summary Stats
for ticker in tickerNames:
    print(f'######### {ticker} #########\n', df[df['Symbol'] == ticker].describe())

# General info
for ticker in tickerNames:
    print(f'######### {ticker} #########\n', df[df['Symbol'] == ticker].info())

# =============================================================================
# Closing Price
# The closing price is the last price at which the stock is traded during the regular trading day. A stock’s closing price is the standard benchmark used by investors to track its performance over time.
# =============================================================================
# Let's see a historical view of the closing price
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, dfTicker in enumerate(df.groupby('Symbol'), 1):
    ticker, dfTicker = dfTicker
    plt.subplot(2, 2, i)
    dfTicker['Close'].plot()
    plt.ylabel('Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {ticker}")
    
plt.tight_layout()

for i, tt in enumerate(df.groupby('Symbol'), 1):
    print(tt)