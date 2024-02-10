# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:41:36 2024

@author: Hemant
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from Scripts.dbConnection import *


def period_decision_maker(startDate, startDateInterval):
    current_date = datetime.now()
    period = {
        'start': startDate,
        'end': current_date.strftime('%Y-%m-%d'),
     }
    periodInterval = period_decision_maker_for_interval_download(startDateInterval, current_date)
    return {**period, **periodInterval}

def period_decision_maker_for_interval_download(startDateInterval, current_date):
    days_ago = (current_date - timedelta(days=59)).strftime('%Y-%m-%d')
    if startDateInterval < days_ago:
        startDateInterval = days_ago
    period = {
        'startInterval': startDateInterval,
        'endInterval': current_date.strftime('%Y-%m-%d'),
     }
    return period
    
def yfDownload(tickerName, period):
    yfTicker = yf.Ticker(f'{tickerName}.NS')
    df = yfTicker.history(start=period['start'], end=period['end']).dropna().reset_index()
    dfInterval = yfTicker.history(start=period['startInterval'], end=period['endInterval'], interval='5m').dropna().reset_index()
    df = ydDownloadClenning(df)
    dfInterval = ydDownloadClenning(dfInterval)
    return df, dfInterval

def ydDownloadClenning(df):
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    return df
    
def stockDataDownload(tickerName):
    query = f'select max(Datetime) from {tickerName}'
    startDate = pd.read_sql(query, cnxn('stockmarketday')).iloc[0].iloc[0].strftime('%Y-%m-%d')
    startDateInterval = pd.read_sql(query, cnxn('stockmarketinterval')).iloc[0].iloc[0].strftime('%Y-%m-%d')
    period = period_decision_maker(startDate, startDateInterval)
    df, dfInterval = yfDownload(tickerName, period)
    result = Data_Inserting_Into_DB(df, 'stockmarketday', tickerName)
    resultInterval = Data_Inserting_Into_DB(dfInterval, 'stockmarketinterval', tickerName)
    return {'day': result, 'Interval': resultInterval}

def UpdateDB():
    stockSymbols = pd.read_csv(r'./Data/stokeSymbol.csv')['SYMBOL \n'][1:].unique()
    with Pool(processes=cpu_count()) as pool:
        result = list(tqdm(pool.imap(stockDataDownload, stockSymbols), total=len(stockSymbols)))
        
        
if __name__ == "__main__":
    UpdateDB()
   