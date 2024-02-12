# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:11:04 2024

@author: Hemant
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings('ignore')

from Scripts.dbConnection import *

def fetch_db_data(tickerName):
    query = f'select * from [{tickerName}]'
    df = pd.read_sql(query, cnxn('stockmarketday'))
    df['tickerName'] = tickerName
    df = fetureExtraction(df)
    return df

def fetureExtraction(df):
    df['Day'] = df['Datetime'].dt.day_name()
    df['PvClose'] = df['Close'].shift(1)
    df = df.dropna().reset_index(drop=True)
    df['OC-P/L'] = ((1-df['Open']/df['Close'])*100).round(2)   
    df['PvCC-P/L'] = ((1-df['PvClose']/df['Close'])*100).round(2)
    df['maxHigh'] = ((df['High']/df['Open']-1)*100).round(2)
    df['maxLow'] = ((df['Low']/df['Open']-1)*100).round(2)
    df['Open-PvClose'] = (df['Open']-df['PvClose']).round(2)  
    df['closeTolerance'] = df.apply(lambda row: row['OC-P/L'] - row['maxHigh'] if row['OC-P/L'] > 0 else row['OC-P/L'] - row['maxLow'] if row['OC-P/L'] < 0 else 0, axis=1)
    df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
    df = df[['Datetime', 'Day', 'tickerName', 'PvClose', 'OC-P/L', 'PvCC-P/L', 'Open-PvClose', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
    return df

def MKdayMain():
    query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
    stockSymbols = pd.read_sql(query, cnxn('stockmarketday'))['TABLE_NAME'].tolist()
    with Pool(processes=cpu_count()) as pool:
        result = list(tqdm(pool.imap(fetch_db_data, stockSymbols), total=len(stockSymbols)))
    dfMKday = pd.concat(result).reset_index(drop=True)
    result = Data_Inserting_Into_DB(dfMKday, 'stockmarket', 'MKday', 'replace')
    print(result)
    return dfMKday
    
if __name__ == "__main__":
    dfMKday = MKdayMain()   
    
