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
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn


class StockDataDownloader:
    def __init__(self):
        self.cpu_count = int(cpu_count() * 0.8)
        self.current_date = datetime.now()
        self.db_name_day = 'mkdaymaster'
        self.db_name_interval = 'mkintervalmaster'
        self.intial_start_date = '2017-10-01'
        self.stock_symbols_file = './Data/stokeSymbol.csv'
   
    def validation(self):
        create_database(self.db_name_day)
        create_database(self.db_name_interval)
        
    def StockDataDownloaderMain(self):
        stock_symbols = pd.read_csv(self.stock_symbols_file)['SYMBOL \n'][1:].unique()
        result = self.yf_download_main(stock_symbols)
        return result
        
    def yf_download_main(self, stock_symbols):
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.stock_data_download, stock_symbols), total=len(stock_symbols), desc='Stock Data Downloader'))
        result = self.update_table(result, 'append')
        return result
            
    def stock_data_download(self, ticker_name):
        query_day = f'select max(Datetime) from [{ticker_name}]'
        query_interval = f'select max(Datetime) from [{ticker_name}]'
        try:
            start_date = (pd.read_sql(query_day, cnxn(self.db_name_day)).iloc[0].iloc[0] + timedelta(days=1)).strftime('%Y-%m-%d')
        except:
            start_date = self.intial_start_date
        start_date_interval = (pd.read_sql(query_interval, cnxn(self.db_name_interval)).iloc[0].iloc[0] + timedelta(days=1)).strftime('%Y-%m-%d')
        period = self.period_decision_maker(start_date, start_date_interval)
        df, df_interval = self.yf_download(ticker_name, period)
        df = df[df['Datetime'] >= period['start']].reset_index(drop=True)
        result = {
            self.db_name_day: {
                'tableName': ticker_name,
                'Dataframe': df
                },
            self.db_name_interval: {
                'tableName': ticker_name,
                'Dataframe': df_interval
                }
            }
        return result
    
    def period_decision_maker(self, start_date, start_date_interval):
        period = {
            'start': start_date,
            'end': (self.current_date + timedelta(days=1)).strftime('%Y-%m-%d')
        }
        period_interval = self.period_decision_maker_for_interval_download(start_date_interval)
        return {**period, **period_interval}
    
    def period_decision_maker_for_interval_download(self, start_date_interval):
        days_ago = (self.current_date - timedelta(days=59)).strftime('%Y-%m-%d')
        if start_date_interval < days_ago:
            start_date_interval = days_ago
        period = {
            'startInterval': start_date_interval,
            'endInterval': (self.current_date + timedelta(days=1)).strftime('%Y-%m-%d')
        }
        return period
    
    def yf_download(self, ticker_name, period):
        yf_ticker = yf.Ticker(f'{ticker_name}.NS')
        df = yf_ticker.history(start=period['start'], end=period['end']).dropna().reset_index()
        df_interval = yf_ticker.history(start=period['startInterval'], end=period['endInterval'], interval='5m').dropna().reset_index()
        df = self.yf_download_cleaning(df)
        df_interval = self.yf_download_cleaning(df_interval)
        return df, df_interval
    
    def yf_download_cleaning(self, df):
        if df.empty:
            return df
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        return df
    
    def update_table(self, result, insertMethod):
        result = [Data_Inserting_Into_DB(subdct.get('Dataframe'), dbName, subdct.get('tableName'), insertMethod) if not subdct.get('Dataframe').empty else {'dbName': dbName, subdct.get('tableName'): 'Unsuccessful Empty DataFrame'} for dct in tqdm(result, desc='Stock Data Downloader Update Table') for dbName, subdct in dct.items()]
        return result

def JobStockDataDownloader():
    downloader = StockDataDownloader()
    downloader.validation()
    result = downloader.StockDataDownloaderMain()
    return result
    
if __name__ == "__main__":
    downloader = StockDataDownloader()
    downloader.validation()
    result = downloader.StockDataDownloaderMain()





