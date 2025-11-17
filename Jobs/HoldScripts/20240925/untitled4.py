# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 01:17:54 2024

@author: Hemant
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from Scripts.dbConnection import cnxn, Data_Inserting_Into_DB, create_database

pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 25)
pd.set_option('expand_frame_repr', True)

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.db_mkanalyzer = 'mkanalyzer'
        self.db_mkintervalmaster = 'mkintervalmaster'
        self.cnxn = cnxn(self.db_mkintervalmaster)
        self.df15 = None
        self.df75 = None
        self.dfI = None
        self.results = []
    
    def fetch_data(self):
        # Query for 75-minute interval data
        query75 = f'''
        WITH TimeFilteredData AS (
            SELECT 
                [Datetime],
                CAST([Datetime] AS DATE) AS TradeDate,
                [High], 
                [Low], 
                [Open], 
                [Close]
            FROM 
                [{self.ticker}]
            WHERE 
                (DATEPART(HOUR, [Datetime]) = 14 AND DATEPART(MINUTE, [Datetime]) BETWEEN 15 AND 59)
                OR 
                (DATEPART(HOUR, [Datetime]) = 15 AND DATEPART(MINUTE, [Datetime]) BETWEEN 0 AND 25)
        ),
        AggregatedData AS (
            SELECT
                TradeDate,
                MAX(High) AS MaxHigh,
                MIN(Low) AS MinLow,
                MIN([Datetime]) AS MinDatetime,
                MAX([Datetime]) AS MaxDatetime
            FROM 
                TimeFilteredData
            GROUP BY 
                TradeDate
        ),
        OpenCloseValues AS (
            SELECT
                tfd.TradeDate,
                ad.MaxHigh,
                ad.MinLow,
                MIN(CASE WHEN tfd.[Datetime] = ad.MinDatetime THEN tfd.[Open] END) AS OpenValue,
                MIN(CASE WHEN tfd.[Datetime] = ad.MaxDatetime THEN tfd.[Close] END) AS CloseValue
            FROM 
                TimeFilteredData tfd
            JOIN 
                AggregatedData ad 
            ON 
                tfd.TradeDate = ad.TradeDate
            GROUP BY 
                tfd.TradeDate, ad.MaxHigh, ad.MinLow, ad.MinDatetime, ad.MaxDatetime
        )
        SELECT
            TradeDate AS Date,
            OpenValue AS Open75,
            MaxHigh as High75,
            MinLow as Low75,
            CloseValue AS Close75
        FROM 
            OpenCloseValues
        ORDER BY 
            TradeDate DESC;
        '''
        self.df75 = pd.read_sql(query75, self.cnxn)
        self.df75['Date75'] = self.df75['Date']
        self.df75['Date'] = self.df75['Date'].shift(1)
        
        # Query for 15-minute interval data
        query15 = f'''
        WITH TimeFilteredData AS (
            SELECT 
                [Datetime],
                CAST([Datetime] AS DATE) AS TradeDate,
                [High], 
                [Low], 
                [Open], 
                [Close]
            FROM 
                [{self.ticker}]
            WHERE 
                (DATEPART(HOUR, [Datetime]) = 9 AND DATEPART(MINUTE, [Datetime]) BETWEEN 15 AND 25)
        ),
        AggregatedData AS (
            SELECT
                TradeDate,
                MAX(High) AS MaxHigh,
                MIN(Low) AS MinLow,
                MIN([Datetime]) AS MinDatetime,
                MAX([Datetime]) AS MaxDatetime
            FROM 
                TimeFilteredData
            GROUP BY 
                TradeDate
        ),
        OpenCloseValues AS (
            SELECT
                tfd.TradeDate,
                ad.MaxHigh,
                ad.MinLow,
                MIN(CASE WHEN tfd.[Datetime] = ad.MinDatetime THEN tfd.[Open] END) AS OpenValue,
                MIN(CASE WHEN tfd.[Datetime] = ad.MaxDatetime THEN tfd.[Close] END) AS CloseValue
            FROM 
                TimeFilteredData tfd
            JOIN 
                AggregatedData ad 
            ON 
                tfd.TradeDate = ad.TradeDate
            GROUP BY 
                tfd.TradeDate, ad.MaxHigh, ad.MinLow, ad.MinDatetime, ad.MaxDatetime
        )
        SELECT
            TradeDate AS Date,
            OpenValue AS Open15,
            MaxHigh as High15,
            MinLow as Low15,
            CloseValue AS Close15
        FROM 
            OpenCloseValues
        ORDER BY 
            TradeDate DESC;
        '''
        self.df15 = pd.read_sql(query15, self.cnxn)
        self.df = pd.merge(self.df15, self.df75, on='Date', how='left', suffixes=('_15', '_75')).dropna().rename(columns={'Date': 'Date15'})
        self.df = self.df[['Date15', 'Date75', 'Open15', 'High15', 'Low15', 'Close15', 'Open75', 'High75', 'Low75', 'Close75']]
    
    def calculate_points(self, row):
        isBreckPrevEntry = 0
        close_point = round(row['Close75'] * (1 + 0.005) / 0.05) * 0.05
        entry = round(row['High75'] * (1 + (0.007 if row['High75'] > close_point else 0.012)) / 0.05) * 0.05
        
        if entry < row['High15']:  
            isBreckPrevEntry = 1
            close_point = round(row['Close15'] * (1 + 0.005) / 0.05) * 0.05
            entry = round(row['High15'] * (1 + (0.007 if row['High15'] > close_point else 0.012)) / 0.05) * 0.05
            target = round(entry * (1 + 0.0125) / 0.05) * 0.05
            stoploss = round(max(entry * (1 - (0.005 if row['High15'] > close_point else 0.007)),
                                  row['Low15'] * (1 - (0.007 if row['High15'] > close_point else 0.012))) / 0.05) * 0.05
        else:
            target = round(entry * (1 + 0.0125) / 0.05) * 0.05
            stoploss = round(max(entry * (1 - (0.005 if row['High75'] > close_point else 0.007)),
                                  row['Low75'] * (1 - (0.007 if row['High75'] > close_point else 0.012))) / 0.05) * 0.05
        return pd.Series([close_point, entry, target, stoploss, isBreckPrevEntry], index=['ClosePoint', 'EntryPoint', 'TargetPoint', 'SLPoint', 'isBreckPrevEntry'])
    
    def apply_calculations(self):
        self.df[['ClosePoint', 'EntryPoint', 'TargetPoint', 'SLPoint', 'isBreckPrevEntry']] = self.df.apply(self.calculate_points, axis=1)
        self.df['isBreckPrevEntry'] = self.df['isBreckPrevEntry'].astype(int)
    
    def fetch_intraday_data(self):
        query = f'''
        SELECT cast(Datetime as Date) as Date, Datetime, 
        [Open], [High], [Low], [Close]
        FROM [{self.ticker}]
        ORDER BY Datetime
        '''
        self.dfI = pd.read_sql(query, self.cnxn)
        self.dfI = self.dfI[(self.dfI['Datetime'].dt.time >= pd.to_datetime('09:30:00').time()) & (self.dfI['Datetime'].dt.time <= pd.to_datetime('15:15:00').time())]
        self.dfI = self.dfI.sort_values(by='Datetime').reset_index(drop=True)
    
    def get_execution_time(self, row):
        dfE = self.dfI[self.dfI['Date'] == row['Date15']].reset_index(drop=True)
        EntryTime, TargetTime, SLTime = None, None, None
        Close315 = dfE['Open'].iloc[-1] if not dfE['Open'].empty else None

        for idx, df_row in dfE.iterrows():
            if EntryTime is None and df_row['Low'] <= row['EntryPoint'] <= df_row['High']:
                EntryTime = df_row['Datetime']
            if EntryTime:
                if TargetTime is None and df_row['Low'] <= row['TargetPoint'] <= df_row['High']:
                    TargetTime = df_row['Datetime']
                if SLTime is None and df_row['Low'] <= row['SLPoint'] <= df_row['High']:
                    SLTime = df_row['Datetime']
                if TargetTime and SLTime:
                    break
        return pd.Series([EntryTime, TargetTime, SLTime, Close315], index=['EntryTime', 'TargetTime', 'SLTime', 'Close315'])
    
    def apply_execution_times(self):
        self.df[['EntryTime', 'TargetTime', 'SLTime', 'Close315']] = self.df.apply(self.get_execution_time, axis=1)
        self.df['Close315'] = self.df['Close315'].astype(float)
        # self.df.drop(columns=['EntryPoint', 'TargetPoint', 'SLPoint', 'isBreckPrevEntry', 'ClosePoint'], inplace=True)
    
    def process(self):
        try:
            self.fetch_data()
            self.apply_calculations()
            self.fetch_intraday_data()
            self.apply_execution_times()
            self.df['tickerName'] = self.ticker
            return self.df
        except:
            return None

def main(tickerName):
    analyzer = StockAnalyzer(tickerName)
    df = analyzer.process()
    return df

class VAR:
    db_name_mkprediction = 'mkprediction'
    
def validation():
    create_database(VAR.db_name_mkprediction)
    
def update_table(self, result, insertMethod):
    result = [Data_Inserting_Into_DB(subdct.get('Dataframe'), dbName, subdct.get('tableName'), insertMethod) if not subdct.get('Dataframe').empty else {'dbName': dbName, subdct.get('tableName'): 'Unsuccessful Empty DataFrame'} for dct in tqdm(result, desc='Stock Data Downloader Update Table') for dbName, subdct in dct.items()]
    return result
    
if __name__ == '__main__':  
    query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
    stockSymbols = pd.read_sql(query, cnxn('mkintervalmaster'))['TABLE_NAME'].tolist()
    with Pool(processes=int(cpu_count() * 0.8)) as pool:
        result = list(tqdm(pool.imap(main, stockSymbols), total=len(stockSymbols), desc='Fetch'))
    df = pd.concat(result).reset_index(drop=True)
    df['gotProfit'] = np.where((df['EntryTime'].notnull()) & (df['TargetTime'].notnull()), 1, 0)
    df['gotEntry'] = np.where((df['EntryTime'].notnull()), 1, 0)
    validation()
    Data_Inserting_Into_DB(df, VAR.db_name_mkprediction, 'algo7515', 'replace')
    
    # dfT = df[pd.to_datetime(df['Date15']) >= '2024-08-20']
    # dfT['filter'] = np.where((dfT['EntryTime'].notnull()) & (dfT['TargetTime'].notnull()), 1, 0)
    # dfT.groupby('tickerName') ['filter'].sum().reset_index().sort_values(by='filter', ascending=False).merge(
    #     dfT[(pd.to_datetime(dfT['Date15']) == '2024-09-20') & (dfT['filter'] == 1)][['Date15', 'tickerName']],
    #     how='left',
    #     on='tickerName'
    # ).sort_values(by=['filter', 'Date15'], ascending=[False, False])
    # dfT[dfT['tickerName'] == 'FDC'].head(25)