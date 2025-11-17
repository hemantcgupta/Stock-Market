# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 22:13:33 2024

@author: heman
"""
import time
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from Scripts.dbConnection import cnxn, Data_Inserting_Into_DB
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
tqdm.pandas()
import yfinance as yf

class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkdaymaster = 'mkdaymaster'
    db_mkintervalmaster = 'mkintervalmaster'
    table_name_mkDayfeature = 'mkDayfeature'
    isLive = True

    

# =============================================================================
# Fetch Data 
# =============================================================================
class fetchData:
    def __init__(self, **kwargs):
        self.tickerNameList = kwargs.get('tickerNameList', [])
        self.startDate = kwargs.get('startDate', None)
        self.endDate = kwargs.get('endDate', None)
        self.currentDate = kwargs.get('currentDate', None)
        if kwargs.get('currentDate', None):
            self.period = {
                'startInterval': kwargs.get('currentDate'),
                'endInterval': (datetime.strptime(kwargs.get('currentDate'), '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            }

    def fetchDayLevelData(self):
        query = 'SELECT Datetime, tickerName, [PvClose] FROM mkDayfeature'
        conditions = []
        if self.tickerNameList:
            tickers = ', '.join(f"'{ticker}'" for ticker in self.tickerNameList)
            conditions.append(f"tickerName IN ({tickers})")
        if self.startDate:
            conditions.append(f"Datetime >= '{self.startDate}'")
        if self.endDate:
            conditions.append(f"Datetime <= '{self.endDate}'")
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))
        self.tickerNameList = list(df['tickerName'].unique())
        self.endDate = df['Datetime'].max().strftime('%Y-%m-%d')
        return df
    
    def fetch5MinLevelData(self, tickerName):
        query = f'''SELECT TOP 750 * FROM [{tickerName}]
        WHERE Datetime <= '{self.currentDate}' ORDER BY Datetime DESC
        '''
        df = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster))
        df['tickerName'] = tickerName
        return df
    
    @staticmethod
    def TenDayAvgVolume(dfInterval):
        dfInterval['Date'] = pd.to_datetime(dfInterval['Datetime']).dt.date
        dfInterval = dfInterval.groupby(['tickerName']) ['Volume'].mean().reset_index()
        return dfInterval
        
    def fetchTodayData(self, tickerName):
        if not VAR.isLive:
            query = f'''SELECT * FROM [{tickerName}]
            WHERE cast(Datetime as date) = '{self.currentDate}' ORDER BY Datetime DESC
            '''
            df = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster))
            df['tickerName'] = tickerName
        if VAR.isLive:
            if isinstance(tickerName, list):
                tickerName = [item + '.NS' for item in tickerName]
                df = yf.download(tickers=" ".join(tickerName), start=self.period['startInterval'], end=self.period['endInterval'], interval="5m", group_by="ticker")
                df = self.transformData(df)
            else:
                df = self.yf_download(tickerName, self.period)
        return df
    

    def transformData(self, df):
        df = df.stack(level=0).reset_index()
        df.columns = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert('Asia/Kolkata')
        df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        desired_columns = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        df = df[desired_columns]
        df.rename(columns={'Ticker': 'tickerName'}, inplace=True)
        df['tickerName'] = df['tickerName'].str.replace('.NS', '')
        return df
    
    def yf_download(self, tickerName, period):
        if tickerName[0] == '^':
            tickerName = f'{tickerName}' 
        else:
            tickerName = f'{tickerName}.NS'
        yf_ticker = yf.Ticker(tickerName)
        df_interval = yf_ticker.history(start=period['startInterval'], end=period['endInterval'], interval='5m').dropna().reset_index()
        df_interval = self.yf_download_cleaning(df_interval)
        return df_interval

    def yf_download_cleaning(self, df):
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        if df.empty:
            return df
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        return df

    @staticmethod
    def volumeSpike(dfToday, dfTenDayAvgVolume):
        df = pd.merge(dfToday, dfTenDayAvgVolume, how='left', on=['tickerName'])
        df['volumeSpike'] = df.apply(
            lambda row: round(((row['Volume_x'] - row['Volume_y']) / row['Volume_y']) * 100, 2)
            if row['Volume_y'] != 0 else 0,
            axis=1
        )
        df.rename(columns={'Volume_x': 'Volume', 'Volume_y': '10DayAvgVolume'}, inplace=True)
        return df
    
    @staticmethod
    def priceAction(df, dfDay):
        df = pd.merge(df, dfDay[['tickerName', 'PvClose']], how='left', on=['tickerName'])
        df['priceAction'] = df.apply(
            lambda row: round(((row['Close'] - row['PvClose']) / row['PvClose']) * 100, 2)
            if row['PvClose'] != 0 else 0,
            axis=1
        )
        return df
    
    @staticmethod
    def calATR(df, dfInterval):
        dfATRC = df[['Datetime', 'tickerName', 'High', 'Low', 'Close']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfATRP = dfInterval[['Datetime', 'tickerName', 'High', 'Low', 'Close']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfATRP = dfATRP.groupby('tickerName', group_keys=False).tail(15)
        dfATR = pd.concat([dfATRP, dfATRC]).sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfATR['PvClose'] = dfATR.groupby('tickerName')['Close'].shift(1)
        dfATR['trueRange'] = pd.concat([(dfATR['High'] - dfATR['Low']), (dfATR['High'] - dfATR['PvClose']).abs(), (dfATR['Low'] - dfATR['PvClose']).abs()], axis=1).max(axis=1).round(2)
        dfATR['ATR'] = dfATR.groupby('tickerName')['trueRange'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().round(2))
        dfATR['ATRStatus'] = np.where(dfATR['ATR'] > (dfATR['Close'] * 0.015), 'QUALIFIED', 'NOT QUALIFIED')
        df = pd.merge(df, dfATR[['Datetime', 'tickerName', 'trueRange', 'ATR', 'ATRStatus']], how='left', on=['Datetime', 'tickerName'])
        return df
    
    @staticmethod
    def calVWAP(df, dfInterval):
        dfVWAPC = df[['Datetime', 'tickerName', 'Close', 'Volume']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfVWAPP = dfInterval[['Datetime', 'tickerName', 'Close', 'Volume']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfVWAPP = dfVWAPP.groupby('tickerName', group_keys=False).tail(75)
        dfVWAP = pd.concat([dfVWAPP, dfVWAPC]).sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfVWAP['VWAP'] = dfVWAP.apply(lambda row: round((row['Volume'] * row['Close']) / row['Volume'], 2) if row['Volume'] != 0 else 0, axis=1)
        dfVWAP['VWAP'] = dfVWAP.groupby('tickerName')['VWAP'].transform(lambda x: x.rolling(window=75, min_periods=1).mean().round(2))
        dfVWAP['VWAPStatus'] = np.where(dfVWAP['VWAP'] < dfVWAP['Close'], 'BULLISH', 'NOT BULLISH')
        df = pd.merge(df, dfVWAP[['Datetime', 'tickerName', 'VWAP', 'VWAPStatus']], how='left', on=['Datetime', 'tickerName'])
        return df
    
    @staticmethod
    def BuyANDSellSignal(df, dfInterval):
        dfBSC = df[['Datetime', 'tickerName', 'High', 'Low']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfBSP = dfInterval[['Datetime', 'tickerName', 'High', 'Low']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfBSP = dfBSP.groupby('tickerName', group_keys=False).tail(5)
        dfBS = pd.concat([dfBSP, dfBSC]).sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfBS[['Max5Candle', 'Min5Candle']] = dfBS.groupby('tickerName').apply(
            lambda group: pd.DataFrame({
                'Max5Candle': group['High'].rolling(window=5, min_periods=1).max().shift(1),
                'Min5Candle': group['Low'].rolling(window=5, min_periods=1).min().shift(1)
            })
        ).reset_index(drop=True)
        dfBS['BuySignal'] = np.where(dfBS['High'] > dfBS['Max5Candle'], 'Buy Signal', 'Hold Buy')
        dfBS['SellSignal'] = np.where(dfBS['Low'] < dfBS['Min5Candle'], 'Sell Signal', 'Hold Sell')
        df = pd.merge(df, dfBS[['Datetime', 'tickerName', 'Max5Candle', 'Min5Candle', 'BuySignal', 'SellSignal']], how='left', on=['Datetime', 'tickerName'])
        return df
    
    @staticmethod
    def calRSI(df, dfInterval):
        dfRSIC = df[['Datetime', 'tickerName', 'Open', 'Close']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfRSIP = dfInterval[['Datetime', 'tickerName', 'Open', 'Close']].sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfRSIP = dfRSIP.groupby('tickerName', group_keys=False).tail(15)
        dfRSI = pd.concat([dfRSIP, dfRSIC]).sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
        dfRSI['AG'], dfRSI['AL'] = dfRSI['Close'].sub(dfRSI['Open']).clip(lower=0), dfRSI['Close'].sub(dfRSI['Open']).clip(upper=0).abs()
        dfRSI[['AG', 'AL']] = dfRSI.groupby('tickerName').apply(
            lambda group: pd.DataFrame({
                'AG': group['AG'].rolling(window=14, min_periods=1).mean().shift(1),
                'AL': group['AL'].rolling(window=14, min_periods=1).mean().shift(1)
            })
        ).reset_index(drop=True)
        dfRSI['RSI'] = round(100 - (100/(1+(dfRSI['AG']/dfRSI['AL']))), 2)
        conditions = [dfRSI['RSI'] > 70, dfRSI['RSI'] < 30, (dfRSI['RSI'] >= 40) & (dfRSI['RSI'] <= 70)]
        choices = ['Overbought', 'Oversold', 'Balanced']
        dfRSI['RSIStatus'] = np.select(conditions, choices, default='Hold')
        df = pd.merge(df, dfRSI[['Datetime', 'tickerName', 'RSI', 'RSIStatus']], how='left', on=['Datetime', 'tickerName'])
        return df
    
        
def funcTimeTaken(startTime):
    endTime = time.time()
    timeTaken = endTime - startTime
    seconds = int(timeTaken)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02}:{remaining_seconds:02}"
        

def main(startDate, endDate, currentDate):
    startTime = time.time()
    query = 'SELECT distinct(tickerName) as tickerName FROM mkDayfeature'
    tickerNameList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['tickerName'].tolist()
    obj = fetchData(tickerNameList=tickerNameList, startDate=startDate, endDate=endDate, currentDate=currentDate)
    dfDay = obj.fetchDayLevelData()
    # dfInterval = pd.concat([obj.fetch5MinLevelData(tickerName) for tickerName in tqdm(obj.tickerNameList)]).reset_index(drop=True)
    with Pool(max(1, int(cpu_count() * 0.8))) as pool:
        results = list(tqdm(pool.imap(obj.fetch5MinLevelData, obj.tickerNameList), total=len(obj.tickerNameList), desc='Previous 10 Day Data'))
    dfInterval = pd.concat(results).reset_index(drop=True)
    dfInterval = dfInterval.sort_values(by=['tickerName', 'Datetime']).reset_index(drop=True)
    dfTenDayAvgVolume = obj.TenDayAvgVolume(dfInterval)
    tickerNameListChunks = [obj.tickerNameList[i:i+100] for i in range(0, len(obj.tickerNameList), 800)]
    # dfToday = pd.concat([obj.fetchTodayData(tickerNameList) for tickerNameList in tqdm(tickerNameListChunks)]).reset_index(drop=True)
    with Pool(max(1, int(cpu_count() * 0.8))) as pool:
        # results = list(tqdm(pool.imap(obj.fetchTodayData, obj.tickerNameList), total=len(obj.tickerNameList), desc='Today Data'))
        results = list(tqdm(pool.imap(obj.fetchTodayData, tickerNameListChunks), total=len(tickerNameListChunks), desc='Today Data'))
    dfToday = pd.concat(results).reset_index(drop=True)
    df = obj.volumeSpike(dfToday, dfTenDayAvgVolume)
    df = obj.priceAction(df, dfDay)
    df = obj.calATR(df, dfInterval)
    df = obj.calVWAP(df, dfInterval)
    df = obj.BuyANDSellSignal(df, dfInterval)
    df = obj.calRSI(df, dfInterval)
    timeTaken = funcTimeTaken(startTime)
    print(f"Run Time: {timeTaken}")
    return df
   
def mainJob(kwargs):
    df = main(startDate=kwargs.get('previousDate'), endDate=kwargs.get('previousDate'), currentDate=kwargs.get('currentDate'))    
    return df


if __name__ == "__main__":  
    query = 'SELECT distinct(Datetime) as Datetime FROM mkDayfeature order by Datetime desc'
    dateList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Datetime'].dt.date.astype(str).unique()[:60]
    dateList = [{"previousDate": dateList[i+1], "currentDate": dateList[i]} for i in range(len(dateList)-1)]
    df = pd.concat([mainJob(kwargs) for kwargs in tqdm(dateList)]).reset_index(drop=True)
    # df.to_excel(r'C:\Users\heman\Desktop\tt.xlsx', index=False)
    
    
# =============================================================================
#     
# =============================================================================

# if __name__ == "__main__":  
#     currentDatetime = '2024-12-30 15:25:00'
#     df = main(startDate='2024-12-27', endDate='2024-12-27', currentDate='2024-12-30')
#     dfC = df[df['Datetime'] == currentDatetime].reset_index(drop=True)
 
    # dfC[(dfC['volumeSpike'] >= 50) & (dfC['volumeSpike'] <= 250) & (dfC['RSIStatus'] == 'Balanced') & (dfC['VWAPStatus'] == 'BULLISH') & (dfC['BuySignal'] == 'Buy Signal')]
    # dfC[(dfC['volumeSpike'] >= 50) & (dfC['volumeSpike'] <= 250) & (dfC['RSIStatus'] == 'Balanced') & (dfC['VWAPStatus'] == 'BULLISH') & (dfC['BuySignal'] == 'Buy Signal')].sort_values(by=['RSI', 'volumeSpike'])
    # dfC[(dfC['volumeSpike'] >= 50) & (dfC['priceAction'] >= 2) & (dfC['ATRStatus'] == 'QUALIFIED') & (dfC['VWAPStatus'] == 'BULLISH')]
    # dfC[(dfC['VWAPStatus'] == 'BULLISH')].head(20)
    
# df[df['BuySignal'] == 'Buy Signal'].head(20)    
   

# dfC[dfC['tickerName'] == 'NIFTY_FIN_SERVICE']
# df[df['VWAPStatus'] != 'BULLISH']

# df[(df['volumeSpike'] >= 50) & (df['priceAction'] >= 2) & (df['ATRStatus'] == 'QUALIFIED') & (df['VWAPStatus'] == 'BULLISH')]
# df[(df['volumeSpike'] >= 50) & (df['volumeSpike'] <= 250) & (df['priceAction'] >= 2) & (df['ATRStatus'] == 'QUALIFIED') & (df['VWAPStatus'] == 'BULLISH')] 
# df[(df['volumeSpike'] >= 50) & (df['volumeSpike'] <= 250) & (df['RSIStatus'] == 'Balanced') & (df['VWAPStatus'] == 'BULLISH')].tail(20)
# df[(df['volumeSpike'] >= 50) & (df['volumeSpike'] <= 250) & (df['RSIStatus'] == 'Balanced') & (df['VWAPStatus'] == 'BULLISH')].iloc[25:45]
# dfC['tickerName'].unique()    
