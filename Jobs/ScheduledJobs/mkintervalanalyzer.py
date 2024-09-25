# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 01:13:15 2024

@author: Hemant
"""


import pandas as pd
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from decorators.retry import retry

# =============================================================================
# Variables
# =============================================================================
class VAR:
    cpu_count = int(cpu_count() * 0.8)
    db_name_master = 'master'
    db_name_day = 'mkdaymaster'
    db_name_interval = 'mkintervalmaster'
    db_name_analyzer = 'mkanalyzer'
    db_name_ianalyzer = 'mkintervalanalyzer'
    table_name_ifeature = 'mkIntervalFeature'

# =============================================================================
# Validaition Before Processing
# =============================================================================
def validation():
    create_database(VAR.db_name_master)
    create_database(VAR.db_name_day)
    create_database(VAR.db_name_interval)
    create_database(VAR.db_name_analyzer)
    create_database(VAR.db_name_ianalyzer)

# =============================================================================
# Interval Analyzer 
# =============================================================================
def mkIntervalAnalyzerMain():
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
    stockSymbols = pd.read_sql(query, cnxn(VAR.db_name_interval))['TABLE_NAME'].tolist()
    result_day, result_interval = mkInterval_Data_Process(stockSymbols)
    return result_day, result_interval

# =============================================================================
# Multi Data Process for each Ticker Name
# =============================================================================
def mkInterval_Data_Process(stockSymbols):
    with Pool(processes=VAR.cpu_count) as pool:
        result = list(tqdm(pool.imap(fetch_db_data, stockSymbols), total=len(stockSymbols), desc='Updating mkintervalanalyzer and mkIntervalFeature'))    
    result_day, result_interval = update_table(result, 'append')
    return result_day, result_interval

# =============================================================================
# Fetch each ticker from both master databse 'daymaster' and 'intervalmaster'
# Process and Finding all features from interval data 
# =============================================================================
@retry(retries=3, delay=1)
def fetch_db_data(tickerName): 
    try:
        latest_date_query = f"SELECT MAX(Datetime) FROM {VAR.db_name_ianalyzer}.dbo.[{tickerName}]"
        latest_date = pd.read_sql(latest_date_query, cnxn(VAR.db_name_ianalyzer)).iloc[0, 0].normalize()
        query = f"SELECT MAX(Datetime) FROM {VAR.table_name_ifeature}"
        minLatestDate = pd.read_sql(query, cnxn(VAR.db_name_analyzer)).iloc[0, 0].normalize()
        if minLatestDate < latest_date:
            latest_date = minLatestDate
        if pd.notnull(latest_date):
            filterDate = latest_date - pd.DateOffset(days=10)
        resultD = Delete_max_date(VAR.db_name_master, tickerName, latest_date)
    except:     
        filterDate = pd.Timestamp('2023-12-01')
        latest_date = filterDate
    query = f'''
            SELECT * FROM {VAR.db_name_interval}.dbo.[{tickerName}] AS interval
            LEFT JOIN (SELECT [Datetime] AS Date, [Open] AS OpenDay FROM {VAR.db_name_day}.dbo.[{tickerName}]) AS day
            ON CONVERT(date, interval.Datetime) = day.Date
            WHERE interval.Datetime >= '{filterDate}'
            ORDER BY Datetime      
            '''
    df = pd.read_sql(query, cnxn(VAR.db_name_master))
    df = MovingAverage44(df)  
    df = df[df['Datetime'] > latest_date].reset_index(drop=True)
    df = yfDownloadProcessingInterval(df)
    df = find_support_resistance(df)
    dfCandle = pd.merge(
        pd.merge(
            df.groupby('Date')['CandleP/N_OpenDay'].value_counts().unstack(fill_value=0).reset_index().rename(columns={-1: 'nCandleBelowOpen', 1: 'pCandleAboveOpen'}).reindex(columns=['Date', 'nCandleBelowOpen', 'pCandleAboveOpen'], fill_value=0),
            df.groupby('Date')['CandleP/N'].value_counts().unstack(fill_value=0).reset_index().rename(columns={-1: 'nCandle', 1: 'pCandle'}).reindex(columns=['Date', 'nCandle', 'pCandle'], fill_value=0),
            how='left', on='Date'
        ),
        df.groupby('Date')['44TF'].value_counts().unstack(fill_value=0).reset_index().rename(columns={1: 'Hits44MA'}).reindex(columns=['Date', 'Hits44MA'], fill_value=0),
        how='left', on='Date'
    )
    dfEtEx = pd.merge(EntryExitMinToMax(df), EntryExitMaxToMin(df), how='left', on='Date')    
    dfItCd = pd.merge(dfCandle, dfEtEx,how='left', on='Date')
    df, dfItCd = data_cleanning(df, dfItCd, tickerName)
    result = {'tableName': tickerName, VAR.db_name_ianalyzer: df, VAR.db_name_analyzer: dfItCd}
    return result

# =============================================================================
# Delete last record
# =============================================================================
def Delete_max_date(dbName, tickerName, max_date):
    try:
        conn = cnxn(dbName)
        cursor = conn.cursor()
        delete_query = f"DELETE FROM {VAR.db_name_ianalyzer}.dbo.[{tickerName}] WHERE Datetime >= '{max_date}';"
        cursor.execute(delete_query)
        delete_query = f"DELETE FROM {VAR.db_name_analyzer}.dbo.[{VAR.table_name_ifeature}] WHERE tickerName = '{tickerName}' and Datetime >= '{max_date}';"
        cursor.execute(delete_query)
        conn.commit()
        return {'status': 'success'}
    except:
        return {'status': 'error'}
    
# =============================================================================
# Process 1: Finding Postive and Negative Candle Patterns
# =============================================================================
def yfDownloadProcessingInterval(df):
    df['Date'] = df['Date'].fillna(df['Datetime'].dt.normalize())
    df['OpenDay'] = df['OpenDay'].fillna(df['Open'])
    df['CandleP/N_OpenDay'] = df.apply(lambda row: 1 if row['OpenDay'] <= row['Open'] else -1, axis=1)
    df['CandleP/N'] = df.apply(lambda row: 1 if row['Open'] <= row['Close'] else -1, axis=1)
    df = df[['Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'OpenDay', 'CandleP/N_OpenDay', 'CandleP/N', '44MA', '44TF']]
    return df

# =============================================================================
# Process 2: Finding Moving Average 44
# =============================================================================
def MovingAverage44(df):
    df['44MA'] = df['Close'].rolling(window=44).mean().fillna(0).round(2)
    df['44TF'] = df.apply(lambda row: 1 if row['44MA'] <= row['High'] and row['44MA'] >= row['Low'] else 0, axis=1)
    return df

# =============================================================================
# Process 3: Finding Supoort and Resistance
# =============================================================================
@retry(retries=3, delay=1)
def find_support_resistance(df, window_size=20):
    data = df[['Datetime', 'Low', 'High']]
    supports = []
    resistances = []
    for i in range(window_size, len(data) - window_size):
        window_low = data['Low'].iloc[i - window_size:i + window_size]
        window_high = data['High'].iloc[i - window_size:i + window_size]
        min_val = window_low.min()
        max_val = window_high.max()
        avg_low = window_low.mean()
        avg_high = window_high.mean()
        if data['Low'].iloc[i] < avg_low and data['Low'].iloc[i] == min_val:
            supports.append({'Datetime': data['Datetime'].iloc[i], 'Support': data['Low'].iloc[i]})
        if data['High'].iloc[i] > avg_high and data['High'].iloc[i] == max_val:
            resistances.append({'Datetime': data['Datetime'].iloc[i], 'Resistance': data['High'].iloc[i]})
    dfS = pd.DataFrame(supports).reindex(columns=['Datetime', 'Support'], fill_value=0)
    dfR = pd.DataFrame(resistances).reindex(columns=['Datetime', 'Resistance'], fill_value=0)  
    dfSR = pd.merge(df, dfS, how='left', on='Datetime')
    dfSR = dfSR.merge(dfR, how='left', on='Datetime')
    return dfSR

# =============================================================================
# Process 4: Finding Entry and Exit Point From Min to Max Profit (ML)
# =============================================================================
def EntryExitMinToMax(dfInterval):
    dfEtEx = dfInterval.groupby('Date')['Low'].min().reset_index()
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'Low', 'Datetime']], how='left', on=['Date', 'Low']).drop_duplicates(subset=['Date', 'Low'], keep='last').reset_index(drop=True).rename(columns={'Low': 'Entry', 'Datetime': 'entryDatetime'})
    dfEtEx[['Exit', 'exitDatetime']] = dfEtEx.apply(lambda row: dfInterval.loc[dfInterval[(dfInterval['Date'] == row['Date']) & (dfInterval['Datetime'] >= row['entryDatetime'])]['High'].idxmax(), ['High', 'Datetime']], axis=1)
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'Datetime', 'OpenDay']].drop_duplicates(subset='Date', keep='first').reset_index(drop=True).rename(columns={'Datetime': 'OpenDayDatetime'}), how='left', on='Date')
    dfEtEx = dfEtEx[['Date', 'OpenDayDatetime', 'entryDatetime', 'exitDatetime', 'OpenDay', 'Entry','Exit']]
    dfEtEx['entrytimeDiff'] = ((dfEtEx['entryDatetime'] - dfEtEx['OpenDayDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['exittimeDiff'] = ((dfEtEx['exitDatetime'] - dfEtEx['entryDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['OpenToEntryLoss'] = ((1-dfEtEx['OpenDay']/dfEtEx['Entry'])*100).round(2)
    dfEtEx['OpenToExitProfit'] = ((1-dfEtEx['OpenDay']/dfEtEx['Exit'])*100).round(2)
    dfEtEx['EtExProfit'] = ((1-dfEtEx['Entry']/dfEtEx['Exit'])*100).round(2)
    dfEtEx = dfEtEx[['Date', 'Entry', 'Exit', 'entrytimeDiff', 'exittimeDiff', 'OpenToEntryLoss', 'OpenToExitProfit', 'EtExProfit']].rename(columns=lambda x: x + '1' if x != 'Date' else x)
    return dfEtEx

# =============================================================================
# Process 5: Finding Entry and Exit Point From Max to Min Profit (MH)
# =============================================================================
def EntryExitMaxToMin(dfInterval):
    dfEtEx = dfInterval.groupby('Date')['High'].max().reset_index()
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'High', 'Datetime']], how='left', on=['Date', 'High']).drop_duplicates(subset=['Date', 'High'], keep='last').reset_index(drop=True).rename(columns={'High': 'Exit', 'Datetime': 'exitDatetime'})
    dfEtEx[['Entry', 'entryDatetime']] = dfEtEx.apply(lambda row: dfInterval.loc[dfInterval[(dfInterval['Date'] == row['Date']) & (dfInterval['Datetime'] <= row['exitDatetime'])]['Low'].idxmin(), ['Low', 'Datetime']], axis=1)
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'Datetime', 'OpenDay']].drop_duplicates(subset='Date', keep='first').reset_index(drop=True).rename(columns={'Datetime': 'OpenDayDatetime'}), how='left', on='Date')
    dfEtEx = dfEtEx[['Date', 'OpenDayDatetime', 'entryDatetime', 'exitDatetime', 'OpenDay', 'Entry','Exit']]
    dfEtEx['entrytimeDiff'] = ((dfEtEx['entryDatetime'] - dfEtEx['OpenDayDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['exittimeDiff'] = ((dfEtEx['exitDatetime'] - dfEtEx['entryDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['OpenToEntryLoss'] = ((1-dfEtEx['OpenDay']/dfEtEx['Entry'])*100).round(2)
    dfEtEx['OpenToExitProfit'] = ((1-dfEtEx['OpenDay']/dfEtEx['Exit'])*100).round(2)
    dfEtEx['EtExProfit'] = (((dfEtEx['Exit'] - dfEtEx['Entry'])/dfEtEx['Exit'])*100).round(2)
    dfEtEx = dfEtEx[['Date', 'Entry', 'Exit', 'entrytimeDiff', 'exittimeDiff', 'OpenToEntryLoss', 'OpenToExitProfit', 'EtExProfit']].rename(columns=lambda x: x + '2' if x != 'Date' else x)
    return dfEtEx

# =============================================================================
# Data Formating Before updating Database
# =============================================================================
def data_cleanning(df, dfItCd, tickerName):
    df = df[['Date', 'Datetime', 'OpenDay', 'CandleP/N_OpenDay', 'CandleP/N', '44MA', '44TF', 'Support', 'Resistance']]
    dfItCd['tickerName'] = tickerName
    dfItCd = dfItCd[['Date', 'tickerName', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA', 'Entry1', 'Exit1', 'entrytimeDiff1', 'exittimeDiff1', 'OpenToEntryLoss1', 'OpenToExitProfit1', 'EtExProfit1', 'Entry2', 'Exit2', 'entrytimeDiff2', 'exittimeDiff2', 'OpenToEntryLoss2', 'OpenToExitProfit2', 'EtExProfit2']].rename(columns={'Date': 'Datetime'})
    return df, dfItCd

# =============================================================================
# update Database
# =============================================================================
def update_table(result, insertMethod):  
    result_interval = [Data_Inserting_Into_DB(dct.get(VAR.db_name_ianalyzer), VAR.db_name_ianalyzer, dct.get('tableName'), insertMethod) for dct in tqdm(result, desc=f'Update Table {VAR.db_name_ianalyzer}') if isinstance(dct, dict)]
    dfItCd = pd.concat([dct.get(VAR.db_name_analyzer) for dct in result if isinstance(dct, dict)]).reset_index(drop=True)
    result_day = Data_Inserting_Into_DB(dfItCd, VAR.db_name_analyzer, VAR.table_name_ifeature, insertMethod)
    return result_day, result_interval
    
# =============================================================================
# Job Function
# =============================================================================
def JobmkIntervalAnalyzer():
    validation()
    result_day, result_interval = mkIntervalAnalyzerMain()
    return result_day, result_interval

if __name__ == "__main__":
    result_day, result_interval = JobmkIntervalAnalyzer()

