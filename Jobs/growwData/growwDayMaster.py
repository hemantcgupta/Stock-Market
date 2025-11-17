# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:55:48 2024

@author: Hemant
"""

import requests
import pandas as pd
import pytz
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from multiprocessing import Pool, cpu_count
from decorators.retry import retry

class VAR:
    db_name_mk_groww_info = 'mkgrowwinfo'
    table_name_mkGrowwInfo = 'mkGrowwInfo'
    db_name_mkgrowwdaymaster = 'mkgrowwdaymaster'
    cpu_count = int(cpu_count() * 0.8)
    parquet_file_path = './Data/parquet_files/mkgrowwdaymaster.parquet'

def date_to_milliseconds(date_string, date_format="%Y-%m-%d %H:%M:%S"):
    dt = datetime.strptime(date_string, date_format)
    timestamp_in_ms = int(dt.timestamp() * 1000)
    return timestamp_in_ms

def milliseconds_to_date(timestamp_in_ms, date_format="%Y-%m-%d %H:%M:%S"):
    timestamp_in_seconds = timestamp_in_ms / 1000
    dt = datetime.fromtimestamp(timestamp_in_seconds)
    return dt.strftime(date_format)

@retry(retries=3, delay=1)
def fetch_stock_data(tickerName, startTimeInMillis, endTimeInMillis, intervalInMinutes):
    url = f"https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/{tickerName}"
    startTimeInMillis = date_to_milliseconds(startTimeInMillis, date_format="%Y-%m-%d %H:%M:%S")
    endTimeInMillis = date_to_milliseconds(endTimeInMillis, date_format="%Y-%m-%d %H:%M:%S")
    params = {
        "endTimeInMillis": endTimeInMillis,
        "intervalInMinutes": intervalInMinutes,
        "startTimeInMillis": startTimeInMillis
    }
    try:
        response = requests.get(url, params=params)
        data = response.json() 
        if 'candles' in data:
            dataRow = data.get('candles', [])
            df = pd.DataFrame(dataRow, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            timezone = pytz.timezone('Asia/Kolkata')
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s', utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df.drop_duplicates().reset_index(drop=True)
        else:
            df = pd.DataFrame()
    except Exception as e:
        print("An error occurred:", e)
        df = pd.DataFrame()  # Return an empty DataFrame on error
    return df
    

def groww_stock_data_download(tickerName):
    defaultStartTime = '2010-01-01 00:00:00'
    # today = datetime.now().date()    
    today = datetime.now().date() if datetime.now().hour >= 16 else (datetime.now() - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0).date()
    endTimeInMillis = f'{today} 23:59:59'
    intervalInMinutes = 1440
    query_day = f'select max(Datetime) from [{tickerName}]'
    try:
        startTimeInMillis = pd.read_sql(query_day, cnxn(VAR.db_name_mkgrowwdaymaster)).iloc[0][0]
        if pd.notnull(startTimeInMillis):  
            startTimeInMillis = (pd.to_datetime(startTimeInMillis) + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            startTimeInMillis = defaultStartTime
    except Exception as e:
        print(f"Error fetching max datetime for {tickerName}: {e}")
        startTimeInMillis = defaultStartTime
    df = fetch_stock_data(tickerName, startTimeInMillis, endTimeInMillis, intervalInMinutes)
    if not df.empty:
        df['Datetime'] = pd.to_datetime(df['Datetime']) - pd.to_timedelta(pd.to_datetime(df['Datetime']).dt.minute % 5, unit='m')
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%Y-%m-%d 00:00:00')
    result = {'tableName': tickerName, 'Dataframe': df}
    return result
    
 
def update_table(result, insertMethod):
    result = [
        Data_Inserting_Into_DB(
            dct.get('Dataframe'), VAR.db_name_mkgrowwdaymaster, dct.get('tableName'), insertMethod
        ) if isinstance(dct.get('Dataframe'), pd.DataFrame) and not dct.get('Dataframe').empty else {
            'dbName': VAR.db_name_mkgrowwdaymaster,
            dct.get('tableName'): 'Unsuccessful - Empty or None DataFrame'
        }
        for dct in tqdm(result, desc='Update Table')
    ]
    return result
    
def save_parquet_files(df_list):
    df = pd.concat(df_list).reset_index(drop=True)
    df.to_parquet(VAR.parquet_file_path, index=False)
    # pd.read_parquet(VAR.parquet_file_path)
    
def save_parquet_files_tickerName(tickerName):
    query_day = f'select * from [{tickerName}]'
    try:
        df = pd.read_sql(query_day, cnxn(VAR.db_name_mkgrowwdaymaster))
        df['tickerName'] = tickerName
        return df
    except Exception as e:
        print(f"Error fetching max datetime for {tickerName}: {e}")
        return None
    
  
def JobGrowwInfoDataDownloader():
    create_database(VAR.db_name_mkgrowwdaymaster)
    query_day = f'select distinct nseScriptCode from {VAR.table_name_mkGrowwInfo} where nseScriptCode is not null'
    tickerNameList = pd.read_sql(query_day, cnxn(VAR.db_name_mk_groww_info))['nseScriptCode'].unique()   
    # tickerNameList = ['BLS']
    with Pool(processes=VAR.cpu_count) as pool:
        results = list(tqdm(pool.imap(groww_stock_data_download, tickerNameList), total=len(tickerNameList), desc='Groww Stock Data Downloader'))
    results = update_table(results, 'append')
    with Pool(processes=VAR.cpu_count) as pool:
        df_list = list(tqdm(pool.imap(save_parquet_files_tickerName, tickerNameList), total=len(tickerNameList), desc='save_parquet_files_tickerName'))
    save_parquet_files(df_list)
    return results
    
if __name__ == "__main__":
    result = JobGrowwInfoDataDownloader()



