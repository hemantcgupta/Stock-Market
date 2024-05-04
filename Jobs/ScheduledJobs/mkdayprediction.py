# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:48:03 2024

@author: Hemant
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:16:28 2024

@author: Hemant
"""

import pandas as pd
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn

# =============================================================================
# Variables
# =============================================================================
class VAR:
    cpu_count = int(cpu_count() * 0.8)
    db_name_analyzer = 'mkanalyzer'
    table_name_dfeature = 'mkDayFeature'
    table_name_ifeature = 'mkIntervalFeature'
    table_name_dprediction = 'mkDayPrediction'
    
# =============================================================================
# Validaition Before Processing    
# =============================================================================
def validation():
    create_database(VAR.db_name_analyzer)    
    
# =============================================================================
# Fetch Min Max  Datetime From Old Data for Each Ticker Names
# =============================================================================
def fetch_max_dates():
    try:
        query = f'''
                SELECT md.tickerName, max(mp.Datetime) as MaxDatetime FROM {VAR.table_name_ifeature} md
                LEFT JOIN {VAR.table_name_dprediction} mp ON md.tickerName = mp.tickerName
                group by md.tickerName
                '''
        df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
    except Exception as e:
        query = f'''
            SELECT md.tickerName, max(md.Datetime) as MaxDatetime FROM {VAR.table_name_ifeature} md
            group by md.tickerName
            '''
        df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
        df['MaxDatetime'] = np.datetime64('NaT')
    df['MinDatetime'] = df['MaxDatetime'] - pd.DateOffset(months=1)
    return df.to_dict('records')   

# =============================================================================
# Fetch New data and Process to Calculate TM Predition
# Update mkDay Prediction Table 
# =============================================================================
def update_mkday_prediction(stock_symbols_dict):
    resultD = Delete_max_date(VAR.db_name_analyzer, stock_symbols_dict)
    ticker_name = stock_symbols_dict.get('tickerName')
    MinDatetime = stock_symbols_dict.get('MinDatetime')
    MaxDatetime = stock_symbols_dict.get('MaxDatetime')
    query = f'''
            SELECT Interval.*, Day.[Open], Day.[Close], Day.maxHigh, Day.maxLow, Day.PvClose 
            FROM {VAR.table_name_ifeature} AS Interval
            LEFT JOIN {VAR.table_name_dfeature} AS Day 
            ON Interval.Datetime = Day.Datetime and Interval.tickerName = Day.tickerName
            WHERE Interval.tickerName = '{ticker_name}'
            '''
    if not pd.isna(MinDatetime):
        query += f"and Interval.Datetime >= '{MinDatetime}'"
    df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
    if pd.isna(MaxDatetime) or df['Datetime'].max() >= MaxDatetime:
        df = mkday_prediction_data_process(ticker_name, stock_symbols_dict, df)
        result = Data_Inserting_Into_DB(df, VAR.db_name_analyzer, VAR.table_name_dprediction, 'append')
        return {**result, 'Message': 'New Data Added', 'tickerName': ticker_name}
    return {'Message': 'Already Upto-Date', 'tickerName': ticker_name}

def Delete_max_date(dbName, stock_symbols_dict):
    try:
        ticker_name = stock_symbols_dict.get("tickerName")
        max_date = stock_symbols_dict.get("MaxDatetime")
        conn = cnxn(dbName)
        cursor = conn.cursor()
        delete_query = f"DELETE FROM {VAR.table_name_dprediction} WHERE tickerName = '{ticker_name}' and Datetime = '{max_date}';"
        cursor.execute(delete_query)
        conn.commit()
        return {**stock_symbols_dict, 'status': 'success'}
    except:
        return {**stock_symbols_dict, 'status': 'error'}
# =============================================================================
# Multi Process Group The 1 month Prediction 
# =============================================================================
def mkday_prediction_data_process(ticker_name, stock_symbols_dict, df):
    df = df.sort_values(by='Datetime').reset_index(drop=True)
    df['MinDatetime'] = df['Datetime'] - pd.DateOffset(months=1)
    df.set_index('Datetime', inplace=True)
    multi_data_processor = MultiDataProcessor(stock_symbols_dict, df)
    dct_data = multi_data_processor.multiprocessing()
    df = pd.DataFrame(dct_data)
    df['tickerName'] = ticker_name
    df = df.apply(lambda col: col.apply(json.dumps) if col.apply(lambda x: isinstance(x, (dict, list))).any() else col)
    df = df[['Datetime', 'tickerName', 'predTmOpen', 'predTmEntry1', 'predTmExit1', 'predTmEntry2', 'predTmExit2', 'predTmClose', 'predTmMaxhigh', 'predTmMaxlow', 'EtEx1Profit', 'EtEx2Profit', 'predTmP/L']]
    return df

# =============================================================================
# Multiprocess Clsss That Help to Calculate TM Predition for Each Datetime
# =============================================================================
class MultiDataProcessor:
    def __init__(self, stock_symbols_dict, df):
        self.stock_symbols_dict = stock_symbols_dict
        self.df = df
        
    def multiprocessing(self):
        MaxDatetime = self.stock_symbols_dict.get('MaxDatetime')
        DateRange = [(current_date, past_date) for current_date, past_date in zip(self.df.index, self.df['MinDatetime']) if pd.isna(MaxDatetime) or current_date >= MaxDatetime]
        result = [self.prediction_process(arg) for arg in DateRange]
        return result

    def prediction_process(self, arg):
        try:
            current_date, past_date = arg
            subset_df = self.df.iloc[(self.df.index >= past_date) & (self.df.index <= current_date)]
            dct = NextDayPrediction(subset_df)
            dct['Datetime'] = current_date
            return dct
        except:
            return {}

def NextDayPrediction(df):
    predTmOpen = round((df['Open']/df['PvClose']-1).mean()*df['Close'].iloc[-1]+df['Close'].iloc[-1], 2)
    predTmEntry1 = round((df['Entry1']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmEntry2 = round((df['Entry2']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmExit1 = round((df['Exit1']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmExit2 = round((df['Exit2']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmClose = round(predTmOpen/(1-(1-df['Open']/df['Close']).mean()), 2)
    predTmMaxhigh = round((df['maxHigh'].mean()/100)*predTmOpen+predTmOpen, 2)
    predTmMaxlow = round((df['maxLow'].mean()/100)*predTmOpen+predTmOpen, 2)
    EtEx1Profit = round((1-predTmEntry1/predTmExit1)*100, 2)
    EtEx2Profit = round((1-predTmEntry2/predTmExit2)*100, 2)
    predPL = round((1-predTmOpen/predTmClose)*100, 2)
    predDct = {
        'predTmOpen': predTmOpen,
        'predTmEntry1': predTmEntry1,
        'predTmExit1': predTmExit1,
        'predTmEntry2': predTmEntry2,
        'predTmExit2': predTmExit2,
        'predTmClose': predTmClose,
        'predTmMaxhigh': predTmMaxhigh,
        'predTmMaxlow': predTmMaxlow,
        'EtEx1Profit': EtEx1Profit,
        'EtEx2Profit': EtEx2Profit,
        'predTmP/L': predPL
        }
    return predDct
        

# =============================================================================
# Job Function
# =============================================================================
def JobmkDayPrediction():
    validation()
    stock_symbols = fetch_max_dates()
    with Pool(processes=int(cpu_count() * 0.8)) as pool:
        result = list(tqdm(pool.imap(update_mkday_prediction, stock_symbols), total=len(stock_symbols), desc=f'Update Table {VAR.table_name_dprediction}'))
    return result
    
if __name__ == "__main__":
    result = JobmkDayPrediction()
