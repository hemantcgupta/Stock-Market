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
from Scripts.dataprocess import NextDayPrediction

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
    ticker_name = stock_symbols_dict.get('tickerName')
    MinDatetime = stock_symbols_dict.get('MinDatetime')
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
    if pd.isna(stock_symbols_dict.get('MaxDatetime')) or df['Datetime'].max() > stock_symbols_dict.get('MaxDatetime'):
        df = mkday_prediction_data_process(ticker_name, stock_symbols_dict, df)
        result = Data_Inserting_Into_DB(df, VAR.db_name_analyzer, VAR.table_name_dprediction, 'append')
        return {**result, 'Message': 'New Data Added', 'tickerName': ticker_name}
    return {'Message': 'Already Upto-Date', 'tickerName': ticker_name}

# =============================================================================
# Multi Process Group The 1 month Prediction 
# =============================================================================
def mkday_prediction_data_process(ticker_name, stock_symbols_dict, df):
    df = df.sort_values(by='Datetime').reset_index(drop=True)
    df['MinDatetime'] = df['Datetime'] - pd.DateOffset(months=1)
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
        grouped_df = self.df.groupby('Datetime')
        with Pool(processes=VAR.cpu_count) as pool:
            result = list(pool.imap(self.prediction_process, grouped_df))
        return result

    def prediction_process(self, arg):
        current_date, group = arg
        past_date = group['MinDatetime'].iloc[0]
        if pd.isna(self.stock_symbols_dict.get('MaxDatetime')) or current_date > self.stock_symbols_dict.get('MaxDatetime'):
            subset_df = self.df[(self.df['Datetime'] >= past_date) & (self.df['Datetime'] <= current_date)]
            dct = NextDayPrediction(subset_df)
            dct['Datetime'] = current_date
            return dct
        return {}

# =============================================================================
# Job Function
# =============================================================================
def JobmkDayPrediction():
    validation()
    stock_symbols = fetch_max_dates()
    result = [update_mkday_prediction(stock_symbol) for stock_symbol in tqdm(stock_symbols, desc=f'Update Table {VAR.table_name_dprediction}')]
    return result
    
if __name__ == "__main__":
    result = JobmkDayPrediction()
