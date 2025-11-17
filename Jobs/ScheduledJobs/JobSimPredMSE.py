# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 02:08:29 2024

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
    table_name_simulationPrediction = 'simulationPrediction'
    table_name_simulationPredMSE = 'simulationPredMSE'
    
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
                SELECT md.tickerName, max(mp.Datetime) as MaxDatetime FROM {VAR.table_name_simulationPrediction} md
                LEFT JOIN {VAR.table_name_simulationPredMSE} mp ON md.tickerName = mp.tickerName
                group by md.tickerName
                '''
        df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
    except Exception as e:
        query = f'''
            SELECT md.tickerName, max(md.Datetime) as MaxDatetime FROM {VAR.table_name_simulationPrediction} md
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
def update_SimPredMSE(stock_symbols_dict):
    resultD = Delete_max_date(VAR.db_name_analyzer, stock_symbols_dict)
    ticker_name = stock_symbols_dict.get('tickerName')
    MinDatetime = stock_symbols_dict.get('MinDatetime')
    MaxDatetime = stock_symbols_dict.get('MaxDatetime')
    query = f'''
            SELECT Datetime, predDatetime, Entry2, Exit2, predTmEntry2, predTmExit2
            FROM {VAR.table_name_simulationPrediction} 
            WHERE tickerName = '{ticker_name}'
            '''
    if not pd.isna(MinDatetime):
        query += f"and Datetime >= '{MinDatetime}'"
    df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
    if pd.isna(MaxDatetime) or df['Datetime'].max() >= MaxDatetime:
        df = SimPredMSE_data_process(ticker_name, stock_symbols_dict, df)
        # result = Data_Inserting_Into_DB(df, VAR.db_name_analyzer, VAR.table_name_simulationPredMSE, 'append')
        return {'Message': 'New Data Added', 'tickerName': ticker_name, 'Dataframe': df}
    return {'Message': 'Already Upto-Date', 'tickerName': ticker_name, 'Dataframe': None}

def Delete_max_date(dbName, stock_symbols_dict):
    try:
        ticker_name = stock_symbols_dict.get("tickerName")
        max_date = stock_symbols_dict.get("MaxDatetime")
        conn = cnxn(dbName)
        cursor = conn.cursor()
        delete_query = f"DELETE FROM {VAR.table_name_simulationPredMSE} WHERE tickerName = '{ticker_name}' and Datetime = '{max_date}';"
        cursor.execute(delete_query)
        conn.commit()
        return {**stock_symbols_dict, 'status': 'success'}
    except:
        return {**stock_symbols_dict, 'status': 'error'}
# =============================================================================
# Multi Process Group The 1 month Prediction 
# =============================================================================
def SimPredMSE_data_process(ticker_name, stock_symbols_dict, df):
    df = df.sort_values(by='Datetime').reset_index(drop=True)
    df['MinDatetime'] = df['Datetime'] - pd.DateOffset(months=1)
    df.set_index('Datetime', inplace=True)
    df = Square_Error(df)
    multi_data_processor = MultiDataProcessor(stock_symbols_dict, df)
    dct_data = multi_data_processor.multiprocessing()
    df = pd.DataFrame(dct_data)
    df['tickerName'] = ticker_name
    df = df.apply(lambda col: col.apply(json.dumps) if col.apply(lambda x: isinstance(x, (dict, list))).any() else col)
    return df

def Square_Error(df):
    try:
        df['predTmEntry2MSE'] = ((((df['Entry2'] - df['predTmEntry2'])/df['Entry2'])*100)**2).round(2)
        df['predTmExit2MSE'] = ((((df['Exit2'] - df['predTmExit2'])/df['Exit2'])*100)**2).round(2)
    except:
        df['predTmEntry2MSE'] = 0
        df['predTmExit2MSE'] = 0
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
            dct = NextMSEPrediction(subset_df)
            dct['Datetime'] = current_date
            return dct
        except:
            return {}

def NextMSEPrediction(df):
    predTmEntry2MSE = round(df['predTmEntry2MSE'].mean(), 2)
    predTmExit2MSE = round(df['predTmExit2MSE'].mean(), 2)
    predDct = {
        'predTmEntry2MSE': predTmEntry2MSE,
        'predTmExit2MSE': predTmExit2MSE
        }
    return predDct
        

# =============================================================================
# Job Function
# =============================================================================
def JobSimPredMSE():
    validation()
    stock_symbols = fetch_max_dates()
    with Pool(processes=int(cpu_count() * 0.8)) as pool:
        result = list(tqdm(pool.imap(update_SimPredMSE, stock_symbols), total=len(stock_symbols), desc=f'Update Table {VAR.table_name_simulationPredMSE}'))
    df = pd.concat([dct.get('Dataframe', None) for dct in result]).reset_index(drop=True)
    result = Data_Inserting_Into_DB(df, VAR.db_name_analyzer, VAR.table_name_simulationPredMSE, 'append')
    return result
    
if __name__ == "__main__":
    result = JobSimPredMSE()
