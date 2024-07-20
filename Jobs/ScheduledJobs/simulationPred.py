# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:30:58 2024

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
    table_name_simulationPrediction = 'simulationPrediction'
    
# =============================================================================
# Validaition Before Processing    
# =============================================================================
def validation():
    create_database(VAR.db_name_analyzer)  

# =============================================================================
# Fetch Min Max Datetime From Old Data for Each Ticker Names
# =============================================================================
def fetch_max_dates():
    try:
        query = f'''
                SELECT tickerName, max(Datetime) as MaxDatetime FROM {VAR.table_name_simulationPrediction}
                group by tickerName
                '''
        df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
    except Exception as e:
        query = f'''
            SELECT tickerName, max(Datetime) as MaxDatetime FROM {VAR.table_name_dprediction} 
            group by tickerName
            '''
        df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
        df['MaxDatetime'] = np.datetime64('NaT')
    return df.to_dict('records')   


# =============================================================================
# Fetch New data and Process For Simulation Prediction
# Update simulationPrediction Table 
# =============================================================================
def update_simulationPrediction(stock_symbols_dict):
    ticker_name = stock_symbols_dict.get('tickerName')
    MaxDatetime = stock_symbols_dict.get('MaxDatetime')
    resultD = Delete_max_date(ticker_name, MaxDatetime)
    query1 =  f'''
    SELECT *, LEAD(Datetime, 1) OVER (ORDER BY Datetime) AS predDatetime FROM mkDayPrediction WHERE tickerName = '{ticker_name}'
    '''
    query2 =  f'''
    SELECT Datetime as predDatetime, Entry1, Exit1, EtExProfit1, Entry2, Exit2, EtExProfit2 FROM mkIntervalFeature WHERE tickerName = '{ticker_name}'
    '''
    query3 =  f'''
    SELECT Datetime as predDatetime, [Open], [High], [Low], [Close], [OC-P/L] FROM mkDayFeature WHERE tickerName = '{ticker_name}'
    '''
    if not pd.isna(MaxDatetime):
        query1 += f"and Datetime >= '{MaxDatetime}'"
        query2 += f"and Datetime >= '{MaxDatetime}'"
        query3 += f"and Datetime >= '{MaxDatetime}'"
    
    df1 = pd.read_sql(query1, cnxn(VAR.db_name_analyzer))
    df2 = pd.read_sql(query2, cnxn(VAR.db_name_analyzer))
    df3 = pd.read_sql(query3, cnxn(VAR.db_name_analyzer))
    dataframes = [df1, df2, df3]
    dataframes = [df.apply(lambda x: pd.to_datetime(x) if 'Datetime' in x.name else x) for df in dataframes]
    df1, df2, df3 = dataframes
    df = pd.merge(df1, df2, how='left', on='predDatetime')
    df = pd.merge(df, df3, how='left', on='predDatetime')
    df['predDatetime'].fillna(df['predDatetime'].max()+ pd.Timedelta(days=1), inplace=True)
    if pd.isna(MaxDatetime) or df['Datetime'].max() >= MaxDatetime:
        df = simulation_prediction_data_process(ticker_name, df)
        result = Data_Inserting_Into_DB(df, VAR.db_name_analyzer, VAR.table_name_simulationPrediction, 'append')
        return {**result, 'Message': 'New Data Added', 'tickerName': ticker_name}
    return {'Message': 'Already Upto-Date', 'tickerName': ticker_name}


def Delete_max_date(ticker_name, MaxDatetime):
    try:
        conn = cnxn(f'{VAR.db_name_analyzer}')
        cursor = conn.cursor()
        delete_query = f"DELETE FROM {VAR.table_name_simulationPrediction} WHERE tickerName = '{ticker_name}' and Datetime = '{MaxDatetime}';"
        cursor.execute(delete_query)
        conn.commit()
        return {'tickerName': ticker_name, 'status': 'success'}
    except:
        return {'tickerName': ticker_name, 'status': 'error'}

# =============================================================================
# Simulation Prediction Logic
# =============================================================================
def simulation_prediction_data_process(ticker_name, df):
    df.loc[(df['predTmEntry1'] >= df['Entry1']) & (df['predTmExit1'] <= df['Exit1']), 'SimEtEx1-P/L'] = df['EtEx1Profit'].round(2)
    df.loc[(df['predTmEntry1'] >= df['Entry1']) & (df['predTmExit1'] > df['Exit1']), 'SimEtEx1-P/L'] = ((1 - (df['predTmEntry1']/df['Close']))*100).round(2)
    df.loc[df['predTmEntry1'] < df['Entry1'], 'SimEtEx1-P/L'] = 0
    df.loc[(df['predTmEntry2'] >= df['Entry2']) & (df['predTmExit2'] <= df['Exit2']), 'SimEtEx2-P/L'] = df['EtEx2Profit'].round(2)
    df.loc[(df['predTmEntry2'] >= df['Entry2']) & (df['predTmExit2'] > df['Exit2']), 'SimEtEx2-P/L'] = ((1 - (df['predTmEntry2']/df['Close']))*100).round(2)
    df.loc[df['predTmEntry2'] < df['Entry2'], 'SimEtEx2-P/L'] = 0
    SeqColumns = ['Datetime', 'predDatetime', 'tickerName', 'predTmOpen', 'predTmEntry1', 'predTmExit1',
     'predTmEntry2', 'predTmExit2', 'predTmClose', 'predTmMaxhigh',
     'predTmMaxlow', 'EtEx1Profit', 'EtEx2Profit', 'predTmP/L',
     'Entry1', 'Exit1', 'EtExProfit1', 'Entry2', 'Exit2',
     'EtExProfit2', 'Open', 'High', 'Low', 'Close', 'OC-P/L', 'SimEtEx1-P/L', 'SimEtEx2-P/L']
    df = df[SeqColumns]
    return df


# =============================================================================
# Job Function
# =============================================================================
def JobSimulationPrediction():
    validation()
    stock_symbols = fetch_max_dates()
    with Pool(processes=int(cpu_count() * 0.8)) as pool:
        result = list(tqdm(pool.imap(update_simulationPrediction, stock_symbols), total=len(stock_symbols), desc=f'Update Table {VAR.table_name_simulationPrediction}'))
    return result
    
if __name__ == "__main__":
    result = JobSimulationPrediction()