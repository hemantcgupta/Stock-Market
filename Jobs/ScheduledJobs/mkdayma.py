# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 01:13:35 2024

@author: Hemant
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn

class mkDayMa:
    def __init__(self):
        self.cpu_count = int(cpu_count() * 0.8)
        self.db_name_analyzer = 'mkanalyzer'
        self.table_name_dma = 'mkDayMA'
        self.table_name_dfeature = 'mkDayFeature'

    def validation(self):
        create_database(self.db_name_analyzer)
        
    def fetch_max_dates(self):
        try:
            query = f'''
                    SELECT md.tickerName, max(mp.Datetime) as MaxDatetime FROM {self.table_name_dfeature} md
                    LEFT JOIN {self.table_name_dma} mp ON md.tickerName = mp.tickerName
                    group by md.tickerName
                    '''
            df = pd.read_sql(query, cnxn(self.db_name_analyzer))
        except Exception as e:
            query = f'''
                SELECT md.tickerName, max(md.Datetime) as MaxDatetime FROM {self.table_name_dfeature} md
                group by md.tickerName
                '''
            df = pd.read_sql(query, cnxn(self.db_name_analyzer))
            df['MaxDatetime'] = np.datetime64('NaT')
        df['MinDatetime'] = pd.to_datetime(df['MaxDatetime']) - pd.DateOffset(days=88)
        return df.to_dict('records')

    def update_mkday_ma(self, stock_symbols_dict):
        resultD = self.Delete_max_date(self.db_name_analyzer, stock_symbols_dict)
        ticker_name = stock_symbols_dict.get('tickerName')
        MinDatetime = stock_symbols_dict.get('MinDatetime')
        MaxDatetime = stock_symbols_dict.get('MaxDatetime')
        query = f'''
                SELECT Datetime, tickerName, High, Low, [Close] FROM {self.table_name_dfeature} 
                WHERE tickerName='{ticker_name}'
                '''
        if not pd.isna(MinDatetime):
            query += f"and Datetime >= '{MinDatetime}'"
        df = pd.read_sql(query, cnxn(self.db_name_analyzer))
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        MaxDatetime = pd.to_datetime(MaxDatetime) if MaxDatetime is not None else None
        if pd.isna(MaxDatetime) or df['Datetime'].max() >= MaxDatetime:
            df = df.sort_values(by='Datetime', ascending=True).reset_index(drop=True)
            df = self.MovingAverage44(df, MaxDatetime)
            if not df.empty:
                df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            return {'Message': 'New Data Added', 'tickerName': ticker_name, 'Dataframe': df}
        return {'Message': 'Already Upto-Date', 'tickerName': ticker_name, 'Dataframe': None}

    def Delete_max_date(self, dbName, stock_symbols_dict):
        try:
            ticker_name = stock_symbols_dict.get("tickerName")
            max_date = stock_symbols_dict.get("MaxDatetime")
            conn = cnxn(dbName)
            cursor = conn.cursor()
            delete_query = f"DELETE FROM {self.table_name_dma} WHERE tickerName = '{ticker_name}' and Datetime = '{max_date}';"
            cursor.execute(delete_query)
            conn.commit()
            return {**stock_symbols_dict, 'status': 'success'}
        except:
            return {**stock_symbols_dict, 'status': 'error'}
        
    @staticmethod
    def MovingAverage44(df, MaxDatetime):
        df['44MA'] = df['Close'].rolling(window=44).mean().fillna(0).round(2)
        df['44TF'] = df.apply(lambda row: 1 if row['44MA'] <= row['High'] and row['44MA'] >= row['Low'] else 0, axis=1)
        df = df[['Datetime', 'tickerName', '44MA', '44TF']]
        if not pd.isna(MaxDatetime):
            df = df[df['Datetime'] >= MaxDatetime].reset_index(drop=True)
        return df
    
    
    def update_all_mkday_ma(self):
        stock_symbols = self.fetch_max_dates()
        with Pool(processes=self.cpu_count) as pool:
            results = list(tqdm(pool.imap(self.update_mkday_ma, stock_symbols), total=len(stock_symbols), desc='Update Table mkDayMA'))
        df = pd.concat([dct.get('Dataframe', None) for dct in results]).reset_index(drop=True)
        result = Data_Inserting_Into_DB(df, self.db_name_analyzer, self.table_name_dma, 'append')
        return results

def JobmkDayMa():
    MA = mkDayMa()
    MA.validation()
    result = MA.update_all_mkday_ma()
    return result

if __name__ == "__main__":
    MA = mkDayMa()
    MA.validation()
    results = MA.update_all_mkday_ma()










