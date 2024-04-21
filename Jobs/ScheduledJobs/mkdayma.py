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
        df['MinDatetime'] = df['MaxDatetime'] - pd.DateOffset(days=44)
        return df.to_dict('records')

    def update_mkday_ma(self, stock_symbols_dict):
        ticker_name = stock_symbols_dict.get('tickerName')
        MinDatetime = stock_symbols_dict.get('MinDatetime')
        query = f'''
                SELECT Datetime, tickerName, High, Low, [Close] FROM {self.table_name_dfeature} 
                WHERE tickerName='{ticker_name}'
                '''
        if not pd.isna(MinDatetime):
            query += f"and Datetime >= '{MinDatetime}'"
        df = pd.read_sql(query, cnxn(self.db_name_analyzer))
        if pd.isna(stock_symbols_dict.get('MaxDatetime')) or df['Datetime'].max() > stock_symbols_dict.get('MaxDatetime'):
            df = self.MovingAverage44(df)
            result = Data_Inserting_Into_DB(df, self.db_name_analyzer, self.table_name_dma, 'append')
            return {**result, 'Message': 'New Data Added', 'tickerName': ticker_name}
        return {'Message': 'Already Upto-Date', 'tickerName': ticker_name}

    @staticmethod
    def MovingAverage44(df):
        df['44MA'] = df['Close'].rolling(window=44).mean().fillna(0)
        df['44TF'] = df.apply(lambda row: 1 if row['44MA'] <= row['High'] and row['44MA'] >= row['Low'] else 0, axis=1)
        df = df[['Datetime', 'tickerName', '44MA', '44TF']]
        return df

    def update_all_mkday_ma(self):
        stock_symbols = self.fetch_max_dates()
        with Pool(processes=self.cpu_count) as pool:
            results = list(tqdm(pool.imap(self.update_mkday_ma, stock_symbols), total=len(stock_symbols), desc='Update Table mkDayMA'))
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










