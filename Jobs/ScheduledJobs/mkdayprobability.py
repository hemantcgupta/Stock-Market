# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:16:28 2024

@author: Hemant
"""

from bisect import bisect_left, bisect_right
import pandas as pd
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from decorators.retry import retry


class mkDayProbability:
    def __init__(self):
        self.cpu_count = int(cpu_count() * 0.8)
        self.db_name_analyzer = 'mkanalyzer'
        self.table_name_dfeature = 'mkDayFeature'
        self.table_name_dprobability = 'mkDayProbability'

    def validation(self):
        create_database(self.db_name_analyzer)
        
    def fetch_max_dates(self):
        try:
            query = f'''
                    SELECT md.tickerName, max(mp.Datetime) as MaxDatetime FROM {self.table_name_dfeature} md
                    LEFT JOIN {self.table_name_dprobability} mp ON md.tickerName = mp.tickerName
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
        df['MinDatetime'] = pd.to_datetime(df['MaxDatetime']) - pd.DateOffset(months=1)
        return df.to_dict('records')
    
    @retry(retries=3, delay=1)
    def update_mkday_probability(self, stock_symbols_dict):
        resultD = self.Delete_max_date(self.db_name_analyzer, stock_symbols_dict)
        ticker_name = stock_symbols_dict.get('tickerName')
        MinDatetime = stock_symbols_dict.get('MinDatetime')
        MaxDatetime = stock_symbols_dict.get('MaxDatetime')
        query = f'''
                SELECT * FROM {self.table_name_dfeature} 
                WHERE tickerName='{ticker_name}'
                '''                
        if not pd.isna(MinDatetime):
            query += f"and Datetime >= '{MinDatetime}'"
        df = pd.read_sql(query, cnxn(self.db_name_analyzer))
        if pd.isna(MaxDatetime) or df['Datetime'].max() >= MaxDatetime:
            df = self.mkday_probability_data_process(ticker_name, stock_symbols_dict, df)
            if not df.empty:
                df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            result = Data_Inserting_Into_DB(df, self.db_name_analyzer, self.table_name_dprobability, 'append')
            return {**result, 'Message': 'New Data Added', 'tickerName': ticker_name}
        return {'Message': 'Already Upto-Date', 'tickerName': ticker_name}

    def Delete_max_date(self, dbName, stock_symbols_dict):
        try:
            ticker_name = stock_symbols_dict.get("tickerName")
            max_date = stock_symbols_dict.get("MaxDatetime")
            conn = cnxn(dbName)
            cursor = conn.cursor()
            delete_query = f"DELETE FROM {self.table_name_dprobability} WHERE tickerName = '{ticker_name}' and Datetime = '{max_date}';"
            cursor.execute(delete_query)
            conn.commit()
            return {**stock_symbols_dict, 'status': 'success'}
        except:
            return {**stock_symbols_dict, 'status': 'error'}
        
    def mkday_probability_data_process(self, ticker_name, stock_symbols_dict, df):
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values(by='Datetime').reset_index(drop=True)
        df['MinDatetime'] = df['Datetime'] - pd.DateOffset(months=1)
        df.set_index('Datetime', inplace=True)
        multi_data_processor = MultiDataProcessor(stock_symbols_dict, df)
        dct_data = multi_data_processor.multiprocessing()
        df = pd.DataFrame(dct_data)
        df['tickerName'] = ticker_name
        df = df.apply(lambda col: col.apply(json.dumps) if col.apply(lambda x: isinstance(x, (dict, list))).any() else col)
        df = df[['Datetime', 'tickerName', 'BuyInProfit MP::HP::MP::HP', 'SellInLoss MP::MP::LP::LP', 'BuyInLoss MP::HP::LP::HP', 'SellInProfit MP::HP::LP::LP', 'ProbabilityOfProfitMT2Percent', 'ProbabilityOfLoss1ratio3Percent', 'ProbabilityOfProfitTomorrow', 'ProbabilityOfLossTomorrow', 'ProbabilityOfProfitLoss', 'ProbabilityOfmaxHigh', 'ProbabilityOfmaxLow', 'ProbabilityOfpriceBand', 'ProbabilityOfCloseTolerance']]
        return df
    
class MultiDataProcessor:
    def __init__(self, stock_symbols_dict, df):
        self.stock_symbols_dict = stock_symbols_dict
        self.df = df
        
    def multiprocessing(self):
        MaxDatetime = self.stock_symbols_dict.get('MaxDatetime')
        DateRange = [(current_date, past_date) for current_date, past_date in zip(self.df.index, self.df['MinDatetime']) if pd.isna(MaxDatetime) or current_date >= MaxDatetime]
        result = [self.probability_process(arg) for arg in DateRange]
        return result

    def probability_process(self, arg):
        try:
            current_date, past_date = arg
            subset_df = self.df.iloc[(self.df.index >= past_date) & (self.df.index <= current_date)]
            dct = buy_sell_probability_in_profit_and_loss(subset_df)
            dct['Datetime'] = current_date
            return dct
        except:
            return {}
        
def buy_sell_probability_in_profit_and_loss(df):
    BuyInProfit = len(df[df['maxLow'] == 0])
    SellInLoss = len(df[df['maxHigh'] == 0])
    BuyInLoss = len(df[(df['maxLow'] != 0) & (df['maxHigh'] != 0) & (df['Open'] > df['Close'])])
    SellInProfit = len(df[(df['maxLow'] != 0) & (df['maxHigh'] != 0) & (df['Open'] < df['Close'])])
    Total = BuyInProfit+SellInLoss+BuyInLoss+SellInProfit
    ProbabilityOfCloseTolerance = df['closeTolerance'].astype(int).value_counts()
    ProbabilityOfCloseTolerance = round((ProbabilityOfCloseTolerance/ProbabilityOfCloseTolerance.sum())*100, 2).to_dict()
    ProbabilityOfProfitLoss = df['OC-P/L'].astype(int).value_counts()
    ProbabilityOfProfitLoss = round((ProbabilityOfProfitLoss/ProbabilityOfProfitLoss.sum())*100, 2).to_dict()  
    ProbabilityOfProfitLossTomorrow = {'Profit': round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key >= 0), 2), 'Loss': round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key < 0), 2)}
    ProbabilityOfProfitMT2Percent= round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key >= 2), 2)
    ProbabilityOfLoss1ratio3Percent= round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key <= -1), 2)
    ProbabilityOfmaxHigh = df['maxHigh'].astype(int).value_counts()
    ProbabilityOfmaxHigh = round((ProbabilityOfmaxHigh/ProbabilityOfmaxHigh.sum())*100, 2).to_dict()  
    ProbabilityOfmaxLow = df['maxLow'].astype(int).value_counts()
    ProbabilityOfmaxLow = round((ProbabilityOfmaxLow/ProbabilityOfmaxLow.sum())*100, 2).to_dict()  
    ProbabilityOfpriceBand = df['priceBand'].astype(int).value_counts()
    ProbabilityOfpriceBand = round((ProbabilityOfpriceBand/ProbabilityOfpriceBand.sum())*100, 2).to_dict() 
    buysellProbability = {
        'BuyInProfit MP::HP::MP::HP': round((BuyInProfit/Total)*100, 2) if Total != 0 else 0,
        'SellInLoss MP::MP::LP::LP': round((SellInLoss/Total)*100, 2) if Total != 0 else 0,
        'BuyInLoss MP::HP::LP::HP': round((BuyInLoss/Total)*100, 2) if Total != 0 else 0,
        'SellInProfit MP::HP::LP::LP': round((SellInProfit/Total)*100, 2) if Total != 0 else 0,
        'ProbabilityOfCloseTolerance': ProbabilityOfCloseTolerance,
        'ProbabilityOfProfitLoss': ProbabilityOfProfitLoss,
        'ProbabilityOfProfitTomorrow': ProbabilityOfProfitLossTomorrow.get('Profit'),
        'ProbabilityOfLossTomorrow': ProbabilityOfProfitLossTomorrow.get('Loss'),
        'ProbabilityOfProfitMT2Percent': ProbabilityOfProfitMT2Percent,
        'ProbabilityOfLoss1ratio3Percent': ProbabilityOfLoss1ratio3Percent,
        'ProbabilityOfmaxHigh': ProbabilityOfmaxHigh,
        'ProbabilityOfmaxLow': ProbabilityOfmaxLow,
        'ProbabilityOfpriceBand': ProbabilityOfpriceBand
        }
    return buysellProbability
    
def JobmkDayProbability():
    probability = mkDayProbability()
    stock_symbols = probability.fetch_max_dates()
    with Pool(processes=int(cpu_count() * 0.8)) as pool:
        result = list(tqdm(pool.imap(probability.update_mkday_probability, stock_symbols), total=len(stock_symbols), desc='Update Table mkDayProbability'))
    return result
    
if __name__ == "__main__":
    result = JobmkDayProbability()



