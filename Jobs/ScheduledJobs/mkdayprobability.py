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
from Scripts.dataprocess import buy_sell_probability_in_profit_and_loss


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
        df['MinDatetime'] = df['MaxDatetime'] - pd.DateOffset(months=1)
        return df.to_dict('records')
    
    def update_mkday_probability(self, stock_symbols_dict):
        ticker_name = stock_symbols_dict.get('tickerName')
        MinDatetime = stock_symbols_dict.get('MinDatetime')
        query = f'''
                SELECT * FROM {self.table_name_dfeature} 
                WHERE tickerName='{ticker_name}'
                '''
        if not pd.isna(MinDatetime):
            query += f"and Datetime >= '{MinDatetime}'"
        df = pd.read_sql(query, cnxn(self.db_name_analyzer))
        if pd.isna(stock_symbols_dict.get('MaxDatetime')) or df['Datetime'].max() > stock_symbols_dict.get('MaxDatetime'):
            df = self.mkday_probability_data_process(ticker_name, stock_symbols_dict, df)
            result = Data_Inserting_Into_DB(df, self.db_name_analyzer, self.table_name_dprobability, 'append')
            return {**result, 'Message': 'New Data Added', 'tickerName': ticker_name}
        return {'Message': 'Already Upto-Date', 'tickerName': ticker_name}

    def mkday_probability_data_process(self, ticker_name, stock_symbols_dict, df):
        df = df.sort_values(by='Datetime').reset_index(drop=True)
        df['MinDatetime'] = df['Datetime'] - pd.DateOffset(months=1)
        multi_data_processor = MultiDataProcessor(stock_symbols_dict, df)
        dct_data = multi_data_processor.multiprocessing(self.cpu_count)
        df = pd.DataFrame(dct_data)
        df['tickerName'] = ticker_name
        df = df.apply(lambda col: col.apply(json.dumps) if col.apply(lambda x: isinstance(x, (dict, list))).any() else col)
        df = df[['Datetime', 'tickerName', 'BuyInProfit MP::HP::MP::HP', 'SellInLoss MP::MP::LP::LP', 'BuyInLoss MP::HP::LP::HP', 'SellInProfit MP::HP::LP::LP', 'ProbabilityOfProfitMT2Percent', 'ProbabilityOfLoss1ratio3Percent', 'ProbabilityOfProfitTomorrow', 'ProbabilityOfLossTomorrow', 'ProbabilityOfProfitLoss', 'ProbabilityOfmaxHigh', 'ProbabilityOfmaxLow', 'ProbabilityOfpriceBand', 'ProbabilityOfCloseTolerance']]
        return df
    
class MultiDataProcessor:
    def __init__(self, stock_symbols_dict, df):
        self.stock_symbols_dict = stock_symbols_dict
        self.df = df
        
    def multiprocessing(self, cpu_count):
        grouped_df = self.df.groupby('Datetime')
        with Pool(processes=cpu_count) as pool:
            result = list(pool.imap(self.probability_process, grouped_df))
        return result

    def probability_process(self, arg):
        current_date, group = arg
        past_date = group['MinDatetime'].iloc[0]
        if pd.isna(self.stock_symbols_dict.get('MaxDatetime')) or current_date > self.stock_symbols_dict.get('MaxDatetime'):
            subset_df =self.df[(self.df['Datetime'] >= past_date) & (self.df['Datetime'] <= current_date)]
            dct = buy_sell_probability_in_profit_and_loss(subset_df)
            dct['Datetime'] = current_date
            return dct
        return {}

def JobmkDayProbability():
    probability = mkDayProbability()
    stock_symbols = probability.fetch_max_dates()
    result = [probability.update_mkday_probability(stock_symbol) for stock_symbol in tqdm(stock_symbols, desc='Update Table mkDayProbability')]
    return result
    
if __name__ == "__main__":
    result = JobmkDayProbability()










































# =============================================================================
# OLD CODE --> 29/03/2024
# =============================================================================
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import json
# from multiprocessing import Pool, cpu_count
# from functools import partial 

# import warnings
# warnings.filterwarnings('ignore')

# from Scripts.dbConnection import *
# from Scripts.dataprocess import *


# def mkDayProbability(stockSymbolsDict):
#     tickerName = stockSymbolsDict.get('tickerName')
#     Past_Datetime = stockSymbolsDict.get('Past_Datetime')
#     query = f'''
#             SELECT * FROM mkDay 
#             WHERE tickerName='{tickerName}'
#             '''
#     if not pd.isna(Past_Datetime):
#         query += f"and Datetime >= '{Past_Datetime}'"
#     df = pd.read_sql(query, cnxn('stockmarket'))
#     if pd.isna(stockSymbolsDict.get('Datetime')) or df['Datetime'].max() > stockSymbolsDict.get('Datetime'):           
#         df = mkDayProbability_Data_Process(tickerName, stockSymbolsDict, df)
#         result = Data_Inserting_Into_DB(df, 'stockmarket', 'mkDayProbability', 'append')
#         return {**result, 'Message': 'New Data Added', 'tickerName': tickerName}
#     return {'Message': 'Already Upto-Date', 'tickerName': tickerName}

# def mkDayProbability_Data_Process(tickerName, stockSymbolsDict, df):
#     df = df.sort_values(by='Datetime').reset_index(drop=True)    
#     df['Past_Datetime'] = df['Datetime'] - pd.DateOffset(months=1)
#     dctData = Multiprocessing(stockSymbolsDict, df)
#     df = pd.DataFrame(dctData)
#     df['tickerName'] = tickerName
#     df = df.apply(lambda col: col.apply(json.dumps) if col.apply(lambda x: isinstance(x, (dict, list))).any() else col)
#     df = df[['Datetime', 'tickerName', 'BuyInProfit MP::HP::MP::HP', 'SellInLoss MP::MP::LP::LP', 'BuyInLoss MP::HP::LP::HP', 'SellInProfit MP::HP::LP::LP', 'ProbabilityOfProfitMT2Percent', 'ProbabilityOfLoss1ratio3Percent', 'ProbabilityOfProfitTomorrow', 'ProbabilityOfLossTomorrow', 'ProbabilityOfProfitLoss', 'ProbabilityOfmaxHigh', 'ProbabilityOfmaxLow', 'ProbabilityOfpriceBand', 'ProbabilityOfCloseTolerance']]  
#     return df

# def Multiprocessing(stockSymbolsDict, df):
#     grouped_df = df.groupby('Datetime')
#     FetchPartial = partial(probability_process, stockSymbolsDict=stockSymbolsDict, df=df)
#     with Pool(processes=cpu_count()) as pool:
#         result = list(pool.imap(FetchPartial, grouped_df))
#     return result

# def probability_process(arg, stockSymbolsDict, df):
#     current_date, group = arg
#     past_date = group['Past_Datetime'].iloc[0] 
#     if pd.isna(stockSymbolsDict.get('Datetime')) or current_date > stockSymbolsDict.get('Datetime'):
#         subset_df = df[(df['Datetime'] >= past_date) & (df['Datetime'] <= current_date)]
#         dct = buy_sell_probability_in_profit_and_loss(subset_df)
#         dct['Datetime'] = current_date
#         return dct
#     return {}

# if __name__ == "__main__":
#     try:
#         query = f'''
#                 SELECT md.tickerName, max(mp.Datetime) as Datetime FROM mkDay md
#                 LEFT JOIN mkDayProbability mp ON md.tickerName = mp.tickerName
#                 group by md.tickerName
#                 '''
#         dfS = pd.read_sql(query, cnxn('stockmarket'))
#     except Exception as e:
#         query = '''
#             SELECT md.tickerName, max(md.Datetime) as Datetime FROM mkDay md
#             group by md.tickerName
#             '''
#         dfS = pd.read_sql(query, cnxn('stockmarket'))
#         dfS['Datetime'] = np.datetime64('NaT')
#     dfS['Past_Datetime'] = dfS['Datetime'] - pd.DateOffset(months=1)
#     dfSDict = dfS.to_dict('records')
#     result = [mkDayProbability(stockSymbolsDict) for stockSymbolsDict in tqdm(dfSDict, desc='dayprobability table updating')]


