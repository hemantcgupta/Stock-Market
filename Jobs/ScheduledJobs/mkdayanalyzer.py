# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:11:04 2024

@author: Hemant
"""

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn

class mkDayAnalyzer:
    def __init__(self):
        self.cpu_count = int(cpu_count() * 0.8)
        self.db_name_day = 'mkdaymaster'
        self.db_name_analyzer = 'mkanalyzer'
        self.table_name_dfeature = 'mkDayFeature'
        self.table_name_dseasonality = 'mkDaySeasonality'
        
    def validation(self):
        create_database(self.db_name_day)
        create_database(self.db_name_analyzer)
        
    def mkDayAnalyzerMain(self):
        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        stockSymbols = pd.read_sql(query, cnxn(self.db_name_day))['TABLE_NAME'].tolist()
        result = self.mkDay_Data_Process(stockSymbols)
        return result
    
    def mkDay_Data_Process(self, stockSymbols):
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.fetch_db_data, stockSymbols), total=len(stockSymbols), desc='Updating mkDayFeature and mkDaySeasonality'))
        result_feature = self.update_table(result, self.db_name_analyzer, self.table_name_dfeature, 'replace')
        result_seasonality = self.update_table(result, self.db_name_analyzer, self.table_name_dseasonality, 'replace')
        return {'feature': result_feature, 'seasonality': result_seasonality}
        
    def fetch_db_data(self, tickerName):
        query = f'select * from [{tickerName}]'
        df = pd.read_sql(query, cnxn(self.db_name_day))
        df = self.adding_year_month_daynumber(df)
        df['tickerName'] = tickerName
        df = self.fetureExtraction(df)
        dfSC = self.mk_day_seasonality(df, tickerName)
        return {'tickerName': tickerName, self.table_name_dfeature: df, self.table_name_dseasonality: dfSC}
    
    def adding_year_month_daynumber(self, df):
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        df['MM-DD'] = df['Datetime'].dt.strftime('%m-%d')
        return df
    
    def fetureExtraction(self, df):
        df['Day'] = df['Datetime'].dt.day_name()
        df['PvClose'] = df['Close'].shift(1).fillna(df['Open'])
        df['OC-P/L'] = ((1-df['Open']/df['Close'])*100).round(2)   
        df['PvCC-P/L'] = ((1-df['PvClose']/df['Close'])*100).round(2)
        df['maxHigh'] = ((df['High']/df['Open']-1)*100).round(2)
        df['maxLow'] = ((df['Low']/df['Open']-1)*100).round(2)
        df['Open-PvClose'] = (df['Open']-df['PvClose']).round(2)  
        df['closeTolerance'] = df.apply(lambda row: row['OC-P/L'] - row['maxHigh'] if row['OC-P/L'] > 0 else row['OC-P/L'] - row['maxLow'] if row['OC-P/L'] < 0 else 0, axis=1).round(2)  
        df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
        df = df[['Year', 'Month', 'MM-DD', 'Datetime', 'Day', 'tickerName', 'Open', 'High', 'Low', 'Close', 'PvClose', 'OC-P/L', 'PvCC-P/L', 'Open-PvClose', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
        return df
    
    def mk_day_seasonality(self, df, tickerName):
        dfSC = df[['Year', 'Month', 'MM-DD', 'Datetime', 'tickerName', 'Open', 'High', 'Low', 'Close']]
        grouped = dfSC.groupby('Year')
        dfSC['Open'] = grouped['Open'].transform('first')
        dfSC['High'] = grouped['High'].cummax()
        dfSC['Low'] = grouped['Low'].cummin()    
        dfSC = self.fetureExtraction(dfSC)
        dfSC = self.mk_day_summary(dfSC, tickerName)
        return dfSC
    
    def mk_day_summary(self, df, tickerName):
        dfS = df[['Year', 'MM-DD', 'OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
        dfS = pd.melt(dfS, id_vars=['Year', 'MM-DD'], value_vars=['OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand'], var_name='Features', value_name='Value')
        dfS = dfS.pivot_table(index=['Year', 'Features'], columns='MM-DD', values=['Value'])
        dfS.columns = dfS.columns.levels[1]
        dfS = dfS.reset_index()
        colSequence = [item for item in ['Year', 'Features']+sorted([item for item in dfS.columns if item not in ['Year', 'Features']]) if item in dfS.columns]
        dfS = dfS[colSequence]
        dfS['tickerName'] = tickerName
        return dfS

    def update_table(self, result, dbName, tableName, insertMethod):
        df = pd.concat([dct.get(tableName) for dct in result]).reset_index(drop=True)
        result = Data_Inserting_Into_DB(df, dbName, tableName, insertMethod)
        return result
    
def JobmkDayAnalyzer():
    analyzer = mkDayAnalyzer()
    analyzer.validation()
    result = analyzer.mkDayAnalyzerMain()
    return result
           
if __name__ == "__main__":
    analyzer = mkDayAnalyzer()
    analyzer.validation()
    result = analyzer.mkDayAnalyzerMain()






# =============================================================================
# OLD CODE --> 30/03/2024
# =============================================================================
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import json
# from multiprocessing import Pool, cpu_count

# import warnings
# warnings.filterwarnings('ignore')

# from Scripts.dbConnection import *
# from Scripts.dataprocess import *

# def fetch_db_data(tickerName):
#     query = f'select * from [{tickerName}]'
#     df = pd.read_sql(query, cnxn('stockmarketday'))
#     df = adding_year_month_daynumber(df)
#     df['tickerName'] = tickerName
#     df = fetureExtraction(df)
#     dfSC = mk_day_seasonality(df, tickerName)
#     return {'tickerName': tickerName, 'mkDay': df, 'mkDaySeasonality': dfSC}

# def adding_year_month_daynumber(df):
#     df['Year'] = df['Datetime'].dt.year
#     df['Month'] = df['Datetime'].dt.month
#     df['MM-DD'] = df['Datetime'].dt.date.astype(str).str.split('-', n=1).str[-1]
#     return df

# def fetureExtraction(df):
#     df['Day'] = df['Datetime'].dt.day_name()
#     df['PvClose'] = df['Close'].shift(1).fillna(df['Open'])
#     df['OC-P/L'] = ((1-df['Open']/df['Close'])*100).round(2)   
#     df['PvCC-P/L'] = ((1-df['PvClose']/df['Close'])*100).round(2)
#     df['maxHigh'] = ((df['High']/df['Open']-1)*100).round(2)
#     df['maxLow'] = ((df['Low']/df['Open']-1)*100).round(2)
#     df['Open-PvClose'] = (df['Open']-df['PvClose']).round(2)  
#     df['closeTolerance'] = df.apply(lambda row: row['OC-P/L'] - row['maxHigh'] if row['OC-P/L'] > 0 else row['OC-P/L'] - row['maxLow'] if row['OC-P/L'] < 0 else 0, axis=1).round(2)  
#     df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
#     df = df[['Year', 'Month', 'MM-DD', 'Datetime', 'Day', 'tickerName', 'Open', 'High', 'Low', 'Close', 'PvClose', 'OC-P/L', 'PvCC-P/L', 'Open-PvClose', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
#     return df

# def mk_day_summary(df, tickerName):
#     dfS = df[['Year', 'MM-DD', 'OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
#     dfS = pd.melt(dfS, id_vars=['Year', 'MM-DD'], value_vars=['OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand'], var_name='Features', value_name='Value')
#     dfS = dfS.pivot_table(index=['Year', 'Features'], columns='MM-DD', values=['Value'])
#     dfS.columns = dfS.columns.levels[1]
#     dfS = dfS.reset_index()
#     colSequence = [item for item in ['Year', 'Features']+sorted([item for item in dfS.columns if item not in ['Year', 'Features']]) if item in dfS.columns]
#     dfS = dfS[colSequence]
#     dfS['tickerName'] = tickerName
#     return dfS

# def mk_day_seasonality(df, tickerName):
#     dfSC = df[['Year', 'Month', 'MM-DD', 'Datetime', 'tickerName', 'Open', 'High', 'Low', 'Close']]
#     grouped = dfSC.groupby('Year')
#     dfSC['Open'] = grouped['Open'].transform('first')
#     dfSC['High'] = grouped['High'].cummax()
#     dfSC['Low'] = grouped['Low'].cummin()    
#     dfSC = fetureExtraction(dfSC)
#     dfSC = mk_day_summary(dfSC, tickerName)
#     return dfSC

# def update_table(result, dbName, tableName, insertMethod):
#     df = pd.concat([dct.get(tableName) for dct in result]).reset_index(drop=True)
#     result = Data_Inserting_Into_DB(df, dbName, tableName, insertMethod)
#     return df
    
# def mkDay_Data_Process(stockSymbols):
#     with Pool(processes=cpu_count()) as pool:
#         result = list(tqdm(pool.imap(fetch_db_data, stockSymbols), total=len(stockSymbols)))
#     dfD = update_table(result, 'stockmarket', 'mkDay', 'replace')
#     dfS = update_table(result, 'stockmarket', 'mkDaySeasonality', 'replace')
#     return dfD
    
# def mkDayProbability_Data_Process(arg):
#     tickerName, df = arg
#     df = df.sort_values(by='Datetime').reset_index(drop=True)    
#     df['Past_Datetime'] = df['Datetime'] - pd.DateOffset(months=1)
#     grouped_df = df.groupby('Datetime')
#     dctData = []
#     for current_date, group in grouped_df:
#         past_date = group['Past_Datetime'].iloc[0] 
#         subset_df = df[(df['Datetime'] >= past_date) & (df['Datetime'] <= current_date)]
#         dct = buy_sell_probability_in_profit_and_loss(subset_df)
#         dct['Datetime'] = current_date
#         dctData.append(dct)
#     df = pd.DataFrame(dctData)
#     df['tickerName'] = tickerName
#     df = df.apply(lambda col: col.apply(json.dumps) if col.apply(lambda x: isinstance(x, (dict, list))).any() else col)
#     df = df[['Datetime', 'tickerName', 'BuyInProfit MP::HP::MP::HP', 'SellInLoss MP::MP::LP::LP', 'BuyInLoss MP::HP::LP::HP', 'SellInProfit MP::HP::LP::LP', 'ProbabilityOfProfitMT2Percent', 'ProbabilityOfLoss1ratio3Percent', 'ProbabilityOfProfitTomorrow', 'ProbabilityOfLossTomorrow', 'ProbabilityOfProfitLoss', 'ProbabilityOfmaxHigh', 'ProbabilityOfmaxLow', 'ProbabilityOfpriceBand', 'ProbabilityOfCloseTolerance']]  
#     return {'mkDayProbability': df}

# def mkDayProbability(df):
#     grouped_df = df.groupby('tickerName')
#     with Pool(processes=cpu_count()) as pool:
#         result = list(tqdm(pool.imap(mkDayProbability_Data_Process, grouped_df), total=len(grouped_df)))
#     dfP = update_table(result, 'stockmarket', 'mkDayProbability', 'replace')
#     return dfP

# def MKdayMain():
#     query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
#     stockSymbols = pd.read_sql(query, cnxn('stockmarketday'))['TABLE_NAME'].tolist()
#     dfD = mkDay_Data_Process(stockSymbols)
#     dfP = mkDayProbability(dfD)    
#     return 'Completed'
    