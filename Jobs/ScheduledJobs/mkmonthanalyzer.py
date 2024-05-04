# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:12:38 2024

@author: Hemant
"""
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn

class mkMonthAnalyzer:
    def __init__(self):
        self.cpu_count = int(cpu_count() * 0.8)
        self.db_name_day = 'mkdaymaster'
        self.db_name_analyzer = 'mkanalyzer'
        self.table_name_mfeature = 'mkMonthFeature'
        self.table_name_msummary = 'mkMonthSummary'
        self.table_name_mseasonality = 'mkMonthSeasonality'
        
    def validation(self):
        create_database(self.db_name_day)
        create_database(self.db_name_analyzer)
        
    def mkMonthAnalyzerMain(self):
        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        stockSymbols = pd.read_sql(query, cnxn(self.db_name_day))['TABLE_NAME'].tolist()
        result = self.mkMonth_Data_Process(stockSymbols)
        return result
        
    def mkMonth_Data_Process(self, stockSymbols):
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.fetch_db_data, stockSymbols), total=len(stockSymbols), desc='Updating mkMonthFeature, mkMonthSummary, and mkMonthSeasonality'))
        result_feature = self.update_table(result, self.db_name_analyzer, self.table_name_mfeature, 'replace')
        result_summary = self.update_table(result, self.db_name_analyzer, self.table_name_msummary, 'replace')
        result_seasonality = self.update_table(result, self.db_name_analyzer, self.table_name_mseasonality, 'replace')
        return {'feature': result_feature, 'summary': result_summary, 'seasonality': result_seasonality}
        
    def fetch_db_data(self, tickerName):
        query = f'select * from [{tickerName}]'
        df = pd.read_sql(query, cnxn(self.db_name_day)).drop(columns=['Volume'])
        df = self.conversion_into_month(df)
        df['tickerName'] = tickerName
        df = self.fetureExtraction(df)
        dfS = self.mk_month_summary(df, tickerName)
        dfSC = self.mk_month_seasonality(df, tickerName)
        return {'tickerName': tickerName, self.table_name_mfeature: df, self.table_name_msummary: dfS, self.table_name_mseasonality: dfSC}
        
    def conversion_into_month(self, df):
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        df = df.groupby(['Year', 'Month']).agg(agg_funcs).reset_index()
        return df
    
    def fetureExtraction(self, df):
        df['PvClose'] = df['Close'].shift(1).fillna(df['Open'])
        df['OC-P/L'] = ((1-df['Open']/df['Close'])*100).round(2)   
        df['PvCC-P/L'] = ((1-df['PvClose']/df['Close'])*100).round(2)
        df['maxHigh'] = ((df['High']/df['Open']-1)*100).round(2)
        df['maxLow'] = ((df['Low']/df['Open']-1)*100).round(2)
        df['Open-PvClose'] = (df['Open']-df['PvClose']).round(2)  
        df['closeTolerance'] = df.apply(lambda row: row['OC-P/L'] - row['maxHigh'] if row['OC-P/L'] > 0 else row['OC-P/L'] - row['maxLow'] if row['OC-P/L'] < 0 else 0, axis=1).round(2)  
        df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
        df = df[['Year', 'Month', 'tickerName', 'Open', 'High', 'Low', 'Close', 'PvClose', 'OC-P/L', 'PvCC-P/L', 'Open-PvClose', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
        return df
        
    def mk_month_summary(self, df, tickerName):
        dfS = df[['Year', 'Month', 'OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
        monthNames = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        dfS['Month'] = dfS['Month'].map(monthNames)
        dfS = pd.melt(dfS, id_vars=['Year', 'Month'], value_vars=['OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand'], var_name='Features', value_name='Value')
        dfS = dfS.pivot_table(index=['Year', 'Features'], columns='Month', values=['Value'])
        dfS.columns = dfS.columns.levels[1]
        dfS = dfS.reset_index()
        colSequence = [item for item in ['Year', 'Features']+list(monthNames.values()) if item in dfS.columns]
        dfS = dfS[colSequence]
        dfS['tickerName'] = tickerName
        return dfS
    
    def mk_month_seasonality(self, df, tickerName):
        dfSC = df[['Year', 'Month', 'tickerName', 'Open', 'High', 'Low', 'Close']]
        grouped = dfSC.groupby('Year')
        dfSC['Open'] = grouped['Open'].transform('first')
        dfSC['High'] = grouped['High'].cummax()
        dfSC['Low'] = grouped['Low'].cummin()
        dfSC = self.fetureExtraction(dfSC)
        dfSC = self.mk_month_summary(dfSC, tickerName)
        return dfSC
        
    def update_table(self, result, dbName, tableName, insertMethod):
        df = pd.concat([dct.get(tableName) for dct in result]).reset_index(drop=True)
        result = Data_Inserting_Into_DB(df, dbName, tableName, insertMethod)
        return result
    
def JobmkMonthAnalyzer():
    analyzer = mkMonthAnalyzer()
    analyzer.validation()
    result = analyzer.mkMonthAnalyzerMain()
    return result
    
if __name__ == "__main__":
    analyzer = mkMonthAnalyzer()
    analyzer.validation()
    result = analyzer.mkMonthAnalyzerMain()







# =============================================================================
# 30/03/2024
# =============================================================================
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# import warnings
# warnings.filterwarnings('ignore')

# from Scripts.dbConnection import *

# def fetch_db_data(tickerName):
#     query = f'select * from [{tickerName}]'
#     df = pd.read_sql(query, cnxn('stockmarketday')).drop(columns=['Volume'])
#     df = conversion_into_month(df)
#     df['tickerName'] = tickerName
#     df = fetureExtraction(df)
#     dfS = mk_month_summary(df, tickerName)
#     dfSC = mk_month_seasonality(df, tickerName)
#     return {'tickerName': tickerName, 'mkMonth': df, 'mkMonthSummary': dfS, 'mkMonthSeasonality': dfSC}

# def conversion_into_month(df):
#     df['Year'] = df['Datetime'].dt.year
#     df['Month'] = df['Datetime'].dt.month
#     agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
#     df = df.groupby(['Year', 'Month']).agg(agg_funcs).reset_index()
#     return df
    
# def fetureExtraction(df):
#     df['PvClose'] = df['Close'].shift(1).fillna(df['Open'])
#     df['OC-P/L'] = ((1-df['Open']/df['Close'])*100).round(2)   
#     df['PvCC-P/L'] = ((1-df['PvClose']/df['Close'])*100).round(2)
#     df['maxHigh'] = ((df['High']/df['Open']-1)*100).round(2)
#     df['maxLow'] = ((df['Low']/df['Open']-1)*100).round(2)
#     df['Open-PvClose'] = (df['Open']-df['PvClose']).round(2)  
#     df['closeTolerance'] = df.apply(lambda row: row['OC-P/L'] - row['maxHigh'] if row['OC-P/L'] > 0 else row['OC-P/L'] - row['maxLow'] if row['OC-P/L'] < 0 else 0, axis=1).round(2)  
#     df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
#     df = df[['Year', 'Month', 'tickerName', 'Open', 'High', 'Low', 'Close', 'PvClose', 'OC-P/L', 'PvCC-P/L', 'Open-PvClose', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
#     return df

# def mk_month_summary(df, tickerName):
#     dfS = df[['Year', 'Month', 'OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand']]
#     monthNames = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
#     dfS['Month'] = dfS['Month'].map(monthNames)
#     dfS = pd.melt(dfS, id_vars=['Year', 'Month'], value_vars=['OC-P/L', 'PvCC-P/L', 'closeTolerance', 'maxHigh', 'maxLow', 'priceBand'], var_name='Features', value_name='Value')
#     dfS = dfS.pivot_table(index=['Year', 'Features'], columns='Month', values=['Value'])
#     dfS.columns = dfS.columns.levels[1]
#     dfS = dfS.reset_index()
#     colSequence = [item for item in ['Year', 'Features']+list(monthNames.values()) if item in dfS.columns]
#     dfS = dfS[colSequence]
#     dfS['tickerName'] = tickerName
#     return dfS

# def mk_month_seasonality(df, tickerName):
#     dfSC = df[['Year', 'Month', 'tickerName', 'Open', 'High', 'Low', 'Close']]
#     grouped = dfSC.groupby('Year')
#     dfSC['Open'] = grouped['Open'].transform('first')
#     dfSC['High'] = grouped['High'].cummax()
#     dfSC['Low'] = grouped['Low'].cummin()
#     dfSC = fetureExtraction(dfSC)
#     dfSC = mk_month_summary(dfSC, tickerName)
#     return dfSC


# def update_table(result, dbName, tableName, insertMethod):
#     df = pd.concat([dct.get(tableName) for dct in result]).reset_index(drop=True)
#     result = Data_Inserting_Into_DB(df, dbName, tableName, insertMethod)
    
    
# def MKmonthMain():
#     query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
#     stockSymbols = pd.read_sql(query, cnxn('stockmarketday'))['TABLE_NAME'].tolist()
#     with Pool(processes=cpu_count()) as pool:
#         result = list(tqdm(pool.imap(fetch_db_data, stockSymbols), total=len(stockSymbols)))
#     update_table(result, 'stockmarket', 'mkMonth', 'replace')
#     update_table(result, 'stockmarket', 'mkMonthSummary', 'replace')
#     update_table(result, 'stockmarket', 'mkMonthSeasonality', 'replace')
#     return result

    
    
# def mk_month_seasonality(df, tickerName):
#     dfSC = df[['Year', 'Month', 'tickerName', 'Open', 'High', 'Low', 'Close']]
#     grouped = dfSC.groupby('Year')
#     dfSC['Open'] = grouped['Open'].transform('first')    
#     dfSC['High'] = grouped['High'].transform(lambda x: x[:x.index.max() + 1].max())
#     dfSC['Low'] = grouped['Low'].transform(lambda x: x[:x.index.max() + 1].min())
#     dfSC = fetureExtraction(dfSC)
#     dfSC = mk_month_summary(dfSC, tickerName)
#     return dfSC    