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
        self.db_name_day = 'mkgrowwdaymaster'
        self.db_name_analyzer = 'mkanalyzer'
        self.table_name_dfeature = 'mkDayFeature'
        self.table_name_dseasonality = 'mkDaySeasonality'
        self.FeatureMaxDict, self.SeasonalityMaxDict = self.previous_max_details()
        
    def validation(self):
        create_database(self.db_name_day)
        create_database(self.db_name_analyzer)
        
    def previous_max_details(self):
        try:
            query = f"select tickerName, max(Datetime) as Datetime from {self.table_name_dfeature} group by tickerName order by tickerName"
            dfF = pd.read_sql(query, cnxn('mkanalyzer'))
            FeatureMaxDict = dict(zip(dfF['tickerName'], pd.to_datetime(dfF['Datetime']).dt.strftime('%Y-%m-%d')))
        except:
            FeatureMaxDict = {}
        try:
            query = f"select tickerName, max([Year]) as Year from {self.table_name_dseasonality} group by tickerName order by tickerName "
            dfS = pd.read_sql(query, cnxn('mkanalyzer'))
            SeasonalityMaxDict = dict(zip(dfS['tickerName'], dfS['Year']))
        except:
            SeasonalityMaxDict = {}
        return FeatureMaxDict, SeasonalityMaxDict
        
        
    def mkDayAnalyzerMain(self):
        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        stockSymbols = pd.read_sql(query, cnxn(self.db_name_day))['TABLE_NAME'].tolist()
        # stockSymbols = ['BLS']
        result = self.mkDay_Data_Process(stockSymbols)
        return result
    
    def mkDay_Data_Process(self, stockSymbols):
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.fetch_db_data, stockSymbols), total=len(stockSymbols), desc='Updating mkDayFeature and mkDaySeasonality'))
        # result = [self.fetch_db_data(tickerName) for tickerName in stockSymbols]
        result_feature = self.update_table(result, self.db_name_analyzer, self.table_name_dfeature, 'append')
        result_seasonality = self.update_table(result, self.db_name_analyzer, self.table_name_dseasonality, 'append')
        return {'feature': result_feature, 'seasonality': result_seasonality}
        
    def fetch_db_data(self, tickerName='BLS'):
        query = f'select * from [{tickerName}]'
        if filterYear:= self.SeasonalityMaxDict.get(tickerName):
            query += f" where YEAR(cast(Datetime as Date)) >= '{filterYear}'"
        df = pd.read_sql(query, cnxn(self.db_name_day))
        df = self.adding_year_month_daynumber(df)
        df['tickerName'] = tickerName
        df = self.fetureExtraction(df)
        dfSC = self.mk_day_seasonality(df, tickerName)
        if filterDate:= self.FeatureMaxDict.get(tickerName):
            df = df[df['Datetime'] >= filterDate].reset_index(drop=True)
            resultDF = self.Delete_max_date(self.db_name_analyzer, tickerName, filterDate)
        if filterYear:= self.SeasonalityMaxDict.get(tickerName):
            dfSC = dfSC[dfSC['Year'] >= filterYear].reset_index(drop=True)
            resultDS = self.Delete_max_year(self.db_name_analyzer, tickerName, filterYear)
        if not df.empty:
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return {'tickerName': tickerName, self.table_name_dfeature: df, self.table_name_dseasonality: dfSC}
    
    def adding_year_month_daynumber(self, df):
        df['Datetime'] = pd.to_datetime(df['Datetime'])
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

    def Delete_max_date(self, dbName, tickerName, filterDate):
        try:
            conn = cnxn(dbName)
            cursor = conn.cursor()
            delete_query = f"DELETE FROM {self.table_name_dfeature} WHERE tickerName = '{tickerName}' and cast(Datetime as date) >= '{filterDate}';"
            cursor.execute(delete_query)
            conn.commit()
            return {'tickerName': tickerName, 'status': 'success'}
        except:
            return {'tickerName': tickerName, 'status': 'error'}
        
    def Delete_max_year(self, dbName, tickerName, filterYear):
        try:
            conn = cnxn(dbName)
            cursor = conn.cursor()
            delete_query = f"DELETE FROM {self.table_name_dseasonality} WHERE tickerName = '{tickerName}' and cast([Year] as INTEGER) >= '{filterYear}';"
            cursor.execute(delete_query)
            conn.commit()
            return {'tickerName': tickerName, 'status': 'success'}
        except Exception as e:
            print(e)
            return {'tickerName': tickerName, 'status': 'error'}
        
    # def update_table(self, result, dbName, tableName, insertMethod):
    #     result = [
    #         Data_Inserting_Into_DB(
    #             dct.get(tableName), dbName, tableName, insertMethod
    #         ) if isinstance(dct.get(tableName), pd.DataFrame) and not dct.get(tableName).empty else {
    #             'dbName': dbName,
    #             dct.get('tickerName'): 'Unsuccessful - Empty or None DataFrame'
    #         }
    #         for dct in tqdm(result, desc=f'Update Table {tableName}')
    #     ]
    #     return result
        
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



