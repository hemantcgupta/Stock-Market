# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:18:12 2025

@author: heman
"""


import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn

class VAR:
    cpu_count = int(cpu_count() * 0.8)
    db_name_master = 'master'
    db_name_day = 'mkdaymaster'
    db_name_interval = 'mkintervalmaster'
    db_name_analyzer = 'mkanalyzer'
    db_name_ianalyzer = 'mkintervalanalyzer'



class MaterTableValidation_RemovalOfDate:
    def __init__(self):
        self.cpu_count = int(cpu_count() * 0.8)
        self.dbList = [VAR.db_name_day, VAR.db_name_interval, VAR.db_name_ianalyzer]
        self.tableList = ['mkDayProbability', 'mkDayMA', 'mkIntervalFeature', 'mkDayPrediction', 'simulationPrediction', 'simulationPredMSE', 'mkTopPrediction']
        
    def validation(self):
        create_database(VAR.db_name_day)
        create_database(VAR.db_name_interval)
        
    def mkDayAnalyzerMain(self):
        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        stockSymbols = pd.read_sql(query, cnxn(VAR.db_name_day))['TABLE_NAME'].tolist()
        result = self.mkDay_Data_Process(stockSymbols)
        return result
    
    def mkDay_Data_Process(self, stockSymbols):
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.fetch_db_data, stockSymbols), total=len(stockSymbols), desc='Fetch DB data'))
        df = pd.concat(result).reset_index(drop=True)
        SymbolsDateList = df.groupby('tickerName') ['Date'].unique().reset_index().to_dict('records')
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.delete_records_from_tickers, SymbolsDateList), total=len(stockSymbols), desc='Delete query'))
        return result
        
    def fetch_db_data(self, tickerName):
        query = f'''
            SELECT DISTINCT cast(d.Datetime AS Date) as Date, '{tickerName}' as tickerName
            FROM {VAR.db_name_day}.dbo.[{tickerName}] d
            LEFT JOIN {VAR.db_name_interval}.dbo.[{tickerName}] i
                ON cast(d.Datetime AS DATE) = cast(i.Datetime AS DATE)
            WHERE i.Datetime IS NULL and Year(d.Datetime) >= 2024'''
        df = pd.read_sql(query, cnxn(VAR.db_name_master))
        df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
        return df
    

    def delete_records_from_tickers(self, SymbolsDateDict):
        try:
            tickerName = SymbolsDateDict.get("tickerName")
            dateList = SymbolsDateDict.get("Date")
            dateListStr =  ", ".join(["'"+item+"'" for item in dateList])
            conn = cnxn(VAR.db_name_master)
            cursor = conn.cursor()
            for dbname in self.dbList:
                delete_query = f"""
                DELETE FROM {dbname}.dbo.[{tickerName}]
                WHERE CAST(Datetime AS DATE) IN ({dateListStr})
                """
                cursor.execute(delete_query)
                conn.commit()
                # print(delete_query)
            for tablename in self.tableList:
                delete_query = f"""
                DELETE FROM {VAR.db_name_analyzer}.dbo.[{tablename}]
                WHERE tickerName = '{tickerName}' AND CAST(Datetime AS DATE) IN ({dateListStr})
                """
                cursor.execute(delete_query)
                conn.commit()
                # print(delete_query)
                
            return {**SymbolsDateDict, 'status': 'success'}
        except:
            return {**SymbolsDateDict, 'status': 'error'}
    
def JobValidationRemoval():
    analyzer = MaterTableValidation_RemovalOfDate()
    analyzer.validation()
    result = analyzer.mkDayAnalyzerMain()
    return result
           
if __name__ == "__main__":
    result = JobValidationRemoval()
    
