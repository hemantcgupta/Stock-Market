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
from copy import deepcopy

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
        return pd.concat(result).reset_index(drop=True)
        
    def fetch_db_data(self, tickerName):
        query = f'select * from [{tickerName}]'
        df = pd.read_sql(query, cnxn(self.db_name_day))
        df['tickerName'] = tickerName
        return df
 
if __name__ == "__main__":
    analyzer = mkDayAnalyzer()
    analyzer.validation()
    result = analyzer.mkDayAnalyzerMain()
    

df = deepcopy(result)



