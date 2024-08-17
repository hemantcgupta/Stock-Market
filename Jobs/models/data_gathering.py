# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:31:45 2024

@author: Hemant
"""
import pandas as pd
import os
import re
from datetime import datetime
from typing import Any
from Scripts.dbConnection import cnxn 

class DATA_GATHERING:
    def __init__(self, kwargs: dict[str: Any]) -> None:  
        self.db_name_day = 'mkdaymaster'
        self.db_name_analyzer = 'mkanalyzer'
        self.table_name_ifeature = 'mkIntervalFeature'
        self.table_name_simulationPrediction = 'simulationPrediction'
        self.tickerName = kwargs.get('tickerName')
        self.filterDate = kwargs.get('Datetime')
        self.df = None
        self.dfPred = None
        
    def __str__(self) -> str:
        return f'Ticker({self.tickerName})'
        
    def query(self) -> pd.DataFrame:
        query = f'''
        SELECT 
            CONVERT(date, o.Datetime) AS Date, 
            CASE
                WHEN MAX(o.[Close]) = 0 THEN NULL
                ELSE (MAX(o.High) - MIN(o.Low)) / MAX(o.[Close])
            END AS BASR_Day,
            SUM(o.Volume) / COUNT(DISTINCT CONVERT(date, o.Datetime)) AS ATV_Day,
            CASE
                WHEN MAX(o.[Close]) = 0 OR SUM(o.Volume) = 0 THEN NULL
                ELSE ((MAX(o.High) - MIN(o.Low)) / MAX(o.[Close])) * LOG(SUM(o.Volume) / COUNT(DISTINCT CONVERT(date, o.Datetime)))
            END AS LS_Day,
        	mif.nCandleBelowOpen, mif.pCandleAboveOpen, mif.nCandle, mif.pCandle, mif.Hits44MA,
        	case when (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) then 'YES' else 'NO' end as entry_close,
        	case when (sp.Entry2 <= sp.predTmEntry2 AND sp.Exit2 >= sp.predTmExit2) then 'YES' else 'NO' end as entry_exit,
        	case when (sp.Entry2 <= sp.predTmEntry2 AND sp.predTmEntry2 >= sp.[Close]) then 'YES' else 'NO' end as entry_loss,
        	sp.Entry2, sp.Exit2, sp.predTmEntry2, sp.predTmExit2, round(sp.Entry2-sp.predTmEntry2, 2) AS diff, round(sp.predTmEntry2-sp.[Close], 2) AS loss,
        	round(sp.predTmEntry2-sp.[Low], 2) AS low_loss
        FROM 
        	[{self.tickerName}] AS o
        LEFT JOIN (select * from {self.db_name_analyzer}.dbo.{self.table_name_ifeature} where tickerName = '{self.tickerName}') AS mif ON mif.Datetime = o.Datetime
        LEFT JOIN (select * from {self.db_name_analyzer}.dbo.{self.table_name_simulationPrediction} where tickerName = '{self.tickerName}') AS sp ON sp.predDatetime = o.Datetime
        WHERE o.Datetime <= '{self.filterDate}'
        GROUP BY 
            CONVERT(date, o.Datetime), o.[Open], o.[High], o.[Low], o.[Close], o.[Volume], 
        	mif.nCandleBelowOpen, mif.pCandleAboveOpen, mif.nCandle, mif.pCandle, mif.Hits44MA,
        	sp.Entry2, sp.predTmEntry2, sp.Exit2, sp.predTmExit2, sp.[Close], sp.[Low]
        ORDER BY
            Date DESC
        '''
        df = pd.read_sql(query, cnxn(self.db_name_day))
        df = df[['Date', 'BASR_Day', 'ATV_Day', 'LS_Day', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA', 'entry_exit', 'entry_close', 'entry_loss']]
        return df

    def data_process(self, df: pd.DataFrame) -> None:
        df['BASR_Day'] = df['BASR_Day'].round(4)
        df['ATV_Day'] = df['ATV_Day'].round(4)
        df['LS_Day'] = df['LS_Day'].round(4)
        df['TmDate'] = df['Date']
        df['TmPL'] = 0
        df.loc[(df['entry_exit'] == 'YES') | (df['entry_close'] == 'YES'), 'TmPL'] = 1
        df['TmDate'] = df['TmDate'].shift(1)
        df['TmPL'] = df['TmPL'].shift(1)
        dfPred = df[['Date', 'BASR_Day', 'ATV_Day', 'LS_Day', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA', 'entry_exit', 'entry_close', 'entry_loss']].dropna().reset_index(drop=True)
        df = df[~((df['entry_exit'] == 'NO') & (df['entry_close'] == 'NO') & (df['entry_loss'] == 'NO') & (df['TmPL'] == 0))].reset_index(drop=True)
        self.dfPred = pd.merge(dfPred, df[['Date', 'TmPL']], how='left',on='Date')
        self.dfPred['TmPL'] = self.dfPred['TmPL'].fillna(0).astype(int)
        df = df[['Date', 'TmDate', 'BASR_Day', 'ATV_Day', 'LS_Day', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA', 'TmPL']]
        df = df.dropna().reset_index(drop=True)
        df['TmPL'] = df['TmPL'].astype(int)
        self.df = df
        
    def get_latest_file_with_highest_percent(self, file_prefix):
        files = os.listdir(self.base_path)
        pattern = re.compile(rf'{file_prefix}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d+\.\d+)\.joblib')
        file_info = []
        for file in files:
            match = pattern.match(file)
            if match:
                date_str = match.group(1)
                percent_value = float(match.group(2))
                date = datetime.strptime(date_str, '%Y-%m-%d')
                file_info.append((file, date, percent_value))
        file_info.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return file_info[0][0] if file_info else None 

    def define_base_path(self, model_name: str) -> None:
        base_path = os.path.join('models', model_name, self.tickerName)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.base_path = base_path