# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 01:43:47 2024

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.set_option('expand_frame_repr', True)
from Scripts.dbConnection import cnxn 

class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkintervalmaster = 'mkintervalmaster'

def predEntryDatetime(row):
    query = f'''
    SELECT TOP 1 '{row['Date']}' as Date, Datetime as predEntryDatetime, '{row['tickerName']}' as tickerName
    FROM [{row['tickerName']}]
    WHERE CAST(Datetime AS DATE) = '{row['Datetime']}'
    ORDER BY ABS([Open] - {row['predTmEntry2']}), ABS(High - {row['predTmEntry2']}), ABS(Low - {row['predTmEntry2']}), ABS([Close] - {row['predTmEntry2']})
    '''
    df = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster))
    return df

def mainEntryTime():
    query = '''
    select sp.Datetime as Date, sp.predDatetime as Datetime, tas.tickerName, sp.predTmEntry2  from topAccurateStats tas
    left join simulationPrediction sp on sp.tickerName = tas.tickerName and cast(sp.predDatetime AS DATE) = cast(tas.Date AS DATE)
    '''
    df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))
    df = df.sort_values(by='Datetime', ascending=False).reset_index(drop=True)
    dfET = pd.concat([predEntryDatetime(row) for row in tqdm(df.to_dict('records'))]).reset_index(drop=True)
    return dfET
dfET = mainEntryTime()
dfET[(dfET['Date'] == '2024-08-16') & (dfET['tickerName'] == 'AWL')]
df1['Date'] = pd.to_datetime(df1['Date'])
dfET['Date'] = pd.to_datetime(dfET['Date'])
df1.merge(dfET, how='left', on=['Date', 'tickerName'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.loc[:, ~df.columns.duplicated()]
df.merge(dfET, how='left', on=['Date', 'tickerName']).tail(30)
