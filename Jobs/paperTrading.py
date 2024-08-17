# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 02:04:50 2024

@author: Hemant
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.set_option('expand_frame_repr', True)
from Scripts.dbConnection import cnxn 

class VAR:
    db_mkanalyzer = 'mkanalyzer'

def fetchData(inputDate):
    query = f'''
    WITH cte AS (
    SELECT 
        tas.Date, 
        tas.tickerName, 
        tas.[AccuracyScore:Mode],
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
            THEN 1 
            ELSE 0 
        END AS TmPL, 
        tas.TmPredPL,
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) 
            THEN 1 
            ELSE 0 
        END AS gotLoss,
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
            THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
            ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
        END AS ProfitPercent,
        CASE 
            WHEN Entry2 <= predTmEntry2
            THEN 1 
            ELSE 0 
        END AS gotEntry,
        round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
		Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
		[predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose
    FROM topAccurateStats AS tas
    LEFT JOIN simulationPrediction AS sp 
    ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
    WHERE tas.Date = '{inputDate}' AND tas.TmPredPL = 1
    ),
    cte1 AS (
        SELECT tickerName, COUNT(Datetime) AS counts 
        FROM simulationPrediction
        WHERE Entry2 <= predTmEntry2 
          AND Exit2 >= predTmExit2
          AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, '{inputDate}') AS DATE)
          AND CAST(Datetime AS DATE) <= '{inputDate}'
        GROUP BY tickerName
    )
    SELECT  * FROM cte c1 
    LEFT JOIN cte1 c2 ON c1.tickerName = c2.tickerName
    ORDER BY c2.counts DESC, [AccuracyScore:Mode] DESC
    '''
    df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))
    return df   

    
query = '''
SELECT DISTINCT(Datetime) FROM simulationPrediction
WHERE Datetime >= DATEADD(MONTH, -3, (SELECT MAX(Datetime) FROM simulationPrediction))
ORDER BY Datetime ASC
'''
dateList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Datetime'].dt.date.astype(str).tolist()
df = pd.concat([fetchData(inputDate) for inputDate in dateList]).fillna(0)

# Top 1
def top1(df):
    df_sorted = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode'], ascending=[True, False, False])
    df_sorted['RankValue'] = 0
    df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 100
    df = df_sorted.groupby('Date').head(1).reset_index(drop=True)
    df = df[df['gotEntry'] == 1].reset_index(drop=True)
    df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
    df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
    df['dayName'] = pd.to_datetime(df['Date']).dt.day_name()
    df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Date', 'dayName', 'tickerName', 'counts', 'RankValue', 'AccuracyScore:Mode', 'gotEntry', 'ActualEntry', 'PredEntry', 'ActulExit', 'PredExit', 'PredOpen', 'ActualOpen', 'ActualClose', 'AOpen/POpen-Diff', 'TmPL', 'TmPredPL', 'gotLoss', 'ProfitPercent', 'investedAmount', 'profitAmount'])
    print(df['profitAmount'].sum().round(2))
    print(df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index())
    return df

# Top 3
def top3(df):
   df_sorted = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode'], ascending=[True, False, False])
   df_sorted['RankValue'] = 0
   df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 50
   df_sorted.loc[df_sorted.groupby('Date').head(2).tail(1).index, 'RankValue'] = 30
   df_sorted.loc[df_sorted.groupby('Date').head(3).tail(1).index, 'RankValue'] = 20
   df = df_sorted.groupby('Date').head(3).reset_index(drop=True)
   df = df[df['gotEntry'] == 1].reset_index(drop=True)
   df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
   df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
   print(df['profitAmount'].sum().round(2))
   print(df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index())
   return df

df1 = top1(df)
df3 = top3(df)
print(df1['dayName'].value_counts())

