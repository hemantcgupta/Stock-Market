# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 02:04:50 2024

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
            WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
            THEN 0
            WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) OR (predTmEntry2 >= [Close])
            THEN 1 
            ELSE 0 
        END AS gotLoss,
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
            THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
            ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
        END AS ProfitPercent,
        CASE 
            WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
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
        SELECT sp.tickerName, COUNT(sp.Datetime) AS counts, '{inputDate}' as Datetime, ETEXProfit
        FROM simulationPrediction AS sp
        LEFT JOIN (SELECT tickerName, EtEx2Profit as ETEXProfit FROM simulationPrediction WHERE Datetime='{inputDate}') p ON p.tickerName=sp.tickerName
        WHERE Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2
        AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, '{inputDate}') AS DATE)
        AND CAST(Datetime AS DATE) < CAST('{inputDate}' AS DATE)
        GROUP BY sp.tickerName, ETEXProfit
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
df = pd.concat([fetchData(inputDate) for inputDate in tqdm(dateList)]).fillna(0)
df = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode', 'ETEXProfit'], ascending=[False, False, False, False])
df = df.loc[:, ~df.columns.duplicated()]
df.head(30)
df.head(5)['tickerName'].unique()
# Top 1
def top1(df):
    df_sorted = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode', 'ETEXProfit'], ascending=[True, False, False, False])
    df_sorted['RankValue'] = 0
    df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 100
    df = df_sorted.groupby('Date').head(1).reset_index(drop=True)
    df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
    df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
    df['dayName'] = pd.to_datetime(df['Date']).dt.day_name()
    df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Date', 'dayName', 'tickerName', 'TmPL', 'TmPredPL', 'gotEntry', 'gotLoss', 'counts', 'AccuracyScore:Mode', 'ETEXProfit',
     'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
     'ProfitPercent', 'investedAmount', 'profitAmount', 'RankValue'])
    print(df)
    df = df[df['gotEntry'] == 1].reset_index(drop=True)
    print('ALL Profit: ', df['profitAmount'].sum().round(2))
    print(df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index())
    return df

# Top 2
def top2(df):
   df_sorted = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode', 'ETEXProfit'], ascending=[True, False, False, False])
   df_sorted['RankValue'] = 50
   df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 60
   df_sorted.loc[df_sorted.groupby('Date').head(2).tail(1).index, 'RankValue'] = 40
   df = df_sorted.groupby('Date').head(2).reset_index(drop=True)
   df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
   df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
   df['dayName'] = pd.to_datetime(df['Date']).dt.day_name()
   df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Date', 'dayName', 'tickerName', 'TmPL', 'TmPredPL', 'gotEntry', 'gotLoss', 'counts', 'AccuracyScore:Mode', 'ETEXProfit',
    'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
    'ProfitPercent', 'investedAmount', 'profitAmount', 'RankValue'])
   print(df)
   df = df[df['gotEntry'] == 1].reset_index(drop=True)
   print('ALL Profit: ', df['profitAmount'].sum().round(2))
   print(df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index())
   return df

# Top 3
def top3(df):
   df_sorted = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode', 'ETEXProfit'], ascending=[True, False, False, False])
   df_sorted['RankValue'] = 0
   df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 50
   df_sorted.loc[df_sorted.groupby('Date').head(2).tail(1).index, 'RankValue'] = 30
   df_sorted.loc[df_sorted.groupby('Date').head(3).tail(1).index, 'RankValue'] = 20
   df = df_sorted.groupby('Date').head(3).reset_index(drop=True)
   df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
   df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
   df['dayName'] = pd.to_datetime(df['Date']).dt.day_name()
   df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Date', 'dayName', 'tickerName', 'TmPL', 'TmPredPL', 'gotEntry', 'gotLoss', 'counts', 'AccuracyScore:Mode', 'ETEXProfit',
    'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
    'ProfitPercent', 'investedAmount', 'profitAmount', 'RankValue'])
   print(df)
   df = df[df['gotEntry'] == 1].reset_index(drop=True)
   print('ALL Profit: ', df['profitAmount'].sum().round(2))
   print(df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index())
   return df

df1 = top1(df)
df2 = top2(df)
df3 = top3(df)
df[pd.to_datetime(df['Date']) == '2024-08-01']
# print(df1['dayName'].value_counts())
# print(df.tail(20))
# df.groupby('Date').head(3).reset_index(drop=True).head(30)
# df.groupby('Date').head(2).reset_index(drop=True)['TmPL'].value_counts()
# for d in [df1, df2, df3]:
#     print(d['TmPL'].value_counts()/len(d['TmPL']))

# query = 'select Date, tickerName, successCounts  from topAccurateStats'
# dfM = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))
# dfC = df[['Date', 'tickerName', 'counts']]
# dfC = dfC.loc[:, ~dfC.columns.duplicated()].reindex(columns=['Date', 'tickerName', 'counts'])
# dfC = dfM.merge(dfC[['Date', 'tickerName', 'counts']], how='left', on=['Date', 'tickerName'])
# dfC = dfC.dropna()
# dfC[dfC['successCounts'] != dfC['counts']]
# for d in dfC.groupby('Date'):
#     print(d)

# df.head(30)
# df1[['Date', 'tickerName', 'ProfitPercent', 'ETEXProfit', 'ActualOpen', 'PredOpen','AOpen/POpen-Diff']]

# def predEntryDatetime(row):
#     query = f'''
#     SELECT TOP 1 '{row['Date']}' as Date, Datetime as predEntryDatetime, '{row['tickerName']}' as tickerName
#     FROM [{row['tickerName']}]
#     WHERE CAST(Datetime AS DATE) = '{row['Datetime']}'
#     ORDER BY CASE WHEN {row['predTmEntry2']} BETWEEN Low AND High THEN 0 ELSE 1 END
#     '''
#     df = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster))
#     return df

# def mainEntryTime():
#     query = '''
#     select sp.Datetime as Date, sp.predDatetime as Datetime, tas.tickerName, sp.predTmEntry2  from topAccurateStats tas
#     left join simulationPrediction sp on sp.tickerName = tas.tickerName and cast(sp.predDatetime AS DATE) = cast(tas.Date AS DATE)
#     '''
#     df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))
#     df = df.sort_values(by='Datetime', ascending=False).reset_index(drop=True)
#     dfET = pd.concat([predEntryDatetime(row) for row in tqdm(df.to_dict('records'))]).reset_index(drop=True)
#     return dfET

# dfET = mainEntryTime()
# dfET['Date'] = pd.to_datetime(dfET['Date'])
# df1['Date'] = pd.to_datetime(df1['Date'])
# df1.merge(dfET, how='left', on=['Date', 'tickerName'])

# df['Date'] = pd.to_datetime(df['Date'])
# df = df.loc[:, ~df.columns.duplicated()]
# tt = df.merge(dfET, how='left', on=['Date', 'tickerName'])
# tt[tt['gotEntry'] ==1].tail(30)
# tt[tt['gotEntry'] ==1][['Date', 'tickerName', 'ProfitPercent', 'predEntryDatetime']].tail(30)


# dfET[(dfET['Date'] == '2024-08-16') & (dfET['tickerName'] == 'STAR')]
# dfET[(dfET['tickerName'] == 'GLOBUSSPR')]

# df.tail(30)[['Datetime', 'tickerName', 'TmPL', 'gotEntry', 'gotLoss', 'ProfitPercent', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'AOpen/POpen-Diff']]
