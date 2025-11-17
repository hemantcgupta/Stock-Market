# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 03:05:19 2024

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)
from Scripts.dbConnection import cnxn
from tqdm import tqdm

def fetch(filterDate):
    query = f'''
    DECLARE @Date DATE;
    SET @Date = '{filterDate}'; 
    WITH PreviousHighs AS (
        SELECT
            Datetime,
            tickerName,
            [Open],
            [High],
            [Low],
            [Close],
            LAG([High], 1) OVER (PARTITION BY tickerName ORDER BY Datetime) AS PrevDayHigh,
            LAG([High], 2) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev2DayHigh,
            LAG([Low], 1) OVER (PARTITION BY tickerName ORDER BY Datetime) AS PrevDayLow
        FROM mkanalyzer.dbo.mkDayFeature
        -- Filter for the last month based on the maximum datetime in the dataset
        WHERE Datetime >= DATEADD(MONTH, -1, @Date) and Datetime <= @Date
    ),
    FilteredResults AS (
        SELECT
            Datetime,
    		@Date as Date,
            tickerName,
            [Open],
            [High],
            [Low],
            [Close],
            CASE
                WHEN [High] > COALESCE(PrevDayHigh, 0) AND COALESCE(PrevDayHigh, 0) > COALESCE(Prev2DayHigh, 0) THEN 1
                ELSE 0
            END AS PrevDayBreak,
            CASE
                WHEN [Low] < COALESCE(PrevDayLow, 0) THEN 1
                ELSE 0
            END AS PrevDayLowBreak,
            ROW_NUMBER() OVER (PARTITION BY tickerName ORDER BY Datetime DESC) AS row_num -- Identify the latest record for each ticker
        FROM PreviousHighs
    )
    -- Count the occurrences of PrevDayBreak and get the latest break status
    SELECT
    	Date,
        tickerName,
        COUNT(CASE WHEN PrevDayBreak = 1 THEN 1 ELSE NULL END) AS PrevDayBreakCount,
        MAX(CASE WHEN row_num = 1 THEN PrevDayBreak ELSE NULL END) AS latestBreak, 
        MAX(CASE WHEN row_num = 1 THEN PrevDayLowBreak ELSE NULL END) AS latestLowBreak 
    FROM FilteredResults
    GROUP BY Date, tickerName
    ORDER BY latestBreak DESC, latestLowBreak, PrevDayBreakCount DESC
    '''
    df = pd.read_sql(query, cnxn('master'))
    return df

query = '''select  distinct(Datetime) as Date from mkanalyzer.dbo.mkDayFeature order by Datetime DESC'''
dateList = pd.read_sql(query, cnxn('master'))['Date'].astype(str).tolist()[:30]
df = pd.concat([fetch(filterDate) for filterDate in tqdm(dateList)]).reset_index(drop=True)

query = '''
select LAG(Date15, 1) OVER (PARTITION BY tickerName ORDER BY Date15) AS Date, tickerName, 
EntryPoint, TargetPoint, SLPoint, isBreckPrevEntry, Close315,
EntryTime, TargetTime, SLTime, gotEntry, gotProfit 
from mkprediction.dbo.algo7515
'''
dfF = pd.read_sql(query, cnxn('master'))
dfF = pd.merge(df, dfF, how='left', on=['Date', 'tickerName'])
dfF['Date'] = pd.to_datetime(dfF['Date'])
dfF['Exit'] = dfF.apply(lambda row: row['TargetPoint'] 
                        if pd.notna(row['EntryTime']) and pd.notna(row['TargetTime']) 
                        else (row['Close315'] if pd.notna(row['EntryTime']) else 0), axis=1)
dfF['PL'] = dfF.apply(lambda row: round((1-row['EntryPoint']/row['Exit'])*100, 2) if row['Exit'] != 0 else 0, axis=1)

dfF[dfF['Date'] == '2024-09-23'].head(20)

dfF.groupby('Date').head(20).reset_index(drop=True).to_excel('C:/Users/heman/Desktop/test.xlsx', index=False)






















