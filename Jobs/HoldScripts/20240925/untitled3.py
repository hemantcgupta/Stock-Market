# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:53:39 2024

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

import numpy as np

Ticker = 'SBILIFE'
def get_closing_point(close):
    return round(close * (1 + 0.005) / 0.05) * 0.05

def get_enrty(high, closing_point):
    if high > closing_point:
        return round(high * (1 + 0.007) / 0.05) * 0.05
    else:
        return round(high * (1 + 0.012) / 0.05) * 0.05
    
def get_target(high, closing_point, entry):
    if high > closing_point:
        return round(entry * (1 + 0.0125) / 0.05) * 0.05
    else:
        return round(entry * (1 + 0.0125) / 0.05) * 0.05
        
def get_stoploss(high, closing_point, entry, low):
    if high > closing_point:
        return round(max(entry * (1 - 0.005), low * (1 - 0.007)) / 0.05) * 0.05
    else:
        return round(max(entry * (1 - 0.007), low * (1 - 0.012)) / 0.05) * 0.05
    
    
query75 = f'''
WITH TimeFilteredData AS (
    SELECT 
        [Datetime],
        CAST([Datetime] AS DATE) AS TradeDate,
        [High], 
        [Low], 
        [Open], 
        [Close]
    FROM 
        {Ticker}
    WHERE 
        (DATEPART(HOUR, [Datetime]) = 14 AND DATEPART(MINUTE, [Datetime]) BETWEEN 15 AND 59)
        OR 
        (DATEPART(HOUR, [Datetime]) = 15 AND DATEPART(MINUTE, [Datetime]) BETWEEN 0 AND 25)
),
AggregatedData AS (
    SELECT
        TradeDate,
        MAX(High) AS MaxHigh,
        MIN(Low) AS MinLow,
        MIN([Datetime]) AS MinDatetime,
        MAX([Datetime]) AS MaxDatetime
    FROM 
        TimeFilteredData
    GROUP BY 
        TradeDate
),
OpenCloseValues AS (
    SELECT
        tfd.TradeDate,
        ad.MaxHigh,
        ad.MinLow,
        MIN(CASE WHEN tfd.[Datetime] = ad.MinDatetime THEN tfd.[Open] END) AS OpenValue,
        MIN(CASE WHEN tfd.[Datetime] = ad.MaxDatetime THEN tfd.[Close] END) AS CloseValue
    FROM 
        TimeFilteredData tfd
    JOIN 
        AggregatedData ad 
    ON 
        tfd.TradeDate = ad.TradeDate
    GROUP BY 
        tfd.TradeDate, ad.MaxHigh, ad.MinLow, ad.MinDatetime, ad.MaxDatetime
)
SELECT
    TradeDate AS Date,
    OpenValue AS Open75,
    MaxHigh as High75,
    MinLow as Low75,
    CloseValue AS Close75
FROM 
    OpenCloseValues
ORDER BY 
    TradeDate DESC;
'''    

query15 = f'''
WITH TimeFilteredData AS (
    SELECT 
        [Datetime],
        CAST([Datetime] AS DATE) AS TradeDate,
        [High], 
        [Low], 
        [Open], 
        [Close]
    FROM 
        {Ticker}
    WHERE 
        (DATEPART(HOUR, [Datetime]) = 9 AND DATEPART(MINUTE, [Datetime]) BETWEEN 15 AND 25)
),
AggregatedData AS (
    SELECT
        TradeDate,
        MAX(High) AS MaxHigh,
        MIN(Low) AS MinLow,
        MIN([Datetime]) AS MinDatetime,
        MAX([Datetime]) AS MaxDatetime
    FROM 
        TimeFilteredData
    GROUP BY 
        TradeDate
),
OpenCloseValues AS (
    SELECT
        tfd.TradeDate,
        ad.MaxHigh,
        ad.MinLow,
        MIN(CASE WHEN tfd.[Datetime] = ad.MinDatetime THEN tfd.[Open] END) AS OpenValue,
        MIN(CASE WHEN tfd.[Datetime] = ad.MaxDatetime THEN tfd.[Close] END) AS CloseValue
    FROM 
        TimeFilteredData tfd
    JOIN 
        AggregatedData ad 
    ON 
        tfd.TradeDate = ad.TradeDate
    GROUP BY 
        tfd.TradeDate, ad.MaxHigh, ad.MinLow, ad.MinDatetime, ad.MaxDatetime
)
SELECT
    TradeDate AS Date,
    OpenValue AS Open15,
    MaxHigh as High15,
    MinLow as Low15,
    CloseValue AS Close15
FROM 
    OpenCloseValues
ORDER BY 
    TradeDate DESC;
'''
# =============================================================================
# 
# =============================================================================
class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkintervalmaster = 'mkintervalmaster'
    
# =============================================================================
#     
# =============================================================================
df15 = pd.read_sql(query15, cnxn(VAR.db_mkintervalmaster)) 
df75 = pd.read_sql(query75, cnxn(VAR.db_mkintervalmaster)) 
df75['Date75'] = df75['Date']
df75['Date'] = df75['Date'].shift(1)
df = pd.merge(df15, df75, on='Date', how='left', suffixes=('_15', '_75')).dropna().rename(columns={'Date': 'Date15'})
df = df[['Date15', 'Date75', 'Open15', 'High15', 'Low15', 'Close15', 'Open75', 'High75', 'Low75', 'Close75']]

# =============================================================================
# 
# =============================================================================
def calculate_points(row):
    isBreckPrevEntry = 0
    close_point = round(row['Close75'] * (1 + 0.005) / 0.05) * 0.05
    entry = round(row['High75'] * (1 + (0.007 if row['High75'] > close_point else 0.012)) / 0.05) * 0.05
    if entry < row['High15']:  
        isBreckPrevEntry = 1
        close_point = round(row['Close15'] * (1 + 0.005) / 0.05) * 0.05
        entry = round(row['High15'] * (1 + (0.007 if row['High15'] > close_point else 0.012)) / 0.05) * 0.05
        target = round(entry * (1 + 0.0125) / 0.05) * 0.05
        stoploss = round(max(entry * (1 - (0.005 if row['High15'] > close_point else 0.007)),
                              row['Low15'] * (1 - (0.007 if row['High15'] > close_point else 0.012))) / 0.05) * 0.05
    else:
        target = round(entry * (1 + 0.0125) / 0.05) * 0.05
        stoploss = round(max(entry * (1 - (0.005 if row['High75'] > close_point else 0.007)),
                              row['Low75'] * (1 - (0.007 if row['High75'] > close_point else 0.012))) / 0.05) * 0.05
    return pd.Series([close_point, entry, target, stoploss, isBreckPrevEntry], index=['ClosePoint', 'EntryPoint', 'TargetPoint', 'SLPoint', 'isBreckPrevEntry'])
df[['ClosePoint', 'EntryPoint', 'TargetPoint', 'SLPoint', 'isBreckPrevEntry']] = df.apply(calculate_points, axis=1)
df['isBreckPrevEntry'] = df['isBreckPrevEntry'].astype(int)

# =============================================================================
# 
# =============================================================================
query = f'''
select cast(Datetime as Date) as Date, Datetime, 
[Open], [High], [Low], [Close]
from {Ticker}
order by Datetime
'''
dfI = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster))
dfI = dfI[(dfI['Datetime'].dt.time >= pd.to_datetime('09:30:00').time()) & (dfI['Datetime'].dt.time <= pd.to_datetime('15:15:00').time())]
dfI = dfI.sort_values(by='Datetime').reset_index(drop=True)


# def get_ExecutionTime(dfE, row):
#     dfE = dfE[dfE['Date'] == row['Date15']]
#     dfE.loc[(dfE['Low'] <= row['EntryPoint']) & (dfE['High'] >= row['EntryPoint']), 'EntryTime'] = dfE['Datetime']
#     dfE.loc[(dfE['Low'] <= row['TargetPoint']) & (dfE['High'] >= row['TargetPoint']), 'TargetTime'] = dfE['Datetime']
#     dfE.loc[(dfE['Low'] <= row['SLPoint']) & (dfE['High'] >= row['SLPoint']), 'SLTime'] = dfE['Datetime']
#     EntryTime = dfE['EntryTime'].dropna().iloc[0] if not dfE['EntryTime'].dropna().empty else None
#     TargetTime = dfE['TargetTime'].dropna().iloc[0] if not dfE['TargetTime'].dropna().empty else None
#     SLTime = dfE['SLTime'].dropna().iloc[0] if not dfE['SLTime'].dropna().empty else None
#     return pd.Series([EntryTime, TargetTime, SLTime], index=['EntryTime', 'TargetTime', 'SLTime'])

def get_ExecutionTime(dfE, row):
    # Filter rows by date first
    dfE = dfE[dfE['Date'] == row['Date15']].reset_index(drop=True)
    
    # Initialize variables to store the result
    EntryTime, TargetTime, SLTime = None, None, None
    
    Close315 = dfE['Open'].iloc[-1] if not dfE['Open'].empty else None
    # Iterate over the DataFrame using a loop
    for idx, df_row in dfE.iterrows():
        # Check for EntryPoint
        if EntryTime is None and df_row['Low'] <= row['EntryPoint'] <= df_row['High']:
            EntryTime = df_row['Datetime']
        
        # Once EntryTime is found, check for TargetPoint and SLPoint
        if EntryTime:
            # Check for TargetPoint
            if TargetTime is None and df_row['Low'] <= row['TargetPoint'] <= df_row['High']:
                TargetTime = df_row['Datetime']
            # Check for SLPoint
            if SLTime is None and df_row['Low'] <= row['SLPoint'] <= df_row['High']:
                SLTime = df_row['Datetime']
            
            # Break the loop once both TargetTime and SLTime are found
            if TargetTime and SLTime:
                break
    
    return pd.Series([EntryTime, TargetTime, SLTime, Close315], index=['EntryTime', 'TargetTime', 'SLTime', 'Close315'])

df[['EntryTime', 'TargetTime', 'SLTime', 'Close315']] = df.apply(lambda row: get_ExecutionTime(dfI, row), axis=1)

df[pd.to_datetime(df['EntryTime']) < pd.to_datetime(df['SLTime'])]
df.isnull().sum()
df[df['TargetTime'].notna()]
dfT = df[df['EntryTime'].notna()]
# dfT['ConditionMet'] = ((dfT['EntryTime'] < dfT['TargetTime']) & (dfT['TargetTime'] < dfT['SLTime'])).astype(int)
dfT['ExecutionType'] = np.where(dfT['TargetTime'] < dfT['SLTime'], 'Target', 'SL')

dfT = dfT.reset_index(drop=True)

dfT['PL'] = (dfT['Close315'] - dfT['EntryPoint'])/dfT['Close315']*100
dfT.iloc[0:20]
# =============================================================================
# 
# =============================================================================
# df['ClosePoint'] = df['Close75'].apply(get_closing_point)  
# df['EntryPoint'] = df.apply(lambda row: get_enrty(row['High75'], row['ClosePoint']), axis=1)  
# df['TargetPoint'] = df.apply(lambda row: get_target(row['High75'], row['ClosePoint'], row['EntryPoint']), axis=1)  
# df['SLPoint'] = df.apply(lambda row: get_stoploss(row['High75'], row['ClosePoint'], row['EntryPoint'], row['Low75']), axis=1)  


# df['ClosePoint'] = df['CloseValue'].apply(get_closing_point)  
# df['EntryPoint'] = df.apply(lambda row: get_enrty(row['HighValue'], row['ClosePoint']), axis=1)  
# df['TargetPoint'] = df.apply(lambda row: get_target(row['HighValue'], row['ClosePoint'], row['EntryPoint']), axis=1)  
# df['SLPoint'] = df.apply(lambda row: get_stoploss(row['HighValue'], row['ClosePoint'], row['EntryPoint'], row['LowValue']), axis=1)  


# =============================================================================
# 
# =============================================================================
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
            LAG([High], 3) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev3DayHigh
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
                WHEN COALESCE(PrevDayHigh, 0) > COALESCE(Prev2DayHigh, 0) AND COALESCE(Prev2DayHigh, 0) > COALESCE(Prev3DayHigh, 0) THEN 1
                ELSE 0
            END AS PrevDayBreak,
            ROW_NUMBER() OVER (PARTITION BY tickerName ORDER BY Datetime DESC) AS row_num -- Identify the latest record for each ticker
        FROM PreviousHighs
    )
    -- Count the occurrences of PrevDayBreak and get the latest break status
    SELECT
    	Date,
        tickerName,
        COUNT(CASE WHEN PrevDayBreak = 1 THEN 1 ELSE NULL END) AS PrevDayBreakCount,
        MAX(CASE WHEN row_num = 1 THEN PrevDayBreak ELSE NULL END) AS latestBreak -- Get the break status for the latest record
    FROM FilteredResults
    GROUP BY Date, tickerName
    ORDER BY latestBreak DESC, PrevDayBreakCount DESC
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

dfF[dfF['Date'] == '2024-09-19'].head(20)

dfF.groupby('Date').head(1).reset_index(drop=True).head(20)






















