# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:18:48 2024

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)
from Scripts.dbConnection import cnxn, Data_Inserting_Into_DB


class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkintervalmaster = 'mkintervalmaster'


query1 = f'''
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
	tas1.pMomentum,
	tas1.nMomentum,
	tas1.buySignal,
	tas1.sellSignal,
	tas1.holdingSignal,
	sp.EtEx2Profit as PredProfit,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL, 
    tas1.TmPredPL,
        CASE 
        WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
        THEN 1 
        ELSE 0 
    END AS gotEntry,
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
    END AS ActualProfit,
    round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
    Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
    [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose,
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, epochLoss ASC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 --AND gotEntry = 1
ORDER BY Date DESC;
'''
query2 = f'''
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
	tas1.pMomentum,
	tas1.nMomentum,
	tas1.buySignal,
	tas1.sellSignal,
	tas1.holdingSignal,
	sp.EtEx2Profit as PredProfit,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL, 
    tas1.TmPredPL,
        CASE 
        WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
        THEN 1 
        ELSE 0 
    END AS gotEntry,
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
    END AS ActualProfit,
    round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
    Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
    [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose,
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, epochLoss ASC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
WHERE tas1.TmPredPL = 1
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 --AND gotEntry = 1
ORDER BY Date DESC;
'''
query3 = f'''
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
	tas1.pMomentum,
	tas1.nMomentum,
	tas1.buySignal,
	tas1.sellSignal,
	tas1.holdingSignal,
	sp.EtEx2Profit as PredProfit,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL, 
    tas1.TmPredPL,
        CASE 
        WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
        THEN 1 
        ELSE 0 
    END AS gotEntry,
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
    END AS ActualProfit,
    round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
    Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
    [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose,
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, modelAccuracy DESC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 --AND gotEntry = 1
ORDER BY Date DESC;
'''
query4 = f'''
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
	tas1.pMomentum,
	tas1.nMomentum,
	tas1.buySignal,
	tas1.sellSignal,
	tas1.holdingSignal,
	sp.EtEx2Profit as PredProfit,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL, 
    tas1.TmPredPL,
        CASE 
        WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
        THEN 1 
        ELSE 0 
    END AS gotEntry,
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
    END AS ActualProfit,
    round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
    Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
    [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose,
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, modelAccuracy DESC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
WHERE tas1.TmPredPL = 1
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 --AND gotEntry = 1
ORDER BY Date DESC;
'''




# Read the SQL queries into dataframes
df1 = pd.read_sql(query1, cnxn(VAR.db_mkanalyzer))
df2 = pd.read_sql(query2, cnxn(VAR.db_mkanalyzer))
df3 = pd.read_sql(query3, cnxn(VAR.db_mkanalyzer))
df4 = pd.read_sql(query4, cnxn(VAR.db_mkanalyzer))

# Extract the dates from each dataframe
dates_df1 = set(df1['Date'])
dates_df2 = set(df2['Date'])
dates_df3 = set(df3['Date'])
dates_df4 = set(df4['Date'])

# Remove rows from df2 where the date is in df1
df2_filtered = df2[~df2['Date'].isin(dates_df1)]

# Remove rows from df3 where the date is in df1 or df2_filtered
dates_df2_filtered = set(df2_filtered['Date'])
df3_filtered = df3[~df3['Date'].isin(dates_df1.union(dates_df2_filtered))]

# Remove rows from df4 where the date is in df1, df2_filtered, or df3_filtered
dates_df3_filtered = set(df3_filtered['Date'])
df4_filtered = df4[~df4['Date'].isin(dates_df1.union(dates_df2_filtered).union(dates_df3_filtered))]

# Concatenate all dataframes
final_df = pd.concat([df1, df2_filtered, df3_filtered, df4_filtered], ignore_index=True)

# Optionally, remove duplicates if needed
final_df = final_df.drop_duplicates()

final_df = final_df.sort_values(by='Date', ascending=False)

# Optionally, reset the index after sorting
final_df.reset_index(drop=True, inplace=True)

# Display or save the final dataframe
print(final_df)

final_df[final_df['gotEntry'] == 1].groupby(pd.to_datetime(final_df['Date']).dt.month) ['ActualProfit'].sum()
df1[df1['gotEntry'] == 1].groupby(pd.to_datetime(df1['Date']).dt.month) ['ActualProfit'].sum()
df2[df2['gotEntry'] == 1].groupby(pd.to_datetime(df2['Date']).dt.month) ['ActualProfit'].sum()
df3[df3['gotEntry'] == 1].groupby(pd.to_datetime(df3['Date']).dt.month) ['ActualProfit'].sum()
df4[df4['gotEntry'] == 1].groupby(pd.to_datetime(df4['Date']).dt.month) ['ActualProfit'].sum()


df1.head(20)
