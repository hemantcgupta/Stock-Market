# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:11:03 2024

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')
import os
from scipy import stats
import numpy as np
from tqdm import tqdm
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from Scripts.dbConnection import cnxn, Data_Inserting_Into_DB
from multiprocessing import Pool, cpu_count

class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkintervalmaster = 'mkintervalmaster'
    
    
query = f'''
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracyPL,
	tas1.epochLossPL,
	tas1.modelAccuracygotLoss,
	tas1.epochLossgotLoss,
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
	tas1.TmPredgotLoss,
	tas1.TmPredPL5Summary, tas1.TmPredgotLoss5Summary,
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
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, epochLossPL ASC) as rn
FROM mkTopPrediction AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
--WHERE TmPredPL = 1 and TmPredgotLoss = 0
)
SELECT Date, tickerName, TmPL, TmPredPL, TmPredgotLoss, gotEntry, gotLoss, rn, PredEntry, PredExit, ActualProfit, PredProfit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], 
successCount, modelAccuracyPL, epochLossPL, modelAccuracygotLoss, epochLossgotLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal, TmPredPL5Summary, TmPredgotLoss5Summary
FROM cte 
--WHERE rn = 1 AND TmPredPL = 1 AND gotEntry = 1
ORDER BY Date DESC;
'''
df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))   
df['TmPredPL1'] = df['TmPredPL5Summary'].str.split(', ').apply(lambda x: stats.mode([item.split(' :: ')[0] for item in x]).mode[0])
df['TmPredgotLoss1'] = df['TmPredgotLoss5Summary'].str.split(', ').apply(lambda x: stats.mode([item.split(' :: ')[0] for item in x]).mode[0])
# df = df[(df['TmPredPL1'] == '1') & (df['TmPredgotLoss1'] == '0')]
# df = df.sort_values(by=['Date', 'successCount', 'epochLossPL'], ascending=[False, False, True])
# df = df.groupby('Date').head(1).reset_index(drop=True)
# df = df[df['gotEntry'] == 1]
# df.groupby(pd.to_datetime(df['Date']).dt.month) ['ActualProfit'].sum()


# df.iloc[14]['TmPredPL5Summary']
# df.iloc[14]['TmPredgotLoss5Summary']

from sklearn.preprocessing import MinMaxScaler
# Define columns to normalize
cols_to_normalize = ['successCount', 'modelAccuracyPL', 'epochLossPL', 'modelAccuracygotLoss', 
                      'epochLossgotLoss', 'pMomentum', 'nMomentum', 'buySignal', 'sellSignal', 'holdingSignal']

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize only the selected columns
df_normalized = df.copy()
df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Calculate composite score with weights
weights = {
    'successCount': 0.1,
    'modelAccuracyPL': 0.2,
    'epochLossPL': 0.1,
    'modelAccuracygotLoss': 0.15,
    'epochLossgotLoss': 0.15,
    'pMomentum': 0.1,
    'nMomentum': 0.1,
    'buySignal': 0.05,
    'sellSignal': 0.05,
    'holdingSignal': 0.05
}

# Calculate the composite score
df_normalized['score'] = sum(df_normalized[col] * weight for col, weight in weights.items())

# Rank stocks within each Date group
df_normalized['rank'] = df_normalized.groupby('Date')['score'].rank(ascending=False)

# Sort by Date and Rank
df_sorted = df_normalized.sort_values(by=['Date', 'rank'])

# Display the sorted DataFrame
print(df_sorted)
df_sorted = df_sorted[(df_sorted['TmPredPL1'] == '1') & (df_sorted['TmPredgotLoss1'] == '0')]
df_sorted = df_sorted.sort_values(by=['Date', 'rank'], ascending=[False, True])
df_sorted = df_sorted.groupby('Date').head(1).reset_index(drop=True)
df_sorted = df_sorted[df_sorted['gotEntry'] == 1]
df_sorted.groupby(pd.to_datetime(df_sorted['Date']).dt.month) ['ActualProfit'].sum()
