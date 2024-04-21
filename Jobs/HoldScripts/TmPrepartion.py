# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:04:46 2024

@author: Hemant
"""

import os
import glob
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 20)
pd.set_option('expand_frame_repr', True)

# =============================================================================
# Start
# =============================================================================
directory_path = "./Data/Probability/"
excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
excel_files.sort(key=os.path.getmtime, reverse=False)
df = pd.read_excel(excel_files[-1])


df.sort_values(by='ProbabilityOfmaxHigh', ascending=False).head(20)
df[df['Symbol'] == 'BLS.NS']['ProbabilityOfCloseTolerance'][439]































# =============================================================================
# 
# =============================================================================
excel_files = excel_files[7:]
def compare_predicate_data(excel_files):
    realList, predList = [], []
    for i, file in enumerate(excel_files):
        df = pd.read_excel(file)
        try:
            df['predDate'] = excel_files[i+1].split('\\')[-1].split('.')[0]
        except: 
            continue
        dfR = df[['Symbol', 'Date', 'Open','High', 'Low', 'Close', 'P/L']]
        dfR['predDate'] = dfR['Date'].dt.date.astype(str)
        realList.append(dfR)
        predList.append(df[['Symbol', 'predDate', 'predTmOpen', 'predTmEntry1', 'predTmExit1', 'predTmEntry2', 'predTmExit2', 'predTmClose', 'predTmMaxhigh', 'predTmMaxlow', 'EtEx1Profit', 'EtEx2Profit', 'predTmP/L']])
    df = pd.merge(pd.concat(realList).reset_index(drop=True), pd.concat(predList).reset_index(drop=True), how='right', on=['Symbol', 'predDate'])
    return df.dropna().reset_index(drop=True)

df = compare_predicate_data(excel_files)
df.apply(lambda row: True if row['Open'] <= row['predOpen'] else False)
df.apply(lambda row: True if row['Open'] <= row['predOpen'] else False)























