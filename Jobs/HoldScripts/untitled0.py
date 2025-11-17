# # -*- coding: utf-8 -*-
# """
# Created on Fri Aug 30 11:26:16 2024

# @author: Hemant
# """

# import pandas as pd
# pd.options.display.float_format = '{:.2f}'.format
# pd.set_option('display.max_columns', 15)
# pd.set_option('display.max_rows', 20)
# pd.set_option('expand_frame_repr', True)

# # Sample data
# data = {
#     'tickerName': ['MOL', 'CENTURYPLY', 'BARBEQUE', 'CONFIPET', 'TIMETECHNO', 'AJANTPHARM', 'UTIAMC', 'ASTERDM'],
#     'pMomentum': [60.22, 65, 75, 66.1, 69.42, 89, 38.71, 86.15],
#     'nMomentum': [39.78, 35, 25, 33.9, 30.58, 11, 61.29, 13.85],
#     'buySignal': [22, 0, 19, 16, 11, 8, 25, 0],
#     'sellSignal': [10, 21, 10, 26, 2, 14, 15, 18],
#     'holdingSignal': [43, 54, 46, 33, 62, 53, 35, 57]
# }

# df = pd.DataFrame(data)

# # Assign weights to each column
# weights = {
#     'pMomentum': 0.4,
#     'nMomentum': -0.3,
#     'buySignal': 0.2,
#     'sellSignal': -0.2,
#     'holdingSignal': 0.1
# }

# # Calculate priority score
# df['priorityScore'] = (
#     df['pMomentum'] * weights['pMomentum'] +
#     df['nMomentum'] * weights['nMomentum'] +
#     df['buySignal'] * weights['buySignal'] +
#     df['sellSignal'] * weights['sellSignal'] +
#     df['holdingSignal'] * weights['holdingSignal']
# )

# # Sort stocks by priority score
# df = df.sort_values(by='priorityScore', ascending=False)

# print(df[['tickerName', 'priorityScore']])


# =============================================================================
# 
# =============================================================================
import pandas as pd

# Sample data
data = {
    'tickerName': ['MOL', 'CENTURYPLY', 'BARBEQUE', 'CONFIPET', 'TIMETECHNO', 'AJANTPHARM', 'UTIAMC', 'ASTERDM'],
    'pMomentum': [60.22, 65, 75, 66.1, 69.42, 89, 38.71, 86.15],
    'nMomentum': [39.78, 35, 25, 33.9, 30.58, 11, 61.29, 13.85],
    'buySignal': [22, 0, 19, 16, 11, 8, 25, 0],
    'sellSignal': [10, 21, 10, 26, 2, 14, 15, 18],
    'holdingSignal': [43, 54, 46, 33, 62, 53, 35, 57],
    'successCount': [7, 7, 6, 6, 6, 5, 5, 5],
    'modelAccuracy': [75.32, 81.82, 82.67, 77.5, 77.33, 88, 77.46, 77.63],
    'epochLoss': [13.25, 15.54, 7.14, 11.52, 12, 8.35, 12.67, 14.31]
}

df = pd.DataFrame(data)

# Assign weights to each column
weights = {
    'pMomentum': 0.2,
    'nMomentum': -0.2,
    'buySignal': 0.2,
    'sellSignal': -0.2,
    'holdingSignal': 0.2,
    'successCount': 0.3,
    'modelAccuracy': 0.2,
    'epochLoss': -0.2
}

# Calculate priority score
df['priorityScore'] = (
    df['pMomentum'] * weights['pMomentum'] +
    df['nMomentum'] * weights['nMomentum'] +
    df['buySignal'] * weights['buySignal'] +
    df['sellSignal'] * weights['sellSignal'] +
    df['holdingSignal'] * weights['holdingSignal'] +
    df['successCount'] * weights['successCount'] +
    df['modelAccuracy'] * weights['modelAccuracy'] +
    df['epochLoss'] * weights['epochLoss']
)

# Sort stocks by priority score
df = df.sort_values(by='priorityScore', ascending=False)

print(df[['tickerName', 'priorityScore']])
