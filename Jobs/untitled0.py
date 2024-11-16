# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 02:20:06 2024

@author: Hemant
"""
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)

import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from tqdm import tqdm

# query = '''
# select * from mkTopPrediction
# where Datetime = (select max(Datetime) from mkTopPrediction)
# '''
# df = pd.read_sql(query, cnxn('mkanalyzer'))
# df['Date'] = df['Datetime'].astype(str)
# ValueIterator = df[['Date', 'tickerName']].to_dict('records')

df = pd.read_excel(r'C:\Users\heman\Desktop\Tomorrow Market Prediction.xlsx')
df['Date'] = df['Date'].astype(str)
ValueIterator = df[['Date', 'tickerName']].to_dict('records')

def dataFetch(dct):
    query = f'''
    with cte as(
    select Top 15 *,
    case when [Open] < [Close] then 1 else -1 end as PN,
    ROUND((1-([Open]/[Close]))*100, 2) as Body,
    case when [Open] < [Close] then ROUND((1-([Close]/[High]))*100, 2) else ROUND((1-([Open]/[High]))*100, 2)  end as UT,
    case when [Open] < [Close] then ROUND((1-([Open]/[Low]))*100, 2) else ROUND((1-([Close]/[Low]))*100, 2)  end as LT
    from [{dct.get('tickerName')}]
    where cast(Datetime as Date) = '{dct.get('Date')}'
    order by Datetime Desc
    )
    select cast(Datetime as Date) as Date,
    sum(PN) as PNS,
    sum(Body) as Body,
    sum(UT) as UT,
    sum(LT) as LT
    from cte
    group by cast(Datetime as Date)
    '''
    df = pd.read_sql(query, cnxn('mkintervalmaster'))
    df['tickerName'] = dct.get('tickerName')
    df['TotalPN'] = ((df['Body']+df['UT']+df['LT'])/3).round(2)
    return df

df75 = pd.concat([dataFetch(dct) for dct in tqdm(ValueIterator)]).reset_index(drop=True)
df75['Date'] = df75['Date'].astype(str)
df = df.merge(df75, how='left', on=['Date', 'tickerName'])
df.to_excel(r'C:\Users\heman\Desktop\test.xlsx', index=False)

# =============================================================================
# 
# =============================================================================

import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)

import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from tqdm import tqdm

def dataFetch(tickerName):
    query = f'''
    with cte as(
    SELECT  *,
      CASE 
        WHEN [Open] < [Close] THEN 1 ELSE -1 
      END AS PN,
      ROUND((1-([Open]/[Close]))*100, 2) AS Body,
      CASE 
        WHEN [Open] < [Close] THEN ROUND((1-([Close]/[High]))*100, 2)
        ELSE ROUND((1-([Open]/[High]))*100, 2)
      END AS UT,
      CASE 
        WHEN [Open] < [Close] THEN ROUND((1-([Open]/[Low]))*100, 2)
        ELSE ROUND((1-([Close]/[Low]))*100, 2)
      END AS LT
    FROM [{tickerName}]
    WHERE CONVERT(VARCHAR(8), Datetime, 108) BETWEEN '14:15:00' AND '15:25:00'
    )
    select cast(Datetime as Date) as Date,
    SUM(PN) as PNS,
    round(sum(Body),2) as Body,
    round(sum(UT),2) as UT,
    round(sum(LT),2) as LT,
    round((sum(Body) + sum(UT) + sum(LT))/3, 2) As Total
    from cte
    group by cast(Datetime as Date)
    order by round((sum(Body) + sum(UT) + sum(LT))/3, 2) DESC
    '''
    df = pd.read_sql(query, cnxn('mkintervalmaster'))
    df['tickerName'] = tickerName
    return df

query = '''
select distinct(tickerName) from mkDayFeature
'''
tickerList = pd.read_sql(query, cnxn('mkanalyzer'))['tickerName'].tolist()
df = pd.concat([dataFetch(tickerName) for tickerName in tqdm(tickerList)]).reset_index(drop=True)
df = df.sort_values(by=['Date', 'Total'], ascending=[False, True]).reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Date'])
# df.groupby('Date').head(1).reset_index(drop=True).head(10)

query = '''
select Datetime as Date, tickerName, [Open], [High],
[Low], [Close]
from mkDayFeature
'''
dfS = pd.read_sql(query, cnxn('mkanalyzer')).dropna().reset_index(drop=True)
dfM = df.merge(dfS, how='left', on=['Date', 'tickerName'])

query = '''
select Datetime as Date, tickerName, [Open] as PredOpen, [High] as PredHigh, 
[Low] as PredLow, [Close] as PredClose
from simulationPrediction
'''
dfS = pd.read_sql(query, cnxn('mkanalyzer')).dropna().reset_index(drop=True)
dfM = dfM.merge(dfS, how='left', on=['Date', 'tickerName'])
dfM['PL'] = ((1-dfM['Close']/dfM['PredClose'])*100).round(2)
dfM['Range'] = dfM.groupby('Date')['Total'].rank(ascending=False, method='first').astype(int)
dfM[dfM['Range'] == 78].head(22)['PL'].sum()
dfM.groupby('Range') ['PL'].sum().reset_index().sort_values(by='PL').tail(20)


dfM[dfM['Date'] >= '2024-09-14'].groupby('Range').agg(
    TotalMedian=('Total', lambda x: x.median()),  
    TotalMean=('Total', lambda x: x.mean()),  
    PL=('PL', 'sum')  ,
    Total=('Total', lambda x: sorted(x)),
    TotalMM=('Total', lambda x: [min(x), max(x)]),
    PNS=('PNS', lambda x: sorted(x)),
    PNSMM=('PNS', lambda x: [min(x), max(x)]),
).reset_index().sort_values(by='PL').head(20)


dfM.groupby('Range').agg(
    TotalMedian=('Total', lambda x: x.median()),  
    TotalMean=('Total', lambda x: x.mean()),  
    PL=('PL', 'sum')  ,
    Total=('Total', lambda x: sorted(x)),
    PNS=('PNS', lambda x: sorted(x))
).reset_index().sort_values(by='PL').tail(20)



# dfM.groupby('Date').head(1).reset_index(drop=True).head(22)
# dfM.groupby('Date').head(1).reset_index(drop=True).head(18)['PL'].sum()
# dfM.iloc[700:].head(10)
# sorted(list(dfM['tickerName'].unique()))[-100:]

dfM[dfM['tickerName'] == '^NSEBANK'].dropna().reset_index(drop=True)[:20]

bins = [-float('inf'), -1.5, -1, -0.5, 0, 0.5, 1, 1.5, float('inf')]
labels = ['< -1.5', '-1.5 to -1', '-1 to -0.5', '-0.5 to 0', '0 to 0.5', '0.5 to 1', '1 to 1.5', '> 1.5']
dfM['category'] = pd.cut(dfM['PL'], bins=bins, labels=labels)


dfM.groupby('category').agg(
    TotalMedian=('Total', lambda x: x.median()),  
    TotalMean=('Total', lambda x: x.mean()),  
    PL=('PL', 'sum')  ,
    Total=('Total', lambda x: sorted(x)),
    PNS=('PNS', lambda x: sorted(x))
).reset_index().sort_values(by='PL').tail(20)


dfM[dfM['Date'] >= '2024-09-14'].groupby('category').agg(
    TotalMedian=('Total', lambda x: x.median()),  
    TotalMean=('Total', lambda x: x.mean()),  
    PL=('PL', 'sum'),
    PLlist=('PL', lambda x: sorted({item for item in x})),
    Total=('Total', lambda x: sorted(x)),
    TotalMM=('Total', lambda x: [min(x), max(x)]),
    PNS=('PNS', lambda x: sorted(x)),
    PNSMM=('PNS', lambda x: [min(x), max(x)]),
).reset_index().sort_values(by='PL')

dfM[(dfM['Date'] >= '2024-09-14') & (dfM['tickerName'] == '^NSEBANK')].dropna().reset_index(drop=True).groupby('category').agg(
    TotalMedian=('Total', lambda x: x.median()),  
    TotalMean=('Total', lambda x: x.mean()),  
    PL=('PL', 'sum') ,
    PLlist=('PL', lambda x: sorted({item for item in x})),
    Total=('Total', lambda x: sorted(x)),
    TotalMM=('Total', lambda x: [min(x), max(x)] if not x.empty else []),
    PNS=('PNS', lambda x: sorted(x) if not x.empty else []),
    PNSMM=('PNS', lambda x: [min(x), max(x)] if not x.empty else []),
).reset_index().sort_values(by='PL')



