# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:10:18 2024

@author: Hemant
"""


import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from Scripts.TmInsights import Uplode_Latest_Insights 
from Scripts.TmPrediction import StockAnalyzer, StockMarketPhasesAnalyzer, StockPredictorAnalyzer, TopPredictor

# =============================================================================
# Variables
# =============================================================================
class VAR:
    cpu_count = int(cpu_count() * 0.8)
    db_name_master = 'master'
    db_name_analyzer = 'mkanalyzer'
    db_name_ianalyzer = 'mkintervalanalyzer'
    

# =============================================================================
# Interval Analyzer 
# =============================================================================
def JobTomorrowAnalyzerMain():
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
    stockSymbols = pd.read_sql(query, cnxn(VAR.db_name_ianalyzer))['TABLE_NAME'].tolist()
    df = Tomorrow_Data_Process(stockSymbols)
    return df


# =============================================================================
# Multi Data Process for each Ticker Name
# =============================================================================
def Tomorrow_Data_Process(stockSymbols):
    with Pool(processes=VAR.cpu_count) as pool:
        resultSR = list(tqdm(pool.imap(fetch_db_data, stockSymbols), total=len(stockSymbols), desc='Fetch mkintervalanalyzer Data'))    
    df = fetch_db_all_feature(resultSR)
    return df

# =============================================================================
# Fetch mkintervalanalyzer Data
# =============================================================================
def fetch_db_data(tickerName):
    query = f'''SELECT Support, Resistance FROM dbo.[{tickerName}]'''
    df = pd.read_sql(query, cnxn(VAR.db_name_ianalyzer)).dropna(how='all').reset_index(drop=True)
    resultSR = {'tickerName': tickerName, 'Support': sorted(df['Support'].dropna().tolist()), 'Resistance': sorted(df['Resistance'].dropna().tolist())}
    return resultSR

def fetch_db_all_feature(resultSR):
    query = '''
    with main as (
     	select tickerName, max(Datetime) as Datetime from mkDayFeature
     	group by tickerName
     	)
    select m.[tickerName], m.[Datetime], f.[Open], f.[High], f.[Low], f.[Close], 
    f.[PvClose], f.[OC-P/L], f.[PvCC-P/L], f.[maxHigh], f.[maxLow], f.[closeTolerance], f.[priceBand],
    pb.[BuyInProfit MP::HP::MP::HP], pb.[SellInLoss MP::MP::LP::LP], pb.[BuyInLoss MP::HP::LP::HP], pb.[SellInProfit MP::HP::LP::LP],
    pb.[ProbabilityOfProfitMT2Percent], pb.[ProbabilityOfLoss1ratio3Percent], pb.[ProbabilityOfProfitTomorrow], pb.[ProbabilityOfLossTomorrow], 
    pb.[ProbabilityOfProfitLoss], pb.[ProbabilityOfmaxHigh], pb.[ProbabilityOfmaxLow], pb.[ProbabilityOfpriceBand], pb.[ProbabilityOfCloseTolerance],
    pd.[predTmOpen], pd.[predTmEntry1], pd.[predTmExit1], pd.[predTmEntry2], 
    pd.[predTmExit2], pd.[predTmClose], pd.[predTmMaxhigh], pd.[predTmMaxlow],
    pd.[EtEx1Profit], pd.[EtEx2Profit], pd.[predTmP/L]
    from main m
    left join mkDayFeature f on m.tickerName=f.tickerName and m.Datetime=f.Datetime
    left join mkDayProbability pb on m.tickerName=pb.tickerName and m.Datetime=pb.Datetime
    left join mkDayPrediction pd on m.tickerName=pd.tickerName and m.Datetime=pd.Datetime
    order by pb.[ProbabilityOfProfitMT2Percent] desc, pb.[ProbabilityOfProfitTomorrow] desc
    '''
    df = pd.read_sql(query, cnxn(VAR.db_name_analyzer))
    dfSR = pd.DataFrame(resultSR)
    df = pd.merge(df, dfSR, how='left', on='tickerName')
    df[['SupportAbove', 'SupportBelow', 'ResistanceAbove', 'ResistanceBelow']] = df.apply(get_nearest_values, axis=1)
    # df.drop(columns=['Support', 'Resistance'], inplace=True)
    return df

def get_nearest_values(row):
    SupportAbove = sorted([item for item in row['Support'] if row['Close'] <= item])[:10]
    SupportBelow = sorted([item for item in row['Support'] if row['Close'] > item])[-10:]
    ResistanceAbove = sorted([item for item in row['Resistance'] if row['Close'] <= item])[:10]
    ResistanceBelow = sorted([item for item in row['Resistance'] if row['Close'] > item])[-10:]
    return pd.Series({
        'SupportAbove': SupportAbove,
        'SupportBelow': SupportBelow,
        'ResistanceAbove': ResistanceAbove,
        'ResistanceBelow': ResistanceBelow
    })  
  
# =============================================================================
# Job Function
# =============================================================================
def JobTomorrowAnalyzer():
    # dfP1 = StockAnalyzer().analyze()
    # dfP1 = StockMarketPhasesAnalyzer.getMarketPhases(dfP1)
    # dfP2 = StockPredictorAnalyzer().analyze()
    dfP3 = TopPredictor().analyze()
    df = JobTomorrowAnalyzerMain()
    filename = './Sample Data/Tomorrow Market Prediction.xlsx'
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # # Write dfP1 to Excel and create a table
        # dfP1.to_excel(writer, sheet_name='TOP 20 Old Logic', index=False)
        # workbook = writer.book
        # worksheet = writer.sheets['TOP 20 Old Logic']
        # worksheet.add_table(0, 0, len(dfP1), len(dfP1.columns) - 1, {'columns': [{'header': col} for col in dfP1.columns]})
    
        # # Write dfP2 to Excel and create a table
        # dfP2.to_excel(writer, sheet_name='TOP 20 New Logic', index=False)
        # worksheet = writer.sheets['TOP 20 New Logic']
        # worksheet.add_table(0, 0, len(dfP2), len(dfP2.columns) - 1, {'columns': [{'header': col} for col in dfP2.columns]})
        
        # Write dfP3 to Excel and create a table
        dfP3.to_excel(writer, sheet_name='TOP 20', index=False)
        worksheet = writer.sheets['TOP 20']
        worksheet.add_table(0, 0, len(dfP3), len(dfP3.columns) - 1, {'columns': [{'header': col} for col in dfP3.columns]})
    
        # Write df to Excel and create a table
        df.to_excel(writer, sheet_name='Market Analysis', index=False)
        worksheet = writer.sheets['Market Analysis']
        worksheet.add_table(0, 0, len(df), len(df.columns) - 1, {'columns': [{'header': col} for col in df.columns]})
    Uplode_Latest_Insights(filename)
    return df

if __name__ == "__main__":
    df = JobTomorrowAnalyzer()

