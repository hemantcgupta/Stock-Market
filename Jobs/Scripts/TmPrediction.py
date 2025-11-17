# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:58:08 2024

@author: Hemant
"""
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import yfinance as yf  
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.set_option('expand_frame_repr', True)
from Scripts.dbConnection import cnxn,  Data_Inserting_Into_DB

class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkintervalmaster = 'mkintervalmaster'

class StockAnalyzer:
    def __init__(self):
        self.cnxn = cnxn(VAR.db_mkanalyzer)

    def fetchData(self, inputDate):
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
        WHERE tas.Date = '{inputDate}' --AND tas.TmPredPL = 1
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
        df = pd.read_sql(query, self.cnxn)
        return df   

    def getDateList(self):
        query = '''
        SELECT DISTINCT(Datetime) FROM simulationPrediction
        WHERE Datetime >= DATEADD(MONTH, -3, (SELECT MAX(Datetime) FROM simulationPrediction))
        ORDER BY Datetime ASC
        '''
        dateList = pd.read_sql(query, self.cnxn)['Datetime'].dt.date.astype(str).tolist()
        return dateList

    def analyze(self):
        dateList = self.getDateList()
        df = pd.concat([self.fetchData(inputDate) for inputDate in tqdm(dateList, desc='Get Market Data')]).fillna(0)
        df = df.sort_values(by=['Date', 'counts', 'AccuracyScore:Mode', 'ETEXProfit'], ascending=[False, False, False, False])
        df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Date', 'tickerName', 'TmPL', 'TmPredPL', 'gotEntry', 'gotLoss', 'counts', 'AccuracyScore:Mode', 'ETEXProfit',
         'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
         'ProfitPercent'])
        return df
  
    def top1(self, df):
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
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary

    def top2(self, df):
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
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary

    def top3(self, df):
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
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary

class MarketAnalyzer:
    def __init__(self, kwargs):
        self.cnxn = cnxn(VAR.db_mkintervalmaster)
        self.Date = kwargs.get('Date')
        self.symbol = kwargs.get('tickerName')
        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')
        self.market_data = None

    def get_market_data(self):
        query = f"select * from [{self.symbol}] where Datetime >= '{self.start_date}' and Datetime <= '{self.end_date}' "
        self.market_data= pd.read_sql(query, self.cnxn)
        return self.market_data

    @staticmethod
    def calculate_rsi(data, window):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def identify_market_phases(self):
        self.market_data['SMA_20'] = self.market_data['Close'].rolling(window=20).mean()
        self.market_data['SMA_50'] = self.market_data['Close'].rolling(window=50).mean()
        self.market_data['RSI'] = self.calculate_rsi(self.market_data['Close'], 14)
        self.market_data['Accumulation'] = np.where((self.market_data['Close'] > self.market_data['SMA_20']) & 
                                                    (self.market_data['Close'] > self.market_data['SMA_50']), 1, 0)
        self.market_data['Advancing'] = np.where((self.market_data['SMA_20'] > self.market_data['SMA_50']) & 
                                                 (self.market_data['Close'] > self.market_data['SMA_20']), 1, 0)
        self.market_data['Distribution'] = np.where((self.market_data['Close'] < self.market_data['SMA_20']) & 
                                                    (self.market_data['Close'] < self.market_data['SMA_50']), 1, 0)
        self.market_data['Declining'] = np.where((self.market_data['SMA_20'] > self.market_data['SMA_50']) & 
                                                 (self.market_data['Close'] < self.market_data['SMA_20']), 1, 0)
        return self.market_data

    def analyze(self):
        self.get_market_data()
        market_data_with_phases = self.identify_market_phases()
        df = market_data_with_phases.tail(75)
        return {'Date': self.Date, 'tickerName': self.symbol, '315Close': round(df.iloc[-3]['Open'], 2),
                **{col: df[col].sum() for col in ['Accumulation', 'Advancing', 'Distribution', 'Declining']}}

class StockMarketPhasesAnalyzer:
    def __init__(self):
        self.cnxn = cnxn(VAR.db_mkanalyzer)

    def fetchData(self, inputDate):
        query = f'''
        DECLARE @inputDate Date;
        SET @inputDate = '{inputDate}';
        WITH cte1 AS (
            SELECT 
        		sp.Datetime, sp.tickerName, sp.predTmEntry2 AS PredEntry, sp.predTmExit2 AS PredExit, sp.predTmOpen as PredOpen, 
                round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
                CASE 
                    WHEN (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) 
                    THEN 1 
                    ELSE 0 
                END AS TmPL,
                CASE 
                    WHEN (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) 
                    THEN 0
                    WHEN (sp.Entry2 <= sp.predTmEntry2 AND sp.predTmEntry2 >= sp.[Close]) OR (sp.predTmEntry2 >= sp.[Close])
                    THEN 1 
                    ELSE 0 
                END AS gotLoss,
                CASE 
                    WHEN (sp.Entry2 <= sp.predTmEntry2 AND sp.Exit2 >= sp.predTmExit2) 
                    THEN ROUND(((sp.predTmExit2 - sp.predTmEntry2) / sp.predTmExit2) * 100, 2)
                    ELSE ROUND(((sp.[Close] - sp.predTmEntry2) / sp.[Close]) * 100, 2)
                END AS ProfitPercent,
                CASE 
                    WHEN (sp.Entry2 <= sp.predTmEntry2) OR (sp.predTmEntry2 >= sp.[Close])
                    THEN 1 
                    ELSE 0 
                END AS gotEntry
            FROM simulationPrediction AS sp
            WHERE sp.Datetime = @inputDate
        ),
        cte2 AS (
            SELECT sp.tickerName, COUNT(sp.Datetime) AS counts, @inputDate as Datetime, ETEXProfit
                FROM simulationPrediction AS sp
                LEFT JOIN (SELECT tickerName, EtEx2Profit as ETEXProfit FROM simulationPrediction WHERE Datetime=@inputDate) p ON p.tickerName=sp.tickerName
                WHERE Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2
                AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, @inputDate) AS DATE)
                AND CAST(Datetime AS DATE) < CAST(@inputDate AS DATE)
                GROUP BY sp.tickerName, ETEXProfit
        )
        SELECT TOP 20 c1.Datetime, c1.tickerName, c1.TmPL, c1.gotEntry, c1.gotLoss, 
        c1.PredEntry, c1.PredExit, c2.counts, c2.ETEXProfit, c1.PredOpen, c1.[AOpen/POpen-Diff], c1.ProfitPercent 
        FROM cte1 c1 LEFT JOIN cte2 c2 ON c1.tickerName = c2.tickerName
        ORDER BY c2.counts DESC, c2.ETEXProfit DESC;
        '''
        df = pd.read_sql(query, self.cnxn)
        return df   

    def getDateList(self):
        query = '''
        SELECT DISTINCT(CAST(Datetime AS DATE)) as Datetime FROM simulationPrediction
        WHERE Datetime >= DATEADD(MONTH, 1, (SELECT MIN(Datetime) FROM simulationPrediction))
        ORDER BY Datetime ASC
        '''
        dateList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Datetime'].astype(str).tolist()
        return dateList

    def analyze(self):
        dateList = self.getDateList()
        df = pd.concat([self.fetchData(inputDate) for inputDate in tqdm(dateList, desc='Get Market Data')]).fillna(0)
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
        return df
    
    @staticmethod
    def getMarketPhases(df):
        tickerDetails = pd.DataFrame({'tickerName': (df['tickerName']).tolist(), 'start_date': (df['Date'] - pd.Timedelta(days=5)).astype(str).tolist(), 'end_date': (df['Date']  + pd.Timedelta(days=1)).astype(str).tolist(), 'Date': df['Date'].tolist()}).to_dict('records')
        results = pd.DataFrame([MarketAnalyzer(kwargs).analyze() for kwargs in tqdm(tickerDetails, desc='Get Market Phases')])
        results['Accumulation_rank'] = results.groupby('Date')['Accumulation'].rank(ascending=False)
        results['Advancing_rank'] = results.groupby('Date')['Advancing'].rank(ascending=False)
        results['Distribution_rank'] = results.groupby('Date')['Distribution'].rank(ascending=True)
        results['Declining_rank'] = results.groupby('Date')['Declining'].rank(ascending=True)
        weights = {
            'Accumulation_rank': 0.25,
            'Advancing_rank': 0.25,
            'Distribution_rank': 0.25,
            'Declining_rank': 0.25
        }
        results['Composite_score'] = (results['Accumulation_rank'] * weights['Accumulation_rank'] +
                                 results['Advancing_rank'] * weights['Advancing_rank'] +
                                 results['Distribution_rank'] * weights['Distribution_rank'] +
                                 results['Declining_rank'] * weights['Declining_rank'])
        results['buyPriority'] = results.groupby('Date')['Composite_score'].rank(ascending=True)
        results = results.sort_values(by=['Date', 'buyPriority'], ascending=[False, True])
        results = results[['Date', 'tickerName', '315Close', 'Accumulation', 'Advancing', 'Distribution', 'Declining', 'buyPriority']]
        df = df.merge(results, how='left', on=['Date','tickerName'])
        return df


class StockPredictorAnalyzer:
    def __init__(self):
        self.cnxn = cnxn(VAR.db_mkanalyzer)

    def fetchData(self, inputDate):
        query = f'''
        DECLARE @inputDate DATE;
        SET @inputDate='{inputDate}';
        WITH cte AS (
        SELECT 
            tps.Datetime as Date, 
            tps.tickerName, 
        	tps.modelAccuacy,
            CASE 
                WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
                THEN 1 
                ELSE 0 
            END AS TmPL, 
            tps.TmPLPred,
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
            END AS ProfitPercent,
            round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
            Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
            [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose
        FROM topPriorityStats AS tps
        LEFT JOIN simulationPrediction AS sp 
        ON tps.tickerName = sp.tickerName AND tps.Datetime = sp.Datetime
        WHERE tps.Datetime = @inputDate --AND tps.TmPLPred = 1
        ),
        cte1 AS (
            SELECT sp.tickerName, COUNT(sp.Datetime) AS counts, @inputDate as Datetime, ETEXProfit
            FROM simulationPrediction AS sp
            LEFT JOIN (SELECT tickerName, EtEx2Profit as ETEXProfit FROM simulationPrediction WHERE Datetime=@inputDate) p ON p.tickerName=sp.tickerName
            WHERE Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2
            AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, @inputDate) AS DATE)
            AND CAST(Datetime AS DATE) < CAST(@inputDate AS DATE)
            GROUP BY sp.tickerName, ETEXProfit
                )
        SELECT  * FROM cte c1 
        LEFT JOIN cte1 c2 ON c1.tickerName = c2.tickerName
        ORDER BY c2.counts DESC, c1.modelAccuacy DESC
        '''
        df = pd.read_sql(query, self.cnxn)
        return df   

    def getDateList(self):
        query = '''
        SELECT DISTINCT(Datetime) FROM simulationPrediction
        WHERE Datetime >= DATEADD(MONTH, -3, (SELECT MAX(Datetime) FROM simulationPrediction))
        ORDER BY Datetime ASC
        '''
        dateList = pd.read_sql(query, self.cnxn)['Datetime'].dt.date.astype(str).tolist()
        return dateList

    def analyze(self):
        dateList = self.getDateList()
        df = pd.concat([self.fetchData(inputDate) for inputDate in tqdm(dateList, desc='Get Market Predictor Data')]).fillna(0)
        df = df.sort_values(by=['Datetime', 'counts', 'modelAccuacy', 'ETEXProfit'], ascending=[False, False, False, False])
        df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Datetime', 'tickerName', 'TmPL', 'TmPLPred', 'gotEntry', 'gotLoss', 
         'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 'counts', 'modelAccuacy', 'ETEXProfit', 'ProfitPercent'
         ])
        df.rename(columns={'modelAccuacy': 'modelAccuracy', 'TmPLPred': 'TmPredPL'}, inplace=True)
        df['TmPredPL'] = df['TmPredPL'].astype(int)
        return df
  
    def top1(self, df):
        df_sorted = df.sort_values(by=['Datetime', 'counts', 'modelAccuracy', 'ETEXProfit'], ascending=[False, False, False, False])
        df_sorted['RankValue'] = 0
        df_sorted.loc[df_sorted.groupby('Datetime').head(1).index, 'RankValue'] = 100
        df = df_sorted.groupby('Datetime').head(1).reset_index(drop=True)
        df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
        df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
        df['dayName'] = pd.to_datetime(df['Datetime']).dt.day_name()
        df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Datetime', 'dayName', 'tickerName', 'TmPL', 'TmPLPred', 'gotEntry', 'gotLoss', 'counts', 'modelAccuacy', 'ETEXProfit',
         'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
         'ProfitPercent', 'investedAmount', 'profitAmount', 'RankValue'])
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Datetime']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary

    def top2(self, df):
        df_sorted = df.sort_values(by=['Datetime', 'counts', 'modelAccuracy', 'ETEXProfit'], ascending=[False, False, False, False])
        df_sorted['RankValue'] = 50
        df_sorted.loc[df_sorted.groupby('Datetime').head(1).index, 'RankValue'] = 60
        df_sorted.loc[df_sorted.groupby('Datetime').head(2).tail(1).index, 'RankValue'] = 40
        df = df_sorted.groupby('Datetime').head(2).reset_index(drop=True)
        df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
        df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
        df['dayName'] = pd.to_datetime(df['Datetime']).dt.day_name()
        df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Datetime', 'dayName', 'tickerName', 'TmPL', 'TmPLPred', 'gotEntry', 'gotLoss', 'counts', 'modelAccuacy', 'ETEXProfit',
         'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
         'ProfitPercent', 'investedAmount', 'profitAmount', 'RankValue'])
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Datetime']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary

    def top3(self, df):
        df_sorted = df.sort_values(by=['Datetime', 'counts', 'modelAccuracy', 'ETEXProfit'], ascending=[False, False, False, False])
        df_sorted['RankValue'] = 0
        df_sorted.loc[df_sorted.groupby('Datetime').head(1).index, 'RankValue'] = 50
        df_sorted.loc[df_sorted.groupby('Datetime').head(2).tail(1).index, 'RankValue'] = 30
        df_sorted.loc[df_sorted.groupby('Datetime').head(3).tail(1).index, 'RankValue'] = 20
        df = df_sorted.groupby('Datetime').head(3).reset_index(drop=True)
        df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
        df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
        df['dayName'] = pd.to_datetime(df['Datetime']).dt.day_name()
        df = df.loc[:, ~df.columns.duplicated()].reindex(columns=['Datetime', 'dayName', 'tickerName', 'TmPL', 'TmPLPred', 'gotEntry', 'gotLoss', 'counts', 'modelAccuacy', 'ETEXProfit',
         'ActualOpen', 'ActualClose', 'ActualEntry', 'ActulExit', 'PredEntry', 'PredExit', 'PredOpen', 'AOpen/POpen-Diff', 
         'ProfitPercent', 'investedAmount', 'profitAmount', 'RankValue'])
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Datetime']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary
    
    

class TopPredictor:
    def __init__(self):
        self.cnxn = cnxn(VAR.db_mkanalyzer)

    def fetchData(self):
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
            --WHERE TmPredPL = 1 --and TmPredgotLoss = 0
        ),
        PreviousLow AS (
            SELECT
                CAST(Datetime AS Date) AS Date,
                tickerName,
                [Open] as TodayOpen,
                [High] as TodayHigh,
                [Low] as TodayLow,
        		[Close] as TodayClose,
                LAG([Low], 1) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev1DayLow,
                LAG([Low], 2) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev2DayLow,
        		LAG([High], 1) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev1DayHigh,
                LAG([High], 2) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev2DayHigh
            FROM mkanalyzer.dbo.mkDayFeature
            WHERE Datetime >= DATEADD(MONTH, -1, (SELECT MIN(CAST(Datetime AS Date)) FROM mkTopPrediction))
        )
        SELECT c.Date, c.tickerName, TmPL, TmPredPL, TmPredgotLoss, gotEntry, gotLoss, PredEntry, PredExit, ActualProfit, PredProfit, PredOpen, 
        ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], 
        successCount, modelAccuracyPL, epochLossPL, modelAccuracygotLoss, epochLossgotLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal, TmPredPL5Summary, TmPredgotLoss5Summary, rn as Priority,
        pl.TodayOpen, pl.TodayHigh, pl.TodayLow, pl.TodayClose,
        CASE
            WHEN pl.TodayLow > COALESCE(pl.Prev1DayLow, 0) 
                AND COALESCE(pl.Prev1DayLow, 0) > COALESCE(pl.Prev2DayLow, 0) 
            THEN 1
            ELSE 0
        END AS Prev2DaysNotBreakLow,
        CASE
            WHEN pl.TodayHigh > COALESCE(Prev1DayHigh, 0) AND COALESCE(Prev1DayHigh, 0) > COALESCE(Prev2DayHigh, 0) THEN 1
            ELSE 0
        END AS Prev2DayBreakHigh
        FROM cte c
        left join PreviousLow pl on pl.Date=c.Date and pl.tickerName=c.tickerName
        --WHERE rn = 1 AND TmPredPL = 1 --AND gotEntry = 1
        ORDER BY c.Date DESC;
        '''
        df = pd.read_sql(query, self.cnxn)  
        def getPN_Data(dct):
            query = f'''
            WITH AggregatedValues AS (
                SELECT 
                    CAST(Datetime AS DATE) AS Date,
                    ROUND(SUM((High - Low) / CASE WHEN High = 0 THEN 1 ELSE High END), 2) AS PN,
                    ROUND(SUM(CASE WHEN [Open] < [Close] THEN (High - Low) / CASE WHEN High = 0 THEN 1 ELSE High END ELSE 0 END), 2) AS P,
                    ROUND(SUM(CASE WHEN [Open] > [Close] THEN (High - Low) / CASE WHEN High = 0 THEN 1 ELSE High END ELSE 0 END), 2) AS N
                FROM [{dct.get('tickerName')}]
            	WHERE CAST(Datetime AS DATE) IN ({dct.get('Date')})
                GROUP BY CAST(Datetime AS DATE)
            )
            SELECT 
                Date, '{dct.get('tickerName')}' AS tickerName,
                ROUND(PN, 2) AS PN,
                ROUND(P, 2) AS P,
                ROUND(N, 2) AS N,
                CASE 
                    WHEN N = 0 THEN 0
                    ELSE ROUND(P / NULLIF(N, 0), 2)
                END AS RPN
            FROM AggregatedValues;
            '''
            df1 = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster)) 
            return df1
        df1 = pd.concat([getPN_Data(dct) for dct in tqdm(df.groupby('tickerName')['Date'].apply(lambda x: ", ".join(["'"+item+"'" for item in x.astype(str).unique()])).reset_index().to_dict('records'))]).reset_index(drop=True)
        df = df.merge(df1, how='left', on = ['Date', 'tickerName'])
        return df   
    

    @staticmethod
    def ExecutionDetails(row):
        try:
            query = f'''
            SELECT [Datetime], [Open], [Low], [High], [Close]
            FROM [{row.get("tickerName")}]
            WHERE [Datetime] >= (
                    SELECT TOP 1 [Datetime]
                    FROM [{row.get("tickerName")}]
                    WHERE 
                        CAST([Datetime] AS DATE) = '{row.get("predDate")}'
                        AND (
                            [Open] <= {row.get("PredEntry")} OR
                            [Low] <= {row.get("PredEntry")} OR
                            [High] <= {row.get("PredEntry")} OR
                            [Close] <= {row.get("PredEntry")}
                        )
                )
                AND CAST([Datetime] AS DATE) = '{row.get("predDate")}'
            ORDER BY [Datetime] ASC
            '''
            df = pd.read_sql(query, cnxn(VAR.db_mkintervalmaster))
            df = df.iloc[:-2]
            if not df.empty:
                dfF = df.iloc[0:1].rename(columns={'Datetime': 'PredEnrtyDatetime'})
                dfF['LAPEDatetime'] = df[df['Low'] == df['Low'].min()].iloc[0]['Datetime']
                dfF['LowAfterPredEntry'] = df['Low'].min()
                dfF['HAPEDatetime'] = df[df['High'] == df['High'].max()].iloc[0]['Datetime']
                dfF['HighAfterPredEntry'] = df['High'].max()
                dfF['315Close'] = df['Open'].iloc[-1]
                dfF['Date'] = row.get('Date')
                dfF['predDate'] = row.get('predDate')
                dfF['tickerName'] = row.get('tickerName')
                dfF['PredEntry'] = row.get('PredEntry')
                dfF['setTarget'] = (dfF['PredEntry'] + dfF['PredEntry']*0.0125).round(2)
                dfF['setSL'] = (dfF['PredEntry'] - dfF['PredEntry']*0.0125).round(2)
                dfF = dfF[['Date', 'predDate', 'tickerName', 'PredEnrtyDatetime', 'Open', 'Low', 'High', 'Close', '315Close', 'setTarget', 'setSL', 'HAPEDatetime', 'HighAfterPredEntry', 'LAPEDatetime', 'LowAfterPredEntry']]
                return dfF
        except:
            pass
        
    def analyze(self):
        df = self.fetchData().fillna('NA')
        df = df.sort_values(by=['Date', 'successCount', 'epochLossPL'], ascending=[False, False, True])
        # df['TmPredPL'] = df['TmPredPL5Summary'].str.split(', ').apply(lambda x: stats.mode([item.split(' :: ')[0] for item in x]).mode[0]).astype(int)
        df['TmPredPL'] = (
            df['TmPredPL5Summary']
            .str.split(', ')
            .apply(lambda x: Counter([item.split(' :: ')[0] for item in x]).most_common(1)[0][0])
            .astype(int)
        )
        # df['TmPredgotLoss'] = df['TmPredgotLoss5Summary'].str.split(', ').apply(lambda x: stats.mode([item.split(' :: ')[0] for item in x]).mode[0]).astype(int)
        df['TmPredgotLoss'] = (
            df['TmPredgotLoss5Summary']
            .str.split(', ')
            .apply(lambda x: Counter([item.split(' :: ')[0] for item in x]).most_common(1)[0][0])
            .astype(int)
        )
        dfP = df[['Date']].drop_duplicates().reset_index(drop=True)
        dfP['predDate'] = dfP['Date'].shift(1)
        dfP = dfP.dropna().reset_index(drop=True)
        dfP = dfP.merge(df[['Date', 'tickerName', 'PredEntry', 'gotEntry']], how='left', on=['Date'])
        dfP = pd.concat([self.ExecutionDetails(row) for row in dfP.to_dict('records')]).reset_index(drop=True)
        df = df.merge(dfP, how='left', on=['Date', 'tickerName'])
        filtered_df = df[df['TmPredPL'] == 1]
        filtered_df['Seq'] = filtered_df.groupby('Date').cumcount() + 1
        df = pd.merge(df, filtered_df[['Date', 'tickerName', 'Seq']], on=['Date', 'tickerName'], how='left')
        df['Seq'].fillna(0, inplace=True)
        df['Seq'] = df['Seq'].astype(int)
        df['125Profit'] = df.apply(lambda row: 1.25 if row['gotEntry'] == 1 and row['setTarget'] <= row['HighAfterPredEntry'] else  round((1-row['PredEntry']/row['315Close'])*100, 2) if row['gotEntry'] == 1 else 0, axis=1)
        return df
  
