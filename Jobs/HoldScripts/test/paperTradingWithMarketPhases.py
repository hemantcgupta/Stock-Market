# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:21:40 2024

@author: Hemant
"""
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
import yfinance as yf  
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.set_option('expand_frame_repr', True)
from Scripts.dbConnection import cnxn

class VAR:
    db_mkanalyzer = 'mkanalyzer'
    db_mkintervalmaster = 'mkintervalmaster'
    
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


    
analyzerPhases = StockMarketPhasesAnalyzer()
df = analyzerPhases.analyze()    
df = analyzerPhases.getMarketPhases(df)



# dfTop1 = df
# dfTop1 = df.sort_values(by=['Date', 'Accumulation', 'Advancing', 'Distribution', 'Declining'], ascending=[True, False, False, True, True])
dfTop1 = df.sort_values(by=['Date', 'buyPriority'], ascending=[True, True])
dfTop1['RankValue'] = 0
dfTop1.loc[dfTop1.groupby('Date').head(1).index, 'RankValue'] = 100
dfTop1 = dfTop1.groupby('Date').head(1).reset_index(drop=True)
dfTop1['investedAmount'] = (dfTop1['RankValue']*10000/100).astype(int)
dfTop1['profitAmount'] = (dfTop1['ProfitPercent']*dfTop1['investedAmount']/100).astype(int)
dfTop1 = dfTop1[dfTop1['gotEntry'] == 1]
dfTop1.groupby(pd.to_datetime(dfTop1['Date']).dt.month) ['profitAmount'].sum().reset_index()
dfTop1.tail(30)



# tt = df[['Date', 'tickerName', 'TmPL', 'gotEntry', 'counts', 'ETEXProfit', 'Accumulation', 'Advancing', 'Distribution', 'Declining', 'buyPriority']]
# tt[tt['TmPL'] == 1]
df_sorted = df.sort_values(by=['Date', 'counts', 'ETEXProfit'], ascending=[True, False, False])
df_sorted['seqCount'] = df_sorted.groupby('Date').cumcount() + 1
df_sorted = df_sorted.sort_values(by=['Date', 'seqCount'])
df_sorted = df_sorted.sort_values(by=['Date', 'buyPriority'], ascending=[True, True])
df_sorted['seqBuy'] = df_sorted.groupby('Date').cumcount() + 1
# df_sorted[df_sorted['TmPL'] == 1]
df_sorted['seq'] = (df_sorted['seqBuy']*df_sorted['seqCount']/20).round(2)
df_sorted.to_excel(r'C:\Users\heman\Desktop\tt.xlsx', index=False)

df_sorted = df_sorted.sort_values(by=['Date', 'seq'], ascending=[True, True])
df_sorted['RankValue'] = 0
df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 100
df_sorted = df_sorted.groupby('Date').head(1).reset_index(drop=True)
df_sorted['investedAmount'] = (df_sorted['RankValue']*10000/100).astype(int)
df_sorted['profitAmount'] = (df_sorted['ProfitPercent']*df_sorted['investedAmount']/100).astype(int)
df_sorted = df_sorted[df_sorted['gotEntry'] == 1]
df_sorted.groupby(pd.to_datetime(df_sorted['Date']).dt.month) ['profitAmount'].sum().reset_index()
