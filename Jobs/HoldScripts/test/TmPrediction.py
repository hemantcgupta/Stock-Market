# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 01:40:17 2024

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
        WHERE tas.Date = '{inputDate}' AND tas.TmPredPL = 1
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



analyzer = StockAnalyzer()
df = analyzer.analyze()
# df = analyzer.getMarketPhases(df)
df1, allProfit1, monthSummary1 = analyzer.top1(df)
df2, allProfit2, monthSummary2 = analyzer.top2(df)
df3, allProfit3, monthSummary3 = analyzer.top3(df)
# df.merge(results, how='left', on=['Date','tickerName']).head(30)[['Date', 'tickerName', 'TmPL', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority', 'ProfitPercent']].rename(columns={'AccuracyScore:Mode': 'ASM', "ProfitPercent": 'PL', 'ETEXProfit': 'ETEXPL'})


# tt = pd.merge(results, df[['Date', 'tickerName', 'TmPL', 'ProfitPercent']], how='left', on=['Date', 'tickerName'])
# tt['RankValue'] = 0
# tt.loc[tt.groupby('Date').head(1).index, 'RankValue'] = 100
# tt = tt.groupby('Date').head(1).reset_index(drop=True)
# tt['investedAmount'] = (tt['RankValue']*10000/100).astype(int)
# tt['profitAmount'] = (tt['ProfitPercent']*tt['investedAmount']/100).astype(int)
# tt.groupby(pd.to_datetime(tt['Date']).dt.month) ['profitAmount'].sum().reset_index()
# tt.head(30)
df1.tail(30)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class StockPredictor:
    def __init__(self, df, test_size=0.2, random_state=42, lr=0.001, num_epochs=1000):
        self.df = df
        self.features = ['counts', 'ETEXProfit', 'buyPriority']
        self.target = 'TmPL'
        self.test_size = test_size
        self.random_state = random_state
        self.lr = lr
        self.num_epochs = num_epochs

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
    
    def prepration_data(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        self.df = df.sort_values(by='Date').reset_index(drop=True)

    def preprocess_data(self):
        maxModelIndex = self.df[self.df['Date'] == self.df['Date'].max()].index.min()
        minEvaluateIndex = self.df[self.df['Date'] >= (self.df['Date'].max() - pd.DateOffset(months=1))].index.min()
        X = self.df[self.features].values
        y = self.df[self.target].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return X, y, maxModelIndex, minEvaluateIndex

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size):
            super(StockPredictor.NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    def train_model(self, X_train, y_train):
        model = self.NeuralNetwork(input_size=X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch + 1) % 10 == 0:
            #     print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        return model

    def evaluate_model(self, model, X):
        model.eval()
        with torch.no_grad():
            predictions = model(X).numpy()
        return predictions

    def analyze_results(self, minEvaluateIndex, predictions):
        self.df = self.df.iloc[minEvaluateIndex:].reset_index(drop=True)
        self.df['TmBuyP'] = predictions
        self.df['TmBuyP'] = self.df['TmBuyP'].round(2)
        self.df = self.df.sort_values(by=['Date', 'TmBuyP'], ascending=[True, False])
        topPredictions = self.df.groupby('Date').head(1).reset_index(drop=True)
        profitScore = round((topPredictions[topPredictions['Date'] < topPredictions['Date'].max()]['TmPL'].value_counts(normalize=True) * 100).round(2).to_dict().get(1, 0), 2)
        profitPercent = round(topPredictions[topPredictions['Date'] < topPredictions['Date'].max()]['ProfitPercent'].sum(), 2)
        self.df = self.df[self.df['Date'] == self.df['Date'].max()].reset_index(drop=True)
        self.df = self.df[['Date', 'tickerName', 'TmBuyP', 'TmPredPL', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'PredEntry', 'PredExit', 'PredOpen', 'buyPriority']]
        self.df['profitScore'] = profitScore
        self.df['profitPercent'] = profitPercent
        return self.df, profitScore, profitPercent

    def run(self):
        results_list = []
        best_dfPred, best_profitScore, best_profitPercent = None, 0, 0
        best_tickerName = {}
        df = self.getMarketPhases(self.df)
        for i in range(10):
            self.prepration_data(df)
            X, y, maxModelIndex, minEvaluateIndex = self.preprocess_data()
            X_train, y_train = X[:minEvaluateIndex-1], y[:minEvaluateIndex-1]
            model = self.train_model(X_train, y_train)
            predictions = self.evaluate_model(model, X[minEvaluateIndex:])
            dfPred, profitScore, profitPercent = self.analyze_results(minEvaluateIndex, predictions)
            tickerName = dfPred[dfPred['TmBuyP'] == max(dfPred['TmBuyP'])].reset_index(drop=True)['tickerName'][0]
            if not best_tickerName or max(best_tickerName, key=lambda x: best_tickerName[x]) == tickerName:
                if profitScore >= best_profitScore:
                    if profitPercent > best_profitPercent:
                        best_profitScore = profitScore
                        best_profitPercent = profitPercent
                        best_dfPred = dfPred
                        results_list.append(dfPred)
        return results_list, best_dfPred, best_profitScore, best_profitPercent


def Delete_max_date(dbName, table_name, row):
    try:
        ticker_name = row['tickerName']
        max_date = row['Date']
        conn = cnxn(dbName)
        cursor = conn.cursor()
        delete_query = f"DELETE FROM {table_name} WHERE tickerName = '{ticker_name}' and Date = '{max_date}';"
        cursor.execute(delete_query)
        conn.commit()
        return True
    except:
        return False
    
    
def jobMarketPhasesModel(df):
    query = '''
    SELECT DISTINCT(Date) as Date FROM topAccurateStats
    WHERE Date >= DATEADD(MONTH, 1, (SELECT MIN(Date) FROM topAccurateStats))
    ORDER BY Date ASC
    '''
    dateList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Date'].astype(str).tolist()
    try:
        query = f'''
        SELECT DISTINCT(CAST(Date AS DATE)) as Date FROM topPriorityStats
        '''
        alredyPresent = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Date'].astype(str).tolist()
        dateList = [item for item in dateList if (item == dateList[-1]) or (item not in alredyPresent)]
    except:
        pass
    resultDict = {}
    for date in dateList:
        print(f'{date:#^75}')
        predictor = StockPredictor(df[pd.to_datetime(df['Date']) <= date].reset_index(drop=True))
        results, dfPred, best_profitScore, best_profitPercent = predictor.run()
        dfPred.apply(lambda row: Delete_max_date(VAR.db_mkanalyzer, 'topPriorityStats', row), axis = 1)
        result = Data_Inserting_Into_DB(dfPred, VAR.db_mkanalyzer, 'topPriorityStats', 'append')
        resultDict[date] = result
    return resultDict
    

if __name__ == "__main__":
    result = jobMarketPhasesModel(df)
# query = '''
# DECLARE @inputDate Date;
# SET @inputDate = '2024-02-01';
# WITH cte AS (
#     SELECT 
# 		sp.Datetime, sp.tickerName,
#         CASE 
#             WHEN (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) 
#             THEN 1 
#             ELSE 0 
#         END AS TmPL,
#         CASE 
#             WHEN (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) 
#             THEN 0
#             WHEN (sp.Entry2 <= sp.predTmEntry2 AND sp.predTmEntry2 >= sp.[Close]) OR (sp.predTmEntry2 >= sp.[Close])
#             THEN 1 
#             ELSE 0 
#         END AS gotLoss,
#         CASE 
#             WHEN (sp.Entry2 <= sp.predTmEntry2 AND sp.Exit2 >= sp.predTmExit2) 
#             THEN ROUND(((sp.predTmExit2 - sp.predTmEntry2) / sp.predTmExit2) * 100, 2)
#             ELSE ROUND(((sp.[Close] - sp.predTmEntry2) / sp.[Close]) * 100, 2)
#         END AS ProfitPercent,
#         CASE 
#             WHEN (sp.Entry2 <= sp.predTmEntry2) OR (sp.predTmEntry2 >= sp.[Close])
#             THEN 1 
#             ELSE 0 
#         END AS gotEntry,
#         sp.predTmEntry2 AS PredEntry, 
#         sp.Exit2 AS ActualExit
#     FROM simulationPrediction AS sp
#     WHERE sp.Datetime = @inputDate
# ),
# cte1 AS (
#     SELECT 
#         sp.tickerName, 
#         COUNT(sp.Datetime) AS counts, 
#         @inputDate AS Datetime, 
#         p.ETEXProfit
#     FROM simulationPrediction AS sp
#     LEFT JOIN (
#         SELECT 
#             tickerName, 
#             EtEx2Profit AS ETEXProfit 
#         FROM simulationPrediction 
#         WHERE Datetime = @inputDate
#     ) p ON p.tickerName = sp.tickerName
#     WHERE sp.Entry2 <= sp.predTmEntry2 
#     AND sp.Exit2 >= sp.predTmExit2
#     AND CAST(sp.Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, @inputDate) AS DATE)
#     AND CAST(sp.Datetime AS DATE) < CAST(@inputDate AS DATE)
#     GROUP BY sp.tickerName, p.ETEXProfit
# )
# SELECT * 
# FROM cte c1 
# LEFT JOIN cte1 c2 ON c1.tickerName = c2.tickerName
# ORDER BY c2.counts DESC;
# '''
# df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))
# df = df.loc[:, ~df.columns.duplicated()].dropna().reset_index(drop=True)

# =============================================================================
# 
# =============================================================================
class StockMarketPhasesAnalyzer:
    def __init__(self):
        self.cnxn = cnxn(VAR.db_mkanalyzer)

    def fetchData(self, inputDate):
        query = f'''
        WITH cte AS (
        SELECT 
            tps.Date, 
            tps.tickerName, 
            tps.[AccuracyScore:Mode],
			tps.TmBuyP,
            tps.buyPriority,
            CASE 
                WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
                THEN 1 
                ELSE 0 
            END AS TmPL, 
            tps.TmPredPL,
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
        FROM topPriorityStats AS tps
        LEFT JOIN simulationPrediction AS sp 
        ON tps.tickerName = sp.tickerName AND tps.Date = sp.Datetime
        WHERE tps.Date = '{inputDate}' AND tps.TmPredPL = 1
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
		order by c1.TmBuyP  desc
        '''
        df = pd.read_sql(query, self.cnxn)
        return df   

    def getDateList(self):
        query = '''
        SELECT DISTINCT(Date) FROM topAccurateStats
        WHERE Date >= DATEADD(MONTH, 1, (SELECT MIN(Date) FROM topAccurateStats))
        ORDER BY Date ASC
        '''
        dateList = pd.read_sql(query, self.cnxn)['Date'].astype(str).tolist()
        return dateList

    def analyze(self):
        dateList = self.getDateList()
        df = pd.concat([self.fetchData(inputDate) for inputDate in tqdm(dateList, desc='Get Market Data')]).fillna(0)
        df = df.sort_values(by=['Date', 'TmBuyP'], ascending=[False, False])
        return df
    
    def top1(self, df):
        df_sorted = df.sort_values(by=['Date', 'TmBuyP'], ascending=[False, False])
        df_sorted['RankValue'] = 0
        df_sorted.loc[df_sorted.groupby('Date').head(1).index, 'RankValue'] = 100
        df = df_sorted.groupby('Date').head(1).reset_index(drop=True)
        df['investedAmount'] = (df['RankValue']*10000/100).astype(int)
        df['profitAmount'] = (df['ProfitPercent']*df['investedAmount']/100).astype(int)
        df['dayName'] = pd.to_datetime(df['Date']).dt.day_name()
        df = df[df['gotEntry'] == 1].reset_index(drop=True)
        allProfit = df['profitAmount'].sum().round(2)
        monthSummary = df.groupby(pd.to_datetime(df['Date']).dt.month) ['profitAmount'].sum().reset_index()
        return df, allProfit, monthSummary
    
analyzerPhases = StockMarketPhasesAnalyzer()
dfP = analyzerPhases.analyze()    
dfP1, PallProfit1, PmonthSummary1 = analyzerPhases.top1(dfP)    

dfP.iloc[:30][['Date', 'tickerName', 'TmPL', 'TmBuyP','gotEntry', 'gotLoss', 'ProfitPercent', 'buyPriority']]
dfP[dfP['TmPL'] == 1].head(30)
dfP.groupby('Date').head(1).reset_index(drop=True).head(30)[['Date', 'tickerName', 'TmPL', 'TmBuyP','gotEntry', 'gotLoss', 'ProfitPercent']]
# =============================================================================
# 
# =============================================================================
# predictor = StockPredictor(df)
# results, dfPred, best_profitScore, best_profitPercent = predictor.run()


# def analyze_results(self, minEvaluateIndex, predictions):
#     self.df = self.df.iloc[minEvaluateIndex:].reset_index(drop=True)
#     self.df['TmBuyP'] = predictions
#     self.df = self.df.sort_values(by=['Date', 'TmBuyP'], ascending=[True, False])
#     top_predictions = self.df.groupby('Date').head(1).reset_index(drop=True)
#     profit_score = (top_predictions[(pd.to_datetime(top_predictions['Date']) >= (top_predictions['Date'].max() - pd.DateOffset(months=1))) & 
#                     (pd.to_datetime(top_predictions['Date']) < pd.Timestamp(top_predictions['Date'].max()))]
#                     .reset_index(drop=True)['TmPL'].value_counts(normalize=True) * 100).round(2).to_dict().get(1, 0)
#     filtered_predictions = top_predictions[top_predictions['gotEntry'] == 1].reset_index(drop=True)
#     filtered_predictions.loc[filtered_predictions.groupby('Date').head(1).index, 'RankValue'] = 100
#     filtered_predictions = filtered_predictions.groupby('Date').head(1).reset_index(drop=True)
#     filtered_predictions['investedAmount'] = (filtered_predictions['RankValue'] * 10000 / 100).astype(int)
#     filtered_predictions['profitAmount'] = (filtered_predictions['ProfitPercent'] * filtered_predictions['investedAmount'] / 100).astype(int)
#     print('ALL Profit: ', filtered_predictions['profitAmount'].sum().round(2))
#     monthly_summary = filtered_predictions.groupby(pd.to_datetime(filtered_predictions['Date']).dt.month)['profitAmount'].sum().reset_index()
#     monthly_summary['score'] = profit_score
#     print(monthly_summary)
#     summary_dict = {
#         'summary': monthly_summary,
#         'df': top_predictions.tail(1)[['Date', 'tickerName', 'TmPredPL', 'TmBuyP', 'PredOpen', 'PredEntry', 'PredExit', 'ETEXProfit', 'buyPriority']]
#     }
#     return summary_dict
# =============================================================================
# 
# =============================================================================
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# lst = []
# for i in range(10):
#     features = ['counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority']
#     X = df[features].values
#     y = df['TmPL'].values
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
#     class NeuralNetwork(nn.Module):
#         def __init__(self, input_size):
#             super(NeuralNetwork, self).__init__()
#             self.fc1 = nn.Linear(input_size, 128)
#             self.fc2 = nn.Linear(128, 64)
#             self.fc3 = nn.Linear(64, 32)
#             self.fc4 = nn.Linear(32, 1)
            
#         def forward(self, x):
#             x = torch.relu(self.fc1(x))
#             x = torch.relu(self.fc2(x))
#             x = torch.relu(self.fc3(x))
#             x = self.fc4(x)
#             return x
#     model = NeuralNetwork(input_size=X.shape[1])
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     # X_train, y_train = X, y
#     num_epochs = 1000
#     for epoch in range(num_epochs):
#         model.train()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X).numpy()
#     df['TmBuyP'] = predictions
#     df = df.sort_values(by=['Date', 'TmBuyP'], ascending=[True, False])
#     tt = df.groupby('Date').head(1).reset_index(drop=True)
#     score = (tt[(pd.to_datetime(tt['Date']) >= (tt['Date'].max() - pd.DateOffset(months=1))) & (pd.to_datetime(tt['Date']) < pd.Timestamp(tt['Date'].max()))].reset_index(drop=True)['TmPL'].value_counts(normalize=True) * 100).round(2).to_dict().get(1, 0)
#     ttt = tt[tt['gotEntry'] == 1].reset_index(drop=True)
#     # print('ALL Profit: ', ttt['ProfitPercent'].sum().round(2))
#     # summary = ttt.groupby(pd.to_datetime(ttt['Date']).dt.month) ['ProfitPercent'].sum().reset_index()
#     ttt.loc[ttt.groupby('Date').head(1).index, 'RankValue'] = 100
#     ttt = ttt.groupby('Date').head(1).reset_index(drop=True)
#     ttt['investedAmount'] = (ttt['RankValue']*10000/100).astype(int)
#     ttt['profitAmount'] = (ttt['ProfitPercent']*ttt['investedAmount']/100).astype(int)
#     print('ALL Profit: ', ttt['profitAmount'].sum().round(2))
#     summary = ttt.groupby(pd.to_datetime(ttt['Date']).dt.month) ['profitAmount'].sum().reset_index()
#     summary['score'] = score
#     print(summary)
#     dct = {'summary': summary, 'df': tt.tail(1)[['Date', 'tickerName', 'TmPredPL', 'TmBuyP', 'PredOpen', 'PredEntry', 'PredExit', 'ETEXProfit', 'buyPriority']]}
#     lst.append(dct)


# df.tail(30)[['Date', 'tickerName', 'TmPL', 'TmPredPL', 'TmBuyP', 'ETEXProfit']]
# tt.tail(30)[['Date', 'tickerName', 'TmPL', 'TmPredPL', 'TmBuyP', 'ETEXProfit']]
# ttt.tail(30)[['Date', 'tickerName', 'TmPL', 'TmPredPL', 'TmBuyP', 'ETEXProfit', 'ProfitPercent']]


# ttt.loc[ttt.groupby('Date').head(1).index, 'RankValue'] = 100
# ttt = ttt.groupby('Date').head(1).reset_index(drop=True)
# ttt['investedAmount'] = (ttt['RankValue']*10000/100).astype(int)
# ttt['profitAmount'] = (ttt['ProfitPercent']*ttt['investedAmount']/100).astype(int)
# print('ALL Profit: ', ttt['profitAmount'].sum().round(2))
# summary = ttt.groupby(pd.to_datetime(ttt['Date']).dt.month) ['profitAmount'].sum().reset_index()
# print(summary)



# # =============================================================================
# # 
# # =============================================================================
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # Setting seed for reproducibility
# torch.manual_seed(42)

# # Features and labels
# features = ['counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority']
# X = df[features].values
# y = df['TmPL'].values

# # Standardization
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Convert to torch tensors
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# # Neural Network Definition
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)
#         self.dropout = nn.Dropout(0.5)  # Dropout to reduce overfitting
#         self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(32)
        
#     def forward(self, x):
#         x = torch.relu(self.bn1(self.fc1(x)))
#         x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
#         x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
#         x = self.fc4(x)
#         return x

# # Instantiate the model
# model = NeuralNetwork(input_size=X.shape[1])

# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Split data into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Training loop with early stopping
# num_epochs = 1000
# best_loss = float('inf')
# patience = 200
# trigger_times = 0

# for epoch in range(num_epochs):
#     model.train()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     # Early stopping
#     if loss.item() < best_loss:
#         best_loss = loss.item()
#         trigger_times = 0
#     else:
#         trigger_times += 1
        
#     if trigger_times >= patience:
#         print(f"Early stopping at epoch {epoch+1}")
#         break
    
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Inference
# model.eval()
# with torch.no_grad():
#     predictions = model(X).numpy()

# df['TmBuyP'] = predictions
# df = df.sort_values(by=['Date', 'TmBuyP'], ascending=[True, False])
# tt = df.groupby('Date').head(1).reset_index(drop=True)

# tt = tt[tt['gotEntry'] == 1].reset_index(drop=True)
# print('ALL Profit: ', tt['ProfitPercent'].sum().round(2))
# print(tt.groupby(pd.to_datetime(tt['Date']).dt.month) ['ProfitPercent'].sum().reset_index())
