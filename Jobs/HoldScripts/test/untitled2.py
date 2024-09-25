# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:00:24 2024

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
        df = pd.concat([self.fetchData(inputDate) for inputDate in tqdm(dateList)]).fillna(0)
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

def getMarketPahses(self, df):
    tickerDetails = pd.DataFrame({'tickerName': (df['tickerName']).tolist(), 'start_date': (df['Date'] - pd.Timedelta(days=5)).astype(str).tolist(), 'end_date': (df['Date']  + pd.Timedelta(days=1)).astype(str).tolist(), 'Date': df['Date'].tolist()}).to_dict('records')
    results = pd.DataFrame([MarketAnalyzer(kwargs).analyze() for kwargs in tqdm(tickerDetails)])
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
    df.merge(results, how='left', on=['Date','tickerName']).head(30)
    df.merge(results, how='left', on=['Date','tickerName']).head(30)[['Date', 'tickerName', 'TmPL', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority', 'ProfitPercent']].rename(columns={'AccuracyScore:Mode': 'ASM', "ProfitPercent": 'PL'})
    # df1.merge(results, how='left', on=['Date','tickerName']).tail(30)[['Date', 'tickerName', 'TmPL', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority', 'ProfitPercent']].rename(columns={'AccuracyScore:Mode': 'ASM', "ProfitPercent": 'PL'})
        

analyzer = StockAnalyzer()
df = analyzer.analyze()
# df1, allProfit1, monthSummary1 = analyzer.top1(df)
# df2, allProfit2, monthSummary2 = analyzer.top2(df)
# df3, allProfit3, monthSummary3 = analyzer.top3(df)
df.merge(results, how='left', on=['Date','tickerName']).head(30)[['Date', 'tickerName', 'TmPL', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority', 'ProfitPercent']].rename(columns={'AccuracyScore:Mode': 'ASM', "ProfitPercent": 'PL', 'ETEXProfit': 'ETEXPL'})


def test(df):
    tt = df.merge(results, how='left', on=['Date','tickerName'])[['Date', 'tickerName', 'TmPL', 'gotEntry', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority', 'ProfitPercent']].rename(columns={'AccuracyScore:Mode': 'ASM', "ProfitPercent": 'PL', 'ETEXProfit': 'ETEXPL', 'gotEntry': 'gET','buyPriority': 'buyPRTY'})
    tt = tt.sort_values(by=['Date', 'counts', 'ASM', 'ETEXPL'], ascending=[True, False, False, False])
    # tt = tt.sort_values(by=['Date', 'counts', 'ASM', 'buyPRTY', 'ETEXPL'], ascending=[True, False, False, False, False])
    tt = tt.groupby('Date').head(1).reset_index(drop=True)
    print(round(tt[tt['gET'] == 1]['PL'].sum(), 2))
test(df)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.ensemble import BaggingRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Step 1: Prepare the Data
# features = ['counts', 'ASM', 'ETEXPL', 'buyPRTY']
# X = tt[features].values
# y = tt['TmPL'].values

# # Normalize the data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Convert data to PyTorch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# # Define a base neural network model
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Function to train a neural network model
# def train_model(X_train, y_train):
#     model = NeuralNetwork(input_size=X_train.shape[1])
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     num_epochs = 100
#     for epoch in range(num_epochs):
#         model.train()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     return model

# # Step 2: Train a Bagging Ensemble of Neural Networks
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# # Bagging with neural networks as base models
# class BaggingNN(BaggingRegressor):
#     def __init__(self, base_model, n_estimators, **kwargs):
#         self.base_model = base_model
#         self.models = [base_model() for _ in range(n_estimators)]
#         super().__init__(base_estimator=self.base_model, n_estimators=n_estimators, **kwargs)

#     def fit(self, X, y):
#         for model in self.models:
#             model = train_model(X, y)
#         return self

#     def predict(self, X):
#         predictions = np.zeros((len(X), len(self.models)))
#         for i, model in enumerate(self.models):
#             predictions[:, i] = model(X).detach().numpy().flatten()
#         return np.mean(predictions, axis=1)

# bagging_model = BaggingNN(base_model=lambda: NeuralNetwork(input_size=X_train.shape[1]), n_estimators=10)
# bagging_model.fit(X_train, y_train)

# # Step 3: Predict using the Bagging Model
# predictions = bagging_model.predict(X_tensor)

# # Step 4: Add the Predicted Column to the DataFrame
# tt['PtmPL'] = predictions


# tt['rank']=tt['PtmPL']/tt['buyPRTY']
# ttt = tt[['Date','tickerName', 'TmPL', 'PtmPL', 'gET','PL', 'buyPRTY','rank']]

# ttt = ttt.sort_values(by=['Date', 'PtmPL'], ascending=[True, False])
# # tt = tt.sort_values(by=['Date', 'counts', 'ASM', 'buyPRTY', 'ETEXPL'], ascending=[True, False, False, False, False])
# ttt = ttt.groupby('Date').head(1).reset_index(drop=True)
# print(round(ttt[ttt['gET'] == 1]['PL'].sum(), 2))

# ttt[ttt['gET'] == 1].tail(30)
# ttt.tail(30)

# print(ttt.groupby(pd.to_datetime(ttt['Date']).dt.month) ['PL'].sum().reset_index())




# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# # Prepare the data
# features = ['counts', 'ASM', 'ETEXPL', 'buyPRTY']
# X = tt[features]
# y = tt['TmPL']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Model training
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Example: Predict for a new date
# new_data = X_test.iloc[-1].values.reshape(1, -1)
# tmPL_pred = model.predict(new_data)

# tt['PtmPL'] = model.predict(X)




# tt.tail(30)[['Date','tickerName', 'TmPL', 'PtmPL', 'gET']]



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Prepare the Data
tt = df.merge(results, how='left', on=['Date','tickerName'])[['Date', 'tickerName', 'TmPL', 'gotEntry', 'counts', 'AccuracyScore:Mode', 'ETEXProfit', 'buyPriority', 'ProfitPercent']].rename(columns={'AccuracyScore:Mode': 'ASM', "ProfitPercent": 'PL', 'ETEXProfit': 'ETEXPL', 'gotEntry': 'gET','buyPriority': 'buyPRTY'})
tt = tt.sort_values(by=['Date', 'counts', 'ASM', 'ETEXPL'], ascending=[True, False, False, False])
features = ['counts', 'ASM', 'ETEXPL', 'buyPRTY']
X = tt[features].values
y = tt['TmPL'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Step 2: Define the Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
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

# Initialize the model, loss function, and optimizer
model = NeuralNetwork(input_size=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: Make Predictions
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()

# Step 5: Add the Predicted Column to the DataFrame
tt['PtmPL'] = predictions

# Now df contains the new column 'predicted_tmPL' with the predictions.

ttt = tt[['Date','tickerName', 'TmPL', 'PtmPL', 'gET','PL', 'buyPRTY']]

ttt = ttt.sort_values(by=['Date', 'PtmPL'], ascending=[True, False])
# tt = tt.sort_values(by=['Date', 'counts', 'ASM', 'buyPRTY', 'ETEXPL'], ascending=[True, False, False, False, False])
ttt = ttt.groupby('Date').head(1).reset_index(drop=True)
print(round(ttt[ttt['gET'] == 1]['PL'].sum(), 2))

ttt[ttt['gET'] == 1].tail(30)
tt.tail(30)

print(ttt.groupby(pd.to_datetime(ttt['Date']).dt.month) ['PL'].sum().reset_index())
# print(df1.groupby(pd.to_datetime(df1['Date']).dt.month) ['ProfitPercent'].sum().reset_index())


