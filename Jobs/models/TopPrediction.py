# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:52:35 2024

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

class MarketAnalyzer:
    def __init__(self, kwargs):
        self.cnxn = cnxn(VAR.db_mkintervalmaster)
        self.Date = kwargs.get('Date')
        self.symbol = kwargs.get('tickerName')
        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')
        self.market_data = None

    def get_market_data(self):
        query = f"select * from [{self.symbol}] where Datetime <= '{self.end_date}' "
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
        return market_data_with_phases
   
        
class StockPredictor:
    def __init__(self, kwargs, test_size=0.3, random_state=42, lr=0.001, num_epochs=1000):
        self.db_mkanalyzer = 'mkanalyzer'
        self.tickerName = kwargs.get('tickerName')
        self.inputDate = kwargs.get('Datetime')
        self.successCount = kwargs.get('counts')
        self.features = ['diffEntry', 'diffExit', 'diffHigh', 'LS_Day', 'pMomentum', 'nMomentum', 'buySignal', 'sellSignal', 'holdingSignal']
        self.target = 'TmPLShifted'
        self.test_size = test_size
        self.random_state = random_state
        self.lr = lr
        self.num_epochs = num_epochs
        self.dfPhases = None
         
    def fetchData(self):
        query = f'''
        WITH cte AS (
            SELECT CAST(predDatetime AS DATE) AS Datetime, 
                   tickerName, 
                   Entry2 AS ActualEntry, 
                   Exit2 AS ActualExit, 
                   [High] AS ActualHigh,
                   predTmEntry2 AS PredEntry, 
                   predTmExit2 AS PredExit, 
                   [Close],
                   CASE 
                       WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
                       THEN 1 
                       ELSE 0 
                   END AS TmPL,
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
                       WHEN [High] >= predTmExit2 
                       THEN 1
                       ELSE 0 
                   END AS gotSell,
                   CASE 
                       WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
                       THEN ROUND(((predTmExit2 - predTmEntry2) / predTmExit2) * 100, 2)
                       ELSE ROUND((([Close] - predTmEntry2) / [Close]) * 100, 2)
                   END AS ActualProfit, 
                   EtEx2Profit AS PredProfit
            FROM simulationPrediction 
            WHERE tickerName = '{self.tickerName}' 
              AND predDatetime <= '{self.inputDate}'
        ),
        LSDAY AS (
            SELECT CAST(Datetime AS DATE) AS Datetime,
                   ROUND(CASE
                       WHEN MAX([Close]) = 0 OR SUM(Volume) = 0 THEN NULL
                       ELSE ((MAX([High]) - MIN([Low])) / MAX([Close])) * LOG(SUM(Volume) / COUNT(DISTINCT CONVERT(date, Datetime)))
                   END, 2) AS LS_Day
            FROM mkdaymaster.dbo.[{self.tickerName}]
            GROUP BY Datetime, [High], [Low], [Close], [Volume]
        )
        SELECT c.*,  
               ROUND((ActualEntry - PredEntry) / PredEntry * 100, 2) AS diffEntry,
               ROUND((ActualExit - PredExit) / PredExit * 100, 2) AS diffExit,
               ROUND((ActualHigh - PredExit) / PredExit * 100, 2) AS diffHigh,
               LAG(CASE WHEN gotSell = 1 THEN ROUND(([Close] - PredExit) / PredExit * 100, 2) ELSE 0 END, 1) 
                   OVER (ORDER BY c.Datetime DESC) AS diffClose,
               LAG(gotSell, 1) OVER (ORDER BY c.Datetime DESC) AS gotSellShifted,
               LAG(TmPL, 1) OVER (ORDER BY c.Datetime DESC) AS TmPLShifted,
               LAG(gotLoss, 1) OVER (ORDER BY c.Datetime DESC) AS gotLossShifted,
               LAG(gotEntry, 1) OVER (ORDER BY c.Datetime DESC) AS gotEntryShifted,
               LAG(ActualProfit, 1) OVER (ORDER BY c.Datetime DESC) AS ActualProfitShifted,
               ld.LS_Day,
               LAG(
                   CASE 
                       WHEN (TmPL = 1 AND ActualProfit = PredProfit) THEN 'ETEX' 
                       WHEN (TmPL = 1 AND ActualProfit != PredProfit) THEN 'ETCL' 
                       WHEN (gotEntry = 1 AND gotLoss = 1) THEN 'ETLS'
                       ELSE 'NOET' 
                   END, 1
               ) OVER (ORDER BY c.Datetime DESC) AS TmPredShifted
        FROM cte c
        LEFT JOIN LSDAY ld ON ld.Datetime = c.Datetime
        ORDER BY c.Datetime DESC;
        '''
        return pd.read_sql(query, cnxn(self.db_mkanalyzer))
    
    def getMarketPhases(self):
        tickerDetails = {
            'tickerName': self.tickerName, 
            'end_date': (pd.to_datetime(self.inputDate) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        }
        dfPhases = MarketAnalyzer(tickerDetails).analyze()
        dfPhases['tickerName'] = self.tickerName
        dfPhases = dfPhases.sort_values(by='Datetime').reset_index(drop=True)
        dfPhases['Date'] = pd.to_datetime(dfPhases['Datetime']).dt.date
        dfPhases = dfPhases.groupby('Date').agg(
            tickerName=('tickerName', lambda x: x.unique()[0]),
            Close315=('Open', lambda x: x.iloc[-3] if len(x) >= 3 else x.iloc[0]),
            Accumulation=('Accumulation', 'sum'),
            Advancing=('Advancing', 'sum'),
            Distribution=('Distribution', 'sum'),
            Declining=('Declining', 'sum'),
            RSI=('RSI', lambda x: {
                'buySignal': len([item for item in x if item < 30]),
                'sellSignal': len([item for item in x if item > 70]),
                'holdingSignal': len([item for item in x if item >= 30 and item <= 70])
            })
        ).reset_index()
        dfPhases['pMomentum'] = (((dfPhases['Accumulation']+dfPhases['Advancing'])/(dfPhases['Accumulation']+dfPhases['Advancing']+dfPhases['Distribution']+dfPhases['Declining']))*100).round(2)
        dfPhases['nMomentum'] = (((dfPhases['Distribution']+dfPhases['Declining'])/(dfPhases['Accumulation']+dfPhases['Advancing']+dfPhases['Distribution']+dfPhases['Declining']))*100).round(2)
        dfPhases = pd.concat([dfPhases.drop(columns=['RSI']), dfPhases['RSI'].apply(pd.Series)], axis=1)
        self.dfPhases = dfPhases

    def preprocess_data(self, df, target):
        first_row = df.iloc[0]
        df_remaining = df.iloc[1:]
        df_remaining = df_remaining.dropna()
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
    
        X_remaining = df_remaining[self.features]
        y_remaining = df_remaining[target]
        
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_remaining, y_remaining)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_resampled)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=self.test_size, random_state=self.random_state)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        batch_size = 64
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = len(label_encoder.classes_)
        model = SimpleNN(input_size, hidden_size, output_size)
        
        class_counts = torch.bincount(y_train_tensor)
        total_samples = len(y_train_tensor)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights.to(torch.float32)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        epochs = self.num_epochs
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            epoch_loss = running_loss / len(train_dataset)
            # print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            # print(f'Accuracy on Test Data: {accuracy * 100:.2f}%')
        
        df_remaining_scaled = scaler.transform(df_remaining[self.features])
        X_full_tensor = torch.tensor(df_remaining_scaled, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(X_full_tensor)
            _, predictions = torch.max(outputs, 1)
        
        df_remaining['TmPredPL'] = label_encoder.inverse_transform(predictions.numpy())
        
        first_row_scaled = scaler.transform(first_row[self.features].values.reshape(1, -1))
        first_row_tensor = torch.tensor(first_row_scaled, dtype=torch.float32)
        with torch.no_grad():
            first_row_output = model(first_row_tensor)
            _, first_row_prediction = torch.max(first_row_output, 1)
        first_row['TmPredPL'] = label_encoder.inverse_transform(first_row_prediction.numpy())[0]
        
        df_updated = pd.concat([pd.DataFrame([first_row]), df_remaining], ignore_index=True)
        modelAccuracy = round(accuracy * 100, 2)
        epochLoss = round(epoch_loss * 100, 2)
        phasesColumns = [item for item in self.dfPhases.columns if item not in ['Date', 'tickerName']]
        dfResult = pd.DataFrame([first_row])
        dfResult['successCount'] = self.successCount
        dfResult['modelAccuracy'] = modelAccuracy
        dfResult['epochLoss'] = epochLoss
        dfResult = dfResult[['Datetime', 'tickerName', 'TmPredPL', 'ActualEntry', 'ActualExit', 'ActualHigh', 'PredEntry', 'PredExit', 'ActualProfit', 'PredProfit', 'diffEntry', 'diffExit', 'diffHigh', 'LS_Day', *phasesColumns, 'successCount', 'modelAccuracy', 'epochLoss']]
        dfResult['TmPredPL'] = dfResult['TmPredPL'].astype(int)
        return dfResult, modelAccuracy, epochLoss
    
    @staticmethod
    def finalProcess(resultDict):
        df1 = resultDict.get('TmPLShifted').get('best_dfResult')
        df1['TmPredPL5Summary'] = ', '.join(resultDict.get('TmPLShifted').get('lstResult'))
        df1.rename(columns={'modelAccuracy': 'modelAccuracyPL', 'epochLoss': 'epochLossPL'}, inplace=True)
        df2 = resultDict.get('gotLossShifted').get('best_dfResult')
        df2['TmPredgotLoss5Summary'] = ', '.join(resultDict.get('gotLossShifted').get('lstResult'))
        df2.rename(columns={'TmPredPL': 'TmPredgotLoss', 'modelAccuracy': 'modelAccuracygotLoss', 'epochLoss': 'epochLossgotLoss'}, inplace=True)
        dfFinal = pd.merge(df1, df2[['Datetime', 'tickerName', 'TmPredgotLoss', 'modelAccuracygotLoss', 'epochLossgotLoss', 'TmPredgotLoss5Summary']], how='left', on=['Datetime', 'tickerName'])
        dfFinal = dfFinal[['Datetime', 'tickerName', 'TmPredPL', 'TmPredgotLoss', 'PredEntry', 'PredExit', 'ActualProfit', 'PredProfit', 'ActualEntry', 'ActualExit', 'ActualHigh', 'diffEntry', 'diffExit', 'diffHigh', 'LS_Day', 'Accumulation', 'Advancing', 'Distribution', 'Declining', 'pMomentum', 'nMomentum', 'buySignal', 'sellSignal', 'holdingSignal', 'successCount', 'modelAccuracyPL', 'epochLossPL', 'modelAccuracygotLoss', 'epochLossgotLoss', 'TmPredPL5Summary', 'TmPredgotLoss5Summary']]
        return dfFinal
    
    def run(self):
        if not self.dfPhases:
            self.getMarketPhases()
        df = self.fetchData()
        df = df.merge(self.dfPhases.drop(columns=['tickerName']), how='left', left_on='Datetime', right_on='Date')
        resultDict = {}
        for target in ['TmPLShifted', 'gotLossShifted']:
            best_dfResult, best_accuracy, best_epochLoss = None, 0, 100
            lstResult = []
            for i in range(5):
                dfResult, accuracy, epochLoss = self.preprocess_data(df, target)
                lstResult.append([f'{pred} :: {accuracy} :: {loss}' for pred, accuracy, loss in zip(dfResult['TmPredPL'], dfResult['modelAccuracy'], dfResult['epochLoss'])][0])
                if epochLoss < best_epochLoss:
                    best_accuracy = accuracy
                    best_epochLoss = epochLoss
                    best_dfResult = dfResult
            resultDict[target] = {'best_dfResult': best_dfResult, 'lstResult': lstResult}
        dfFinal = self.finalProcess(resultDict)
        return dfFinal
 
    
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
    
def run_stock_predictor(kwargs):
    return StockPredictor(kwargs).run()
    
def topAccurateTickers(top=None, filterDate=None): 
    query = f'''
    SELECT sp.tickerName, COUNT(sp.Datetime) AS counts, '{filterDate}' as Datetime, ETEXProfit
    FROM simulationPrediction AS sp
    LEFT JOIN (SELECT tickerName, EtEx2Profit as ETEXProfit FROM simulationPrediction WHERE Datetime='{filterDate}') p ON p.tickerName=sp.tickerName
    WHERE Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2
    AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, '{filterDate}') AS DATE)
    AND CAST(Datetime AS DATE) < CAST('{filterDate}' AS DATE)
    GROUP BY sp.tickerName, ETEXProfit
    ORDER BY counts DESC, ETEXProfit DESC
    '''
    df = pd.read_sql(query, cnxn(VAR.db_mkanalyzer)).iloc[:top]
    try:
        query = f'''
        SELECT DISTINCT(tickerName) as tickerName FROM mkTopPrediction WHERE Datetime = '{filterDate}'
        '''
        alredyPresent = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['tickerName'].tolist()
        df = df[~df['tickerName'].isin(alredyPresent)].reset_index(drop=True)
    except:
        pass
    tickerList = df.to_dict('records')
    with Pool(processes=int(cpu_count() * 0.7)) as pool:
        result = list(tqdm(pool.imap(run_stock_predictor, tickerList), total=len(tickerList), desc='Top Prediction'))
    # result = [run_stock_predictor(kwargs) for kwargs in tqdm(tickerList, desc='Top Prediction')]
    if result:
        df = pd.concat(result).reset_index(drop=True)
        df.apply(lambda row: Delete_max_date(VAR.db_mkanalyzer, 'mkTopPrediction', row), axis = 1)
        result = Data_Inserting_Into_DB(df, VAR.db_mkanalyzer, 'mkTopPrediction', 'append')
        return result
    return None


def MkTopPrediction():
    query = '''
    SELECT DISTINCT(Datetime) FROM simulationPrediction
    WHERE Datetime >= DATEADD(MONTH, -3, (SELECT MAX(Datetime) FROM simulationPrediction))
    ORDER BY Datetime ASC
    '''
    dateList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Datetime'].dt.date.astype(str).tolist()
    for date in dateList:
        print(f'{date:#^75}')
        result = topAccurateTickers(top=20, filterDate=date)
    return result
    
if __name__ == "__main__":
    result = MkTopPrediction()
    
# ['^NSEBANK', '^NSEI', 'NIFTY_FIN_SERVICE', '^BSESN', 'BLS']    
