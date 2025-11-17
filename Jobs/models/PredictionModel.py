# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:52:35 2024

@author: Hemant
"""
import warnings
warnings.filterwarnings('ignore')
import os
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
    
class StockPredictor:
    def __init__(self, kwargs, test_size=0.3, random_state=42, lr=0.001, num_epochs=1000):
        self.db_mkanalyzer = 'mkanalyzer'
        self.tickerName = kwargs.get('tickerName')
        self.inputDate = kwargs.get('Datetime')
        self.features = ['diffEntry', 'diffExit', 'diffHigh', 'LS_Day']
        self.target = 'TmPLShifted'
        self.test_size = test_size
        self.random_state = random_state
        self.lr = lr
        self.num_epochs = num_epochs
         
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
   

    def preprocess_data(self, df):
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
        y_remaining = df_remaining[self.target]
        
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
        
        df_remaining['TmPLPred'] = label_encoder.inverse_transform(predictions.numpy())
        
        first_row_scaled = scaler.transform(first_row[self.features].values.reshape(1, -1))
        first_row_tensor = torch.tensor(first_row_scaled, dtype=torch.float32)
        with torch.no_grad():
            first_row_output = model(first_row_tensor)
            _, first_row_prediction = torch.max(first_row_output, 1)
        first_row['TmPLPred'] = label_encoder.inverse_transform(first_row_prediction.numpy())[0]
        
        df_updated = pd.concat([pd.DataFrame([first_row]), df_remaining], ignore_index=True)
        modelAccuacy = round(accuracy * 100, 2)
        dfResult = pd.DataFrame([first_row])[['Datetime', 'tickerName', 'TmPLPred', 'ActualEntry', 'ActualExit', 'ActualHigh', 'PredEntry', 'PredExit', 'ActualProfit', 'PredProfit', 'diffEntry', 'diffExit', 'diffHigh', 'LS_Day']]
        dfResult['modelAccuacy'] = modelAccuacy
        return dfResult, modelAccuacy

    def run(self):
        df = self.fetchData()
        best_dfResult, best_accuracy = None, 0
        for i in range(5):
            dfResult, accuracy = self.preprocess_data(df)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_dfResult = dfResult
        return best_dfResult


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
        SELECT DISTINCT(tickerName) as tickerName FROM topPriorityStats WHERE Datetime = '{filterDate}'
        '''
        alredyPresent = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['tickerName'].tolist()
        df = df[~df['tickerName'].isin(alredyPresent)].reset_index(drop=True)
    except:
        pass
    tickerList = df.to_dict('records')
    with Pool(processes=int(cpu_count() * 0.7)) as pool:
        result = list(tqdm(pool.imap(run_stock_predictor, tickerList), total=len(tickerList), desc='Tm Prediction'))
    # result = [StockPredictor(kwargs).run() for kwargs in tqdm(tickerList[0:1], desc='Tm Prediction')]
    if result:
        df = pd.concat(result).reset_index(drop=True)
        df.apply(lambda row: Delete_max_date(VAR.db_mkanalyzer, 'topPriorityStats', row), axis = 1)
        result = Data_Inserting_Into_DB(df, VAR.db_mkanalyzer, 'topPriorityStats', 'append')
        return result
    return None


def jobMarketPredictor():
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
    result = jobMarketPredictor()