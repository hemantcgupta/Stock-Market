# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:32:00 2024

@author: Hemant
"""
import os
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

class TmPrectionBagging:
    def __init__(self) -> None:
        self.scalerBagging = None
        self.modelBagging = None
        self.accuracyBagging = None
        self.matrix_report_bagging = None
    
    def modelTrainingBagging(self) -> None:
        dfM = self.df[pd.to_datetime(self.df['Date']) >= (self.df['Date'].iloc[0] - pd.DateOffset(months=1))].reset_index(drop=True)
        XM = dfM.drop(columns=['Date', 'TmDate', 'TmPL'])
        yM = dfM['TmPL']
        XM = XM.astype(float)
        
        X = self.df.drop(columns=['Date', 'TmDate', 'TmPL'])
        y = self.df['TmPL']
        X = X.astype(float)
        
        best_accuracy = 0
        best_model = None
        best_scaler = None
        report = None
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, y_test = XM, yM
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            # Initialize and train the BaggingClassifier
            base_model = SimpleNN(input_dim=X_train.shape[1], output_dim=len(y.unique()))
            model = BaggingClassifier(base_model, n_estimators=50)
            X_train_np = X_train
            y_train_np = y_train_tensor.numpy()
            model.fit(X_train_np, y_train_np)
            X_test_np = X_test
            y_test_np = y_test_tensor.numpy()
            y_pred = model.predict(X_test_np)
            score = accuracy_score(y_test_np, y_pred)
            accuracy = round(score * 100, 2)
            report = classification_report(y_test_np, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_scaler = scaler
                matrix_report = report
            
        self.modelBagging = best_model
        self.scalerBagging = best_scaler
        self.accuracyBagging = best_accuracy 
        self.matrix_report_bagging = matrix_report            
        # Save the model and scaler
        self.define_base_path('modelBagging')
        model_path = os.path.join(self.base_path, self.model_filename.format(self.date, self.accuracyBagging))
        scaler_path = os.path.join(self.base_path, self.scaler_filename.format(self.date, self.accuracyBagging))
        joblib.dump(self.modelBagging, model_path)
        joblib.dump(self.scalerBagging, scaler_path)

    def loadModelBagging(self) -> None:
        self.define_base_path('modelBagging')
        model_path = os.path.join(self.base_path, self.model_filename)
        scaler_path = os.path.join(self.base_path, self.scaler_filename)
        self.modelBagging = joblib.load(model_path)
        self.scalerBagging = joblib.load(scaler_path)
    
    def PredictionBagging(self) -> None:
        if self.modelBagging is None or self.scalerBagging is None:
            self.model_filename = self.get_latest_file_with_highest_percent('model')
            self.scaler_filename = self.get_latest_file_with_highest_percent('scaler')
            self.loadModelBagging()
        columns = ['BASR_Day', 'ATV_Day', 'LS_Day', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA']
        new_df = self.dfPred[columns]
        X_new = new_df.astype(float)
        X_new = self.scalerBagging.transform(X_new) 
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
        prediction = self.modelBagging.predict(X_new)
        self.dfPred['TmPredBaggingPL'] = prediction
       
        
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    
class BaggingClassifier:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_resampled = X[indices]
            y_resampled = y[indices]
            
            model = self.base_model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            model.train()
            for epoch in range(5):  # Adjust number of epochs as needed
                for inputs, labels in DataLoader(TensorDataset(torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.long)), batch_size=4, shuffle=True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.models.append(model)

    def predict(self, X):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(X, dtype=torch.float32))
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.numpy())
        
        # Aggregate predictions (majority voting)
        predictions = np.array(predictions)
        majority_vote = np.mean(predictions, axis=0)
        final_predictions = np.round(majority_vote).astype(int)
        return final_predictions
    
