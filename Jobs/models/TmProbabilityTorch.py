# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:03:58 2024

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


class TmPrectionTorch:
    def __init__(self) -> None:
        self.scalerTorch = None
        self.modelTorch = None
        self.accuracyTorch = None
    
    def modelTrainingTorch(self) -> None:
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
            # Hyperparameters
            input_dim = X_train.shape[1]
            output_dim = len(y.unique())
            learning_rate = 0.001
            num_epochs = 20
            # Initialize the model, loss function, and optimizer
            model = SimpleNN(input_dim=input_dim, output_dim=output_dim)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            # Evaluation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = round((100 * correct / total), 2)
                
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_scaler = scaler
                
        self.modelTorch = best_model
        self.scalerTorch = best_scaler
        self.accuracyTorch = best_accuracy                
        # Save the model and scaler
        self.define_base_path('modelTorch')
        model_path = os.path.join(self.base_path, self.model_filename.format(self.date, self.accuracyTorch))
        scaler_path = os.path.join(self.base_path, self.scaler_filename.format(self.date, self.accuracyTorch))
        joblib.dump(self.modelTorch, model_path)
        joblib.dump(self.scalerTorch, scaler_path)
    
    def loadModelTorch(self) -> None:
        self.define_base_path('modelTorch')
        model_path = os.path.join(self.base_path, self.model_filename)
        scaler_path = os.path.join(self.base_path, self.scaler_filename)
        self.modelTorch = joblib.load(model_path)
        self.scalerTorch = joblib.load(scaler_path)
    
    def PredictionTorch(self) -> None:
        if self.modelTorch is None or self.scalerTorch is None:
            self.model_filename = self.get_latest_file_with_highest_percent('model')
            self.scaler_filename = self.get_latest_file_with_highest_percent('scaler')
            self.loadModelTorch()
        columns = ['BASR_Day', 'ATV_Day', 'LS_Day', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA']
        new_df = self.dfPred[columns]
        X_new = new_df.astype(float)
        X_new = self.scalerTorch.transform(X_new) 
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
        # Make prediction
        self.modelTorch.eval()
        with torch.no_grad():
            output = self.modelTorch(X_new_tensor)
            _, prediction = torch.max(output, 1)
            self.dfPred['TmPredTorchPL'] = prediction
        
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
        


