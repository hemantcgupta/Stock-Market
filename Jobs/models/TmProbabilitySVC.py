# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:06:19 2024

@author: Hemant
"""
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

class TmPrectionSVC:
    def __init__(self) -> None:
        self.scalerSVC = None
        self.best_params = None
        self.best_model = None
        self.accuracySVC = None
        self.matrix_report = None
        
    def modelTrainingSVC(self) -> None:
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
        best_params = None
        matrix_report = None
        for i in range(3):    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, y_test = XM, yM
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            param_grid = {
                'C': [0.1, 1, 10, 100, 150, 200],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
            grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            params = grid_search.best_params_
            model = grid_search.best_estimator_
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            accuracy = round(score * 100, 2)
            report = classification_report(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_scaler = scaler
                best_params = params
                matrix_report = report
        
        self.best_model = best_model
        self.scalerSVC = best_scaler
        self.accuracySVC = best_accuracy
        self.best_params = best_params
        self.matrix_report = matrix_report
        # Save the model and scaler
        self.define_base_path('modelSVC')
        model_path = os.path.join(self.base_path, self.model_filename.format(self.date, self.accuracySVC))
        scaler_path = os.path.join(self.base_path, self.scaler_filename.format(self.date, self.accuracySVC))
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scalerSVC, scaler_path)
    
    def loadModelSVC(self) -> None:
        self.define_base_path('modelSVC')
        model_path = os.path.join(self.base_path, self.model_filename)
        scaler_path = os.path.join(self.base_path, self.scaler_filename)
        self.best_model = joblib.load(model_path)
        self.scalerSVC = joblib.load(scaler_path)
        
    def PredictionSVC(self) -> None:
        if self.best_model is None or self.scalerSVC is None:
            self.model_filename = self.get_latest_file_with_highest_percent('model')
            self.scaler_filename = self.get_latest_file_with_highest_percent('scaler')
            self.loadModelSVC()
        columns = ['BASR_Day', 'ATV_Day', 'LS_Day', 'nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA']
        new_df = self.dfPred[columns]
        X_new = new_df.astype(float)
        X_new = self.scalerSVC.transform(X_new) 
        self.dfPred['TmPredSVCPL'] = self.best_model.predict(X_new)
        

