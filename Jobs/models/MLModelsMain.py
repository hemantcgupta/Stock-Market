# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:50:52 2024

@author: Hemant
"""
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Any
from Scripts.dbConnection import cnxn, Data_Inserting_Into_DB
from models.data_gathering import DATA_GATHERING 
from models.TmProbabilitySVC import TmPrectionSVC 
from models.TmProbabilityTorch import TmPrectionTorch 
from models.TmProbabilityBagging import TmPrectionBagging 

class VAR:
    db_mkanalyzer = 'mkanalyzer'
    
class Main(DATA_GATHERING, TmPrectionSVC, TmPrectionTorch, TmPrectionBagging):
    def __init__(self, kwargs: dict[str: Any]) -> None:
        super().__init__(kwargs)
        self.kwargs = kwargs 
        self.date = datetime.now().strftime('%Y-%m-%d')
        self.dfScore = None
        self.base_path = None
        self.model_filename = 'model_{}_{}.joblib'
        self.scaler_filename = 'scaler_{}_{}.joblib'
        

    def __str__(self) -> str:
        return f'Ticker({self.tickerName})'
    
    def dataPreprocess(self) -> None:
        df = self.dfPred.copy()
        df[['nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA']] =  df[['nCandleBelowOpen', 'pCandleAboveOpen', 'nCandle', 'pCandle', 'Hits44MA']].astype(int)
        df = df.rename(columns={'entry_exit': 'entryExitSuccess', 'entry_close': 'entryCloseSuccess', 'entry_loss': 'entryLossFailed'})
        df[['entryExitSuccess', 'entryCloseSuccess', 'entryLossFailed']] = df[['entryExitSuccess', 'entryCloseSuccess', 'entryLossFailed']].shift(1).fillna('NO')
        df['TmPredPL'] = df.apply(lambda row: mode([row[col] for col in ['TmPredSVCPL', 'TmPredTorchPL', 'TmPredBaggingPL']])[0][0], axis=1)
        df['TmStrength'] = df.apply(lambda row: mode([row[col] for col in ['TmPredSVCPL', 'TmPredTorchPL', 'TmPredBaggingPL']])[1][0], axis=1)
        df['tickerName'] = self.tickerName
        self.dfPred = df
        
    def TmScore(self) -> None:
        df = self.dfPred.copy().iloc[1:].reset_index(drop=True)
        df['MinDate'] = df['Date'] - pd.DateOffset(months=1)   
        MinDate = df.iloc[0]['MinDate']
        df = df[pd.to_datetime(df['Date']) >= MinDate].reset_index(drop=True)
        df = df[['Date', 'TmPL', 'TmPredSVCPL', 'TmPredTorchPL', 'TmPredBaggingPL', 'TmPredPL']]
        models = ['Mode', 'SVC', 'Torch', 'Bagging']
        modelsCol = ['TmPredPL', 'TmPredSVCPL', 'TmPredTorchPL', 'TmPredBaggingPL']
        DictFinal = {}
        for model, modelCol in zip(models, modelsCol):
            Dict = (df['TmPL'].astype(str)+df[modelCol].astype(str)).value_counts().to_dict()
            Dict['TotalDays'] = sum(Dict.values())
            Dict['00'] = Dict.get('00', 0)
            Dict['11'] = Dict.get('11', 0)
            Dict['01'] = Dict.get('01', 0)
            Dict['10'] = Dict.get('10', 0)
            Dict[f'AccuracyScore:{model}'] = round((Dict['11'] + Dict['00']) * 100 / Dict['TotalDays'], 2)
            renameColumns = {'00': f'AP:00:Prevent:{model}', '01': f'AP:01:Wrong:{model}', '10': f'AP:10:Missed:{model}', '11': f'AP:11:Correct:{model}'}
            Dict = {renameColumns[key] if key in renameColumns else key: value for key, value in Dict.items()}
            DictFinal = {**DictFinal, **Dict}
        DictFinal['accuracySVC'] = self.accuracySVC
        DictFinal['accuracyTorch'] = self.accuracyTorch
        DictFinal['accuracyBagging'] = self.accuracyBagging
        DictFinal['tickerName'] = self.tickerName
        DictFinal['Date'] = self.dfPred.iloc[0]['Date']
        df = pd.DataFrame([DictFinal]).rename(columns=renameColumns)
        df1 = self.dfPred.copy()
        df1 = df1[df1['Date'] == max(df1['Date'])][['tickerName', 'TmPredSVCPL', 'TmPredTorchPL', 'TmPredBaggingPL', 'TmPredPL', 'TmStrength']]
        df = pd.merge(df, df1, how='left', on='tickerName')
        df = df.fillna(0)
        df['successCounts'] = self.kwargs.get('counts')
        self.dfScore = df
        seqColumns = ['Date', 'tickerName', 'TmPredPL', 'TmPredSVCPL', 'TmPredTorchPL',
        'TmPredBaggingPL', 'TmStrength', 'accuracySVC', 'accuracyTorch', 'accuracyBagging', 
        'AccuracyScore:Mode', 'AccuracyScore:SVC', 'AccuracyScore:Torch', 'AccuracyScore:Bagging',
        'AP:00:Prevent:Mode', 'AP:01:Wrong:Mode', 'AP:10:Missed:Mode', 'AP:11:Correct:Mode', 
        'AP:00:Prevent:SVC', 'AP:01:Wrong:SVC', 'AP:10:Missed:SVC', 'AP:11:Correct:SVC',  
        'AP:00:Prevent:Torch', 'AP:01:Wrong:Torch', 'AP:10:Missed:Torch', 'AP:11:Correct:Torch',
        'AP:00:Prevent:Bagging','AP:01:Wrong:Bagging', 'AP:10:Missed:Bagging', 'AP:11:Correct:Bagging', 
        'successCounts', 'TotalDays']
        self.dfScore = self.dfScore[seqColumns]

def get_TmPrediction(kwargs):    
    obj = Main(kwargs)
    df = obj.query()
    obj.data_process(df)
    obj.modelTrainingSVC()
    obj.PredictionSVC()
    obj.modelTrainingTorch()
    obj.PredictionTorch()
    obj.modelTrainingBagging()
    obj.PredictionBagging()
    obj.dataPreprocess()
    obj.TmScore()
    obj.dfScore.columns
    return {'predData': obj.dfPred, 'predStats': obj.dfScore}

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
        SELECT DISTINCT(tickerName) as tickerName FROM topAccurateStats WHERE Date = '{filterDate}'
        '''
        alredyPresent = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['tickerName'].tolist()
        df = df[~df['tickerName'].isin(alredyPresent)].reset_index(drop=True)
    except:
        pass
    tickerList = df.to_dict('records')
    with Pool(processes=int(cpu_count() * 0.7)) as pool:
        result = list(tqdm(pool.imap(get_TmPrediction, tickerList), total=len(tickerList), desc='Tm Prediction'))
    # result = [get_TmPrediction(tickerName=ticker, filterDate=filterDate) for ticker in tqdm(tickerList[0:1], desc='Tm Prediction')]
    if result:
        df = pd.concat([value for dct in result for key, value in dct.items() if key == 'predData']).reset_index(drop=True)
        result1 = Data_Inserting_Into_DB(df, VAR.db_mkanalyzer, 'topAccurateTickerDetails', 'replace')
        df = pd.concat([value for dct in result for key, value in dct.items() if key == 'predStats']).reset_index(drop=True)
        df.apply(lambda row: Delete_max_date(VAR.db_mkanalyzer, 'topAccurateStats', row), axis = 1)
        result2 = Data_Inserting_Into_DB(df, VAR.db_mkanalyzer, 'topAccurateStats', 'append')
        return result1, result2
    return None, None

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
    
   
def jobPredictionModel():
    query = '''
    SELECT DISTINCT(Datetime) FROM simulationPrediction
    WHERE Datetime >= DATEADD(MONTH, -3, (SELECT MAX(Datetime) FROM simulationPrediction))
    ORDER BY Datetime ASC
    '''
    dateList = pd.read_sql(query, cnxn(VAR.db_mkanalyzer))['Datetime'].dt.date.astype(str).tolist()
    for date in dateList:
        print(f'{date:#^75}')
        result1, result2 = topAccurateTickers(top=20, filterDate=date)
    return 'Done!'
    
if __name__ == "__main__":
    result = jobPredictionModel()


