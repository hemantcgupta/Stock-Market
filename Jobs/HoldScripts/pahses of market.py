import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 80)
pd.set_option('expand_frame_repr', True)
import numpy as np
import yfinance as yf  # You need to install yfinance: pip install yfinance

def get_market_data(symbol, start_date, end_date):
    # Fetch historical market data from Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date, interval='5m')
    return data

def identify_market_phases(data):
    # Calculate some common technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)

    # Identify market phases
    data['Accumulation'] = np.where((data['Close'] > data['SMA_20']) & (data['Close'] > data['SMA_50']), 1, 0)
    data['Advancing'] = np.where((data['SMA_20'] > data['SMA_50']) & (data['Close'] > data['SMA_20']), 1, 0)
    data['Distribution'] = np.where((data['Close'] < data['SMA_20']) & (data['Close'] < data['SMA_50']), 1, 0)
    data['Declining'] = np.where((data['SMA_20'] > data['SMA_50']) & (data['Close'] < data['SMA_20']), 1, 0)

    return data

def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main(symbol, start_date, end_date):
    market_data = get_market_data(f'{symbol}.NS', start_date, end_date)
    market_data_with_phases = identify_market_phases(market_data)
    df = market_data_with_phases.tail(150)
    return {'tickerName': symbol, **{col: df.tail(75)[col].sum() for col in ['Accumulation', 'Advancing', 'Distribution', 'Declining']}}
       
if __name__ == "__main__":
    symbols = ['MOL', 'AJANTPHARM', 'BARBEQUE', 'JINDWORLD', 'AWL', 'GET&D',
           'GLS', 'HFCL', 'TIMETECHNO']
    start_date = '2024-08-20'
    end_date = '2024-08-23'
    df = pd.DataFrame([main(symbol, start_date, end_date) for symbol in symbols])