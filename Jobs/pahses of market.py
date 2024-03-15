import pandas as pd
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

if __name__ == "__main__":
    # Example usage
    symbol = 'KFINTECH.NS'  # Example stock symbol
    start_date = '2024-01-01'
    end_date = '2024-02-20'

    # Fetch market data
    market_data = get_market_data(symbol, start_date, end_date)

    # Identify market phases
    market_data_with_phases = identify_market_phases(market_data)

    # Print or further analyze the market data with identified phases
    print(market_data_with_phases)
    
    
    tt = market_data_with_phases.tail(75)
    tt.iloc[55:].head(20)
