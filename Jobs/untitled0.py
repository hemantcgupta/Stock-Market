import yfinance as yf
import pandas as pd
from nsetools import Nse

# Initialize NSE object
nse = Nse()

# Fetch all stock tickers from NSE
all_tickers = nse.get_stock_codes()
stock_tickers = [ticker + ".NS" for ticker in all_tickers.keys() if ticker != 'SYMBOL']

print(f"Found {len(stock_tickers)} tickers. Fetching data in a single batch...")
stock_tickers = ['BLS', 'NHPC']
# Fetch data for all tickers in one batch
data = yf.download(tickers=" ".join(stock_tickers), period="1d", interval="5m", group_by="ticker")

# Save the data to an Excel file
data.reset_index(inplace=True)
file_name = "All_Indian_Stocks_Live_Data.xlsx"
data.to_excel(file_name)

print(f"Data saved to {file_name}")



def yf_download(self, ticker_name, period):
    if ticker_name[0] == '^':
        ticker_name = f'{ticker_name}' 
    else:
        ticker_name = f'{ticker_name}.NS'
    yf_ticker = yf.Ticker(" ".join(stock_tickers))
    df_interval = yf_ticker.history(start=period['startInterval'], end=period['endInterval'], interval='5m').dropna().reset_index()
    df_interval = self.yf_download_cleaning(df_interval)
    return df_interval

df = data.copy()

def transformData(df):
    df = df.stack(level=0).reset_index()
    df.columns = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    desired_columns = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    df = df[desired_columns]
    df.rename(columns={'Ticker': 'tickerName'}, inplace=True)
    return df
