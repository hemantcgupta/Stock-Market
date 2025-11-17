# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:55:48 2024

@author: Hemant
"""
from datetime import datetime

# Function to convert a normal date to Unix timestamp in milliseconds
def date_to_milliseconds(date_string, date_format="%Y-%m-%d %H:%M:%S"):
    # Convert the string date to a datetime object
    dt = datetime.strptime(date_string, date_format)
    # Get the Unix timestamp in seconds and convert to milliseconds
    timestamp_in_ms = int(dt.timestamp() * 1000)
    return timestamp_in_ms

# Function to convert Unix timestamp in milliseconds to a normal date
def milliseconds_to_date(timestamp_in_ms, date_format="%Y-%m-%d %H:%M:%S"):
    # Convert milliseconds to seconds by dividing by 1000
    timestamp_in_seconds = timestamp_in_ms / 1000
    # Convert the timestamp to a datetime object
    dt = datetime.fromtimestamp(timestamp_in_seconds)
    # Return the formatted date string
    return dt.strftime(date_format)

# Example Usage:
date_string = "2024-10-08 17:26:29"  # Example date
timestamp_in_ms = 1728388589000  # Example timestamp in milliseconds

# Convert date to milliseconds
milliseconds = date_to_milliseconds(date_string)
print(f"Date to milliseconds: {milliseconds}")

# Convert milliseconds back to date
date = milliseconds_to_date(timestamp_in_ms)
print(f"Milliseconds to date: {date}")


# =============================================================================
# 
# =============================================================================
import requests
import pandas as pd
import pytz
def fetch_stock_data():
    url = "https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/BLS"
    params = {
        "endTimeInMillis": 1739385000,
        "intervalInMinutes": 5,
        "startTimeInMillis": 1728388589
    }
    try:
        response = requests.get(url, params=params)
        url = 'https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/BLS?endTimeInMillis=1736337389000&intervalInMinutes=5&startTimeInMillis=1728388589'
        response = requests.get(url)
        data = response.json() 
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
data = fetch_stock_data()
timezone = pytz.timezone('Asia/Kolkata')
if data:
    dataRow = data['candles']
    df = pd.DataFrame(dataRow, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s', utc=True).dt.tz_convert(timezone)
    df = df.drop_duplicates().reset_index(drop=True)
    df[df['Datetime'] > '2024-10-08']


df['Datetime'].dt.date.astype(str).nunique()


# =============================================================================
# 
# =============================================================================


import requests

def fetch_option_chain(expiry_date):
    url = "https://groww.in/v1/api/option_chain_service/v1/option_chain/nifty"
    
    # Parameters for the request
    params = {
        "expiry": expiry_date
    }
    try:
        # Making the GET request
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Convert the response to JSON
            return data
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
expiry_date = "2024-10-10"  # Set the expiry date
option_chain_data = fetch_option_chain(expiry_date)
if option_chain_data:
    print(option_chain_data)


# =============================================================================
# 
# =============================================================================
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22)
pd.set_option('expand_frame_repr', True)

import warnings
warnings.filterwarnings('ignore')                   
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn
from tqdm import tqdm


url = 'https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/NIFTY/daily?intervalInMinutes=75&minimal=true'
url = 'https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/BANKNIFTY/daily?intervalInMinutes=75&minimal=true'
url='https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/FINNIFTY/daily?intervalInMinutes=75&minimal=true'