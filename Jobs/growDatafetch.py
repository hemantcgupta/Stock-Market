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

# # Example Usage:
# date_string = "2024-10-08 17:26:29"  # Example date
# timestamp_in_ms = 1728388589000  # Example timestamp in milliseconds

# # Convert date to milliseconds
# milliseconds = date_to_milliseconds(date_string)
# print(f"Date to milliseconds: {milliseconds}")

# # Convert milliseconds back to date
# date = milliseconds_to_date(timestamp_in_ms)
# print(f"Milliseconds to date: {date}")


# =============================================================================
# 
# =============================================================================
import requests
import pandas as pd
import pytz
def fetch_stock_data(tickerName, startTimeInMillis, endTimeInMillis, intervalInMinutes):
    url = f"https://groww.in/v1/api/charting_service/v2/chart/delayed/exchange/NSE/segment/CASH/{tickerName}"
    startTimeInMillis = date_to_milliseconds(startTimeInMillis, date_format="%Y-%m-%d %H:%M:%S")
    endTimeInMillis = date_to_milliseconds(endTimeInMillis, date_format="%Y-%m-%d %H:%M:%S")
    params = {
        "endTimeInMillis": endTimeInMillis,
        "intervalInMinutes": intervalInMinutes,
        "startTimeInMillis": startTimeInMillis
    }
    try:
        response = requests.get(url, params=params)
        data = response.json() 
        if 'candles' in data:
            dataRow = data.get('candles', [])
            df = pd.DataFrame(dataRow, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            timezone = pytz.timezone('Asia/Kolkata')
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s', utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df.drop_duplicates().reset_index(drop=True)
        else:
            df = pd.DataFrame()
    except Exception as e:
        print("An error occurred:", e)
        df = pd.DataFrame()  # Return an empty DataFrame on error
    return df
    
tickerName = 'TATAMOTORS'
startTimeInMillis = '2025-02-01 00:00:00'
endTimeInMillis = '2025-05-25 23:59:59'
intervalInMinutes = 5
data = fetch_stock_data(tickerName, startTimeInMillis, endTimeInMillis, intervalInMinutes)




# =============================================================================
# 
# =============================================================================
import requests

totalRecords = 0
# Define the URL and payload
url = "https://groww.in/v1/api/stocks_data/v1/all_stocks"
payload = {
    "listFilters": {
        "INDUSTRY": [],
        "INDEX": []
    },
    "objFilters": {
        "CLOSE_PRICE": {"max": 50000000000, "min": 0},
        "MARKET_CAP": {"min": 0, "max": 3000000000000000}
    },
    "page": "0",
    "size": "1000",
    "sortBy": "NA",
    "sortType": "ASC"
}

# Define headers (if needed)
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Send the POST request
try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    data = response.json()  # Parse the JSON response
    print("Response Data:", data)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)

timezone = pytz.timezone('Asia/Kolkata')
if 'records' in data:
    dataRow = data.get('records', [])
    df = pd.DataFrame(dataRow)

