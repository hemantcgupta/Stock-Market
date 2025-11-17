import requests
import pandas as pd
import pytz
from Scripts.dbConnection import Data_Inserting_Into_DB, create_database, cnxn


class VAR:
    db_name_mk_groww_info = 'mkgrowwinfo'
    table_name_mkGrowwInfo = 'mkGrowwInfo'
    
def growwInfo():
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
        "page": "0",  # Start with page 0
        "size": "1000",  # Page size
        "sortBy": "NA",
        "sortType": "ASC"
    }
    
    # Define headers (if needed)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    all_records = []  # List to store all records
    
    try:
        # Send the initial POST request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        data = response.json()  # Parse the JSON response
    
        # Extract totalRecords and calculate total pages
        total_records = data.get("totalRecords", 0)
        page_size = 1000
        total_pages = (total_records + page_size - 1) // page_size  # Calculate the total pages
    
        print(f"Total Records: {total_records}, Total Pages: {total_pages}")
    
        # Iterate through all pages
        for page in range(total_pages):
            payload["page"] = str(page)  # Update the page number in the payload
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            page_data = response.json()
            
            # Append the records from this page
            if "records" in page_data:
                all_records.extend(page_data["records"])
    
        # Create a DataFrame from all records
        if all_records:
            df = pd.DataFrame(all_records)
            if 'livePriceDto' in df.columns:
                normalized_livePrice = pd.json_normalize(df['livePriceDto'], sep='_')
                normalized_livePrice.columns = [f'livePriceDto_{col}' for col in normalized_livePrice.columns]
                df = pd.concat([df.drop(columns=['livePriceDto']), normalized_livePrice], axis=1)
                timezone = pytz.timezone('Asia/Kolkata')
                df['livePriceDto_tsInMillis'] = pd.to_datetime(df['livePriceDto_tsInMillis'], unit='s', utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
                df['livePriceDto_lastTradeTime'] = pd.to_datetime(df['livePriceDto_lastTradeTime'], unit='s', utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
                float_cols = df.select_dtypes(include=['float']).columns
                df[float_cols] = df[float_cols].round(2)
        else:
            df = pd.DataFrame()  # Empty DataFrame if no records are found
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
        df = pd.DataFrame()  # Return an empty DataFrame on error
    return df

def JobGrowwInfoDataDownloader():
    create_database(VAR.db_name_mk_groww_info)
    df = growwInfo()
    if not df.empty:
       result = Data_Inserting_Into_DB(df, VAR.db_name_mk_groww_info, VAR.table_name_mkGrowwInfo, 'replace')
    return result
    
if __name__ == "__main__":
    result = JobGrowwInfoDataDownloader()


