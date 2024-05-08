import csv
import pandas as pd
from typing import List

import yfinance as yf

def get_valid_tickers():
    symbols: List[str] = []

    with open('markets/nasdaq_screener.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for row in csv_reader:
            symbols.append(row[0])

    return symbols

def get_ticker_name_dict():
    symbols_names: dict = {}

    with open('markets/nasdaq_screener.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for row in csv_reader:
            symbols_names[row[0]] = row[1]

    return symbols_names

def get_name_ticker_dict():
    symbols_names: dict = {}

    with open('markets/nasdaq_screener.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for row in csv_reader:
            symbols_names[row[1]] = row[0]

    return symbols_names

def download_market(stocks, start_date, end_date, save_dir):
    stock_data = {}

    for stock in stocks:
        # Fetch historical stock data
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)

    for stock, data in stock_data.items():
        print(f"Stock Data for {stock}:")
        print(data.head())

    # Since historical options data is not available via yfinance, this is not usable for backtesting.
    # However, the below code was and can be used to test on live options data.

    # for stock, data in options_data.items():
    #     if data.empty:
    #         print(f"No options data found for {stock}.")
    #     else:
    #         print(f"Options Data for {stock}:")
    #         print(data.head())  

    for stock, data in stock_data.items():
        data.to_csv(f".\\markets\\{save_dir}\\{stock}_stock.csv")
    # for stock, data in options_data.items():
    #     data.to_csv(f".\\markets\\{save_dir}\\{stock}_options.csv")


if __name__ == '__main__':
    download_market(['AAPL', 'MSFT', 'GOOGL'], '2024-01-01', '2024-03-01', 'val')
    # download_market(['AAPL', 'MSFT', 'GOOGL'], '2024-01-01', '2024-03-01', 'train')