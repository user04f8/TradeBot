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

def download_market(stocks, start_date, end_date, save_dir):
    stock_data = {}
    options_data = {}

    for stock in stocks:
        # Fetch historical stock data
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)
        
        # Fetch options data
        options_dates = ticker.options  # Get available options expirations
        options_frames = []
        for date in options_dates:
            opt = ticker.option_chain(date)
            combined_options = pd.concat([opt.calls, opt.puts], keys=['calls', 'puts'])
            combined_options['expiration_date'] = date
            options_frames.append(combined_options)
        
        if options_frames:
            options_data[stock] = pd.concat(options_frames)

    for stock, data in stock_data.items():
        print(f"Stock Data for {stock}:")
        print(data.head())

    for stock, data in options_data.items():
        if data.empty:
            print(f"No options data found for {stock}.")
        else:
            print(f"Options Data for {stock}:")
            print(data.head())  

    for stock, data in stock_data.items():
        data.to_csv(f".\\markets\\{save_dir}\\{stock}_stock.csv")
    for stock, data in options_data.items():
        data.to_csv(f".\\markets\\{save_dir}\\{stock}_options.csv")


if __name__ == '__main__':
    pass
    # download_market(['AAPL', 'MSFT', 'GOOGL'], '2024-01-01', '2024-03-01', 'val')
    # download_market(['AAPL', 'MSFT', 'GOOGL'], '2024-01-01', '2024-03-01', 'train')