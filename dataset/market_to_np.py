import os
import numpy as np
import json

import yfinance as yf

from download_markets import get_valid_tickers

DATA_DIR = 'markets'

def download_market_to_numpy(stocks, start_date, end_date, save_dir):
    full_dir = os.path.join(DATA_DIR, save_dir)

    saved_tickers = []

    if not os.path.exists(save_dir):
        os.makedirs(full_dir, exist_ok=True)

    for stock in stocks:
        if '^' in stock:
            print(f'skipping ticker {stock}')
            continue
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start_date, end=end_date, interval='1h')
        if not data.empty:
            # Extract closing prices and times
            close_prices = data['Close'].values
            times = data.index.astype(str).values
            # Save to numpy file
            np.savez_compressed(os.path.join(full_dir, f"{stock}_prices.npz"), prices=close_prices, times=times)

            saved_tickers.append(stock)
        else:
            print(f"No data found for {stock}")

    with open(os.path.join(full_dir, 'metadata.json'), 'w') as f:
        json.dump(saved_tickers, f) 

stocks = get_valid_tickers()

saved = download_market_to_numpy(stocks, '2023-01-01', '2024-01-01', 'train')

