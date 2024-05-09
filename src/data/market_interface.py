import csv
from pathlib import Path
import os
import numpy as np 


class MarketData:
    def __init__(self, dir = 'dataset/markets/train'):
        self.dir = dir

    def load_prices_at_datetime(self, stock, query_datetime, window_size: int):
        # Load data from .npz file
        data_path = os.path.join(self.dir, f"{stock}_prices.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No saved data for stock {stock}")
        
        data = np.load(data_path, allow_pickle=True)
        prices = data['prices']
        times = data['times']
        
        # Convert query_datetime to string in the same format as saved times
        query_str = query_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Find the index of the closest time
        idx = np.argmin(np.abs(np.array(times, dtype='datetime64') - np.datetime64(query_str)))
        times = times[idx-window_size:idx]
        prices = prices[idx-window_size:idx]

        return prices

    # def get_options_chain(self, date):
        # If BU finally buys and makes access available to reasonable data (e.g. Bloomberg terminal for non-Questrom students) or I'm willing to pay $2k,
        # this could be so much more powerful of a model, allowing for finetuning on options chain data for intraday pricing at each timestep.
    