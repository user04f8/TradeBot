import json
import numpy as np

from src.utils import get_ticker_name_dict, quantize_to_option_strikes
from src.ibkr.bot import IB_Interface
from src.data.news_interface import NewsInterface
from src.snram.live_aggregator import SNRAMLive


with open('dataset/markets/train/metadata.json') as f:
    tickers = json.load(f)
ticker_names = get_ticker_name_dict()

ni = NewsInterface()
snram = SNRAMLive()


print('Connecting to IBKR')
ib_int = IB_Interface()
try:
    print('TraderBot is now running!')
    while True:
        ticker = tickers[np.random(len(tickers))]
        ticker_news = ni.get_news(ticker)

        news_summary = snram.summarize_many(ticker_news, ticker)

        

        ib_int.ib.sleep(60)
finally:
    ib_int.disconnect()