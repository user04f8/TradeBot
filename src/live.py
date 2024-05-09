from datetime import datetime, timedelta
import json
import numpy as np

from constants import N_TRIES, DEBUG
from src.utils import get_ticker_name_dict, quantize_to_option_strikes
from src.ibkr.bot import IB_Interface, NEXT_OPTIONS_EXPIRY_DATETIME
from src.data.news_interface import NewsInterface
from src.snram.live_aggregator import SNRAMLive
from src.models.make_preds import make_pred


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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
        assert (NEXT_OPTIONS_EXPIRY_DATETIME > datetime.now())
        overall_penalty_min = 1e9
        strategy = None
        for i in range(N_TRIES):
            try:
                # stochastically guess potential tickers to obtain news data from
                ticker = tickers[np.random.randint(0, len(tickers))]
                ticker_news = ni.get_news(ticker, n=3)
                if len(ticker_news) == 0:
                    continue  # nothing interesting is happening with this particular stock so we move on 
                # ticker_news += ni.get_news(ticker_names[ticker], n=3)

                debug_print(ticker_news)

                stock_history = ib_int.get_local_hourly(ticker, 336)

                news_summary = snram.summarize_many(ticker_news, ticker)

                debug_print(stock_history)
                debug_print(news_summary)


                pred_prices, pred_prices_median = make_pred(stock_history, len_out=(24*(NEXT_OPTIONS_EXPIRY_DATETIME - datetime.now()).days))

                debug_print(pred_prices)
                
                expiry_pred = quantize_to_option_strikes(np.median(pred_prices_median[-24:]))

                variance = np.var(pred_prices_median) # get variance of predicted output ~ predicted volatility which correlates negatively with the value of buying a butterfly spread
                uncertainty = np.median(np.var(pred_prices, axis=0, ddof=1)) # get median variance across columns of prediction matrix ~ uncertainty of model

                debug_print('**************')
                debug_print(f'prediction: {expiry_pred}')
                debug_print(f'{variance=} {uncertainty=}')
                debug_print('**************')

                overall_penalty = variance + uncertainty
                if overall_penalty < overall_penalty_min:
                    overall_penalty_min = overall_penalty
                    strategy = dict(stock_symbol=ticker, predicted_price=expiry_pred, amount_to_purchase=1000)
            except Exception as e:
                print(f'Warn: exception {e}')

        if strategy is None:
            print('Warning: no viable strategy found')
        else:    
            ib_int.buy_butterfly(**strategy)

        ib_int.ib.sleep(15*60)  # wait 15 minutes otherwise we just burn through our API usages way too quickly
finally:
    ib_int.disconnect()
