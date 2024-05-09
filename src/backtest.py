from datetime import datetime, timedelta
import json
import numpy as np
import traceback

from constants import N_TRIES, DEBUG, BUTTERFLY_STRATEGY_SPREAD
from src.utils import get_ticker_name_dict, quantize_to_option_strikes
from src.evaluation.loss import butterfly
from src.data.market_interface import MarketData
from src.data.news_interface import NewsInterfaceHistorical
from src.snram.snram import SNRAM
from src.snram.historic_aggregator import skip_news, get_news_and_tickers
from src.models.make_preds import make_pred


def debug_print(*args, to_log=None):
    if DEBUG:
        print(*args)
    if to_log is not None:
        with open(to_log, 'a') as f:
            f.write(str(args[0]) + '\n')

with open('dataset/markets/train/metadata.json') as f:
    tickers = json.load(f)
ticker_names = get_ticker_name_dict()

# snram = SNRAM()
# nih = NewsInterfaceHistorical()
# nih_gen = nih.news_generator()
skip_news(50) # skip the first 50 samples since they were used in hyperparameter optimization / could feasibly be overfit to

md = MarketData()
snram = SNRAM()

simulated_now = datetime(2024, 1, 1)
time_to_predict = timedelta(days=1)

payoffs = []
net_payoff = 0
n_purchases = 0

N_TRIES = 3 # for hyperparameter tuning

print('Simulation is now running (not actual TradeBot)')
for i in range(50):
    overall_penalty_min = 1e9
    strategy = None
    for i in range(N_TRIES):
        try:
            article, ticker = get_news_and_tickers()
            ticker = ticker.pop() # unwrap set

            stock_history = md.load_prices_at_datetime(ticker, simulated_now, 336)
            actual_price = md.load_prices_at_datetime(ticker, simulated_now + time_to_predict, 1)[0]
            norm_factor = stock_history[-1]

            news_summary = snram.summarize(article, ticker)

            debug_print(stock_history)
            debug_print(news_summary)

            pred_prices, pred_prices_median = make_pred(stock_history, len_out=(24))

            debug_print(pred_prices)
            
            expiry_pred = quantize_to_option_strikes(np.median(pred_prices_median[-24:]))

            variance = np.var(pred_prices_median) / norm_factor # get variance of predicted output ~ predicted volatility which correlates negatively with the value of buying a butterfly spread
            uncertainty = np.median(np.var(pred_prices, axis=0, ddof=1)) / norm_factor # get median variance across columns of prediction matrix ~ uncertainty of model

            debug_print('**************', to_log='logfile.txt')
            debug_print(f'prediction: {expiry_pred}', to_log='logfile.txt')
            debug_print(f'{variance=} {uncertainty=}', to_log='logfile.txt')
            print(type(expiry_pred), expiry_pred)
            print(type(actual_price), actual_price)
            simulated_payoff = butterfly(expiry_pred, actual_price, k_spread=BUTTERFLY_STRATEGY_SPREAD)
            debug_print(f'expected gain: {simulated_payoff:.4f}', to_log='logfile.txt')
            debug_print('**************', to_log='logfile.txt')

            with open('hyperparameter_tuning.txt', 'a') as f:
                f.write(f'{variance}, {uncertainty}, {simulated_payoff}, {norm_factor}, {expiry_pred}, {actual_price}\n')

            overall_penalty = variance + uncertainty
            if overall_penalty < overall_penalty_min:
                overall_penalty_min = overall_penalty
                strategy = dict(stock_symbol=ticker, predicted_price=expiry_pred, amount_to_purchase=1)
        except Exception as e:
            print(traceback.format_exc())

    if strategy is None:
        print('Warning: no viable strategy found')
    else:
        actual_price = md.load_prices_at_datetime(ticker, simulated_now + time_to_predict, 1)[0]
        simulated_payoff = butterfly(strategy['predicted_price'], actual_price, k_spread=BUTTERFLY_STRATEGY_SPREAD)
        debug_print('!!!!!!!!!!!!!!!!!', to_log='logfile.txt')
        debug_print(f'Trade: {strategy}', to_log='logfile.txt')
        debug_print(f'Actual: {actual_price}', to_log='logfile.txt')
        debug_print(f'Simulated diff for trade resulted in {simulated_payoff:.4f}', to_log='logfile.txt')
        debug_print('!!!!!!!!!!!!!!!!!', to_log='logfile.txt')
        payoffs.append(simulated_payoff)
        net_payoff += simulated_payoff
        n_purchases += 1

    simulated_now += timedelta(minutes=15)

print(payoffs)
print(net_payoff)