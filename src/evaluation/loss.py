import numpy as np

from constants import BUTTERFLY_STRATEGY_SPREAD

def butterfly(x, k, k_spread, p=20, amt=1):
    """
    payoff function for short butterfly spread ~ loss of purchasing long butterfly spread

    x: stock price at expiration
    k: middle strike
    k_spread: distance from outer strikes to k (usually a nonzero multiple of 5)
    p: premium
    """
    return amt*(np.max(np.max([np.abs(x - k) - k_spread, np.abs(x - k) + k_spread]), 0) - p)

def butterfly_loss(xs, ys, spread=BUTTERFLY_STRATEGY_SPREAD):
    # since we don't care about a constant factor in a loss function we can ignore p
    
    return np.sum(np.maximum(np.abs(xs - ys), spread))  # == np.sum(np.max(np.abs(x - k), k_spread) for x, k in xs, ks)

