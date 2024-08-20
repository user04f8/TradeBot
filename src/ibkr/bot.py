from ib_insync import *
from datetime import datetime

from constants import BUTTERFLY_STRATEGY_SPREAD

NEXT_OPTIONS_EXPIRY = '20240724'
NEXT_OPTIONS_EXPIRY_DATETIME = datetime(2024, 7, 24)

class IB_Interface:
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)

    def disconnect(self):
        self.ib.disconnect()

    def get_local_daily(self, stock_symbol, n_days: int):
        bars = self.ib.reqHistoricalData(
            Stock(stock_symbol, exchange='SMART', currency='USD'),
            endDateTime='',  # Empty string means up to the current moment
            durationStr=f'{n_days} D',
            barSizeSetting='1 day',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2)
        
        self.ib.sleep(1)

        return [bar.open for bar in bars]

    def get_local_hourly(self, stock_symbol, n_hours: int):
        bars = self.ib.reqHistoricalData(
            Stock(stock_symbol, exchange='SMART', currency='USD'),
            endDateTime='',  # Empty string means up to the current moment
            durationStr=f'{n_hours // 24} D',
            barSizeSetting='1 hour',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2)
        
        self.ib.sleep(1)

        return [bar.open for bar in bars]
        
    
    def get_local_minutely(self, stock_symbol, n_mins: int):
        bars = self.ib.reqHistoricalData(
            stock_symbol,
            endDateTime='',  # Empty string means up to the current moment
            durationStr=f'{60 * n_mins} S',
            barSizeSetting='1 min',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2)
        
        self.ib.sleep(1)

        return [bar.open for bar in bars]
            
        

    def buy_butterfly(self, stock_symbol, predicted_price, amount_to_purchase, expiry_date=NEXT_OPTIONS_EXPIRY, spread=BUTTERFLY_STRATEGY_SPREAD):
        """
        Automatically purchases a butterfly spread about a given predicted_price

        stock_symbol: str   e.g. QQQ 
        expiry_date: str    can be YYYYYYYMM or YYYYMMDD

        """

        # Long call buttefly = call at (k-spread) - 2 * call at k + call at (k+spread)
        option_kwargs = dict(symbol=stock_symbol, lastTradeDateOrContractMonth=expiry_date, right='CALL', multiplier=100, exchange='SMART', currency='USD')
        lower_call = Option(strike = predicted_price - spread, **option_kwargs)
        atm_call = Option(strike = predicted_price, **option_kwargs)
        higher_call = Option(strike = predicted_price + spread, **option_kwargs)

        lower_call, atm_call, higher_call = self.ib.qualifyContracts(lower_call, atm_call, higher_call)

        # THIS WOULD WORK except it's a paper trading account and not a subscribed account :(
        # snapshot = True
        # prices = self.ib.reqMktData(lower_call, '', snapshot, False), self.ib.reqMktData(atm_call, '', snapshot, False), self.ib.reqMktData(higher_call, '', snapshot, False)
        # self.ib.sleep(1) # wait for tick to update

        # Instead we rely on 15 minute old data

        def fetch_price(option):
            bars = self.ib.reqHistoricalData(
                option,
                endDateTime='',
                durationStr='900 S',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=2)
            if bars:
                return bars[0].close  # Close price of the bar
            else:
                return None
            
        prices = [fetch_price(lower_call), fetch_price(atm_call), fetch_price(higher_call)]
        
        net_cost = prices[0] - 2*prices[1] + prices[2]

        quantity_mult = amount_to_purchase // net_cost

        orders = [
            # Order(action='BUY', orderType='LMT', totalQuantity=quantity_mult, lmtPrice=prices[0].ask, transmit=False),
            # Order(action='SELL', orderType='LMT', totalQuantity=2*quantity_mult, lmtPrice=prices[1].bid, transmit=False),
            # Order(action='BUY', orderType='LMT', totalQuantity=quantity_mult, lmtPrice=prices[2].ask, transmit=True) # transmit only on the final order being placed successfully
            MarketOrder('BUY', quantity_mult),
            MarketOrder('SELL', 2*quantity_mult),
            MarketOrder('BUY', quantity_mult)
        ]

        trade1 = self.ib.placeOrder(lower_call, orders[0])
        trade2 = self.ib.placeOrder(atm_call, orders[1])
        trade3 = self.ib.placeOrder(higher_call, orders[2])

        print(f'butterfly: {stock_symbol}x{quantity_mult} for est. ${net_cost * quantity_mult}')

class Fake_IB_Interface:
    def __init__(self):
        print('Init fake IB interface')
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)

    def print(self, msg):
        print(f'fake_ib_interface: {msg}')

    def disconnect(self):
        self.print('disconnect')
        self.ib.disconnect()

    def get_local_daily(self, stock_symbol, n_days: int):
        bars = self.ib.reqHistoricalData(
            Stock(stock_symbol, exchange='SMART', currency='USD'),
            endDateTime='',  # Empty string means up to the current moment
            durationStr=f'{n_days} D',
            barSizeSetting='1 day',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2)
        
        self.ib.sleep(1)

        return [bar.open for bar in bars]

    def get_local_hourly(self, stock_symbol, n_hours: int):
        bars = self.ib.reqHistoricalData(
            Stock(stock_symbol, exchange='SMART', currency='USD'),
            endDateTime='',  # Empty string means up to the current moment
            durationStr=f'{n_hours // 24} D',
            barSizeSetting='1 hour',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2)
        
        self.ib.sleep(1)

        return [bar.open for bar in bars]
        
    
    def get_local_minutely(self, stock_symbol, n_mins: int):
        bars = self.ib.reqHistoricalData(
            stock_symbol,
            endDateTime='',  # Empty string means up to the current moment
            durationStr=f'{60 * n_mins} S',
            barSizeSetting='1 min',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2)
        
        self.ib.sleep(1)

        return [bar.open for bar in bars]
            
        

    def buy_butterfly(self, stock_symbol, predicted_price, amount_to_purchase, expiry_date=NEXT_OPTIONS_EXPIRY, spread=BUTTERFLY_STRATEGY_SPREAD):
        self.print(f'would purchase butterfly: {stock_symbol} w/ predicted {predicted_price}')



if __name__ == '__main__':
    # SANITY TEST

    ib_inter = IB_Interface()
    ib_inter.buy_butterfly(stock_symbol='QQQ', predicted_price=440, amount_to_purchase=10)
    ib_inter.disconnect()