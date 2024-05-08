import csv
from pathlib import Path


class MarketData:
    def __init__(self, dir: Path = Path('dataset/markets/val')):
        self.dir = dir
        self.options_reader = None

    def set_ticker(self, ticker):
        with open(dir + Path(f'{ticker}_options.csv')) as f:
            self.options_reader = csv.reader(f)
            next(self.options_reader)

    def get(self, t):
        return self.options_reader[t]
        # **** TODO ****

    # def get_options_chain(self, date):
        # If BU finally buys and makes access available to reasonable data (e.g. Bloomberg terminal for non-Questrom students) or I'm willing to pay $2k,
        # this could be so much more powerful of a model, allowing for finetuning on options chain data for intraday pricing at each timestep.
    