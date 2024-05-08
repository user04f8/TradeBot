import csv

def get_ticker_name_dict():
    symbols_names: dict = {}

    with open('dataset/markets/nasdaq_screener.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for row in csv_reader:
            if '^' in row[0]:
                continue

            symbol_name = row[1].split('Inc')[0] \
                                    .split('Ltd')[0] \
                                    .split('Corporation')[0] \
                                    .split('N.A.')[0] \
                                    .split('Common Stock')[0] \
                                    .split('Preferred Stock')[0] \
                                    .split('Ordinary Shares')[0] \
                                    .split('Depository Shares')[0] \
                                    .split('(The)')[0] \
                                    .split('Class A')[0] \
                                    .split('Each')[0] \
                                    .strip()

            symbols_names[row[0]] = symbol_name

    return symbols_names

def get_name_ticker_dict(min_market_cap=300e6):
    """
    min_market_cap: minimum market capitalization to include a stock, in USD
    """
    symbols_names: dict = {}

    with open('dataset/markets/nasdaq_screener.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for row in csv_reader:
            if '^' in row[0]:
                continue
            try:
                market_cap = float(row[5])
            except ValueError:
                market_cap = 0
            if market_cap >= min_market_cap:
                # basic preprocessing to keep things readable and concise
                symbol_name = row[1].split('Inc')[0] \
                                    .split('Ltd')[0] \
                                    .split('Corporation')[0] \
                                    .split('N.A.')[0] \
                                    .split('Common Stock')[0] \
                                    .split('Preferred Stock')[0] \
                                    .split('Ordinary Shares')[0] \
                                    .split('Depositary Shares')[0] \
                                    .split('(The)')[0] \
                                    .split('Class A')[0] \
                                    .split('Each')[0] \
                                    .strip()
                
                symbols_names[symbol_name] = row[0]

    return symbols_names

def get_industry_ticker_dict(industry_min_market_cap=1e12, min_market_cap=5e9):
    industries: dict = {}
    industries_market_caps: dict = {}

    with open('dataset/markets/nasdaq_screener.csv') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header
        for row in csv_reader:
            try:
                market_cap = float(row[5])
            except ValueError:
                market_cap = 0
            if market_cap >= min_market_cap:
                industry = row[-1]
                industry = industry.replace('(Production/Distribution)', ''
                                  ).replace('EDP', 'Electronic Data Processing'
                                  ).replace('/Specialty', ''
                                  ).replace('/Service', ''
                                  ).split(':')[0]  # basic preprocessing to keep things readable and concise
                industry = industry.strip()
                if industry == '':
                    continue
                if industry not in industries:
                    # add new industry
                    industries[industry] = []
                    industries_market_caps[industry] = 0
                industries[industry].append(row[0])
                industries_market_caps[industry] += market_cap

    pruned_industries = {}
    for industry in industries.keys():
        if industries_market_caps[industry] >= industry_min_market_cap:
            pruned_industries[industry] = industries[industry]

    return pruned_industries

def quantize_to_option_strikes(x: float) -> int:
    return int(round(x / 5) * 5)
