A trading bot that was the winning submission for [Iddo Drori](https://www.cs.columbia.edu/~idrori/)'s deep learning stock market competition. Full details of the design, implementation, and some backtest metrics are available in (report.pdf)[report.pdf]

# Setup

Substitute in the necessary API keys in secret_constants.py:

`
GNEWS_API_KEY = # replace with your own API key from https://gnews.io/dashboard
`

For reproducability, download the google news dataset available via https://github.com/rowanz/grover/tree/master/realnews. Note that this dataset is licensed for "research or education purposes" only. Full downloaded data is not included here due to its scale (>100 GB) and potential licensing restrictions. RealNews comes from Common Crawl and is thus subject to the common crawl license: http://commoncrawl.org/terms-of-use/

ORATS provides a comprehensive dataset of options data for backtesting, of which a small portion is available is provided as a sample (https://orats.com/university/historical-data#2-minute-snapshot). while I highly recommend utilizing this dataset for future work or further model refinements, to avoid licensing restrictions (https://orats.com/two-minute-data#pricing) no ORATS data or derived results, analyses, etc. are included in this codebase.

To run the live model, run `python src/live.py`.

To reproduce backtests, run `python src/backtest.py`.