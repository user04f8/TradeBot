A trading bot designed for CS 523

# Setup

Substitute in the necessary API keys in secret_constants.py:

`
GNEWS_API_KEY = # replace with your own API key from https://gnews.io/dashboard
`

For reproducability, download the google news dataset available via https://github.com/rowanz/grover/tree/master/realnews. Note that this dataset is licensed for "research or education purposes" only. Full downloaded data is not included here due to its scale (>40 GB) and potential licensing restrictions. RealNews comes from Common Crawl and is thus subject to the common crawl license: http://commoncrawl.org/terms-of-use/

ORATS provides a comprehensive dataset of options data for backtesting, of which a small portion is available is provided as a sample (https://orats.com/university/historical-data#2-minute-snapshot). while I highly recommend utilizing this dataset for future work or further model refinements, to avoid licensing restrictions (https://orats.com/two-minute-data#pricing) no ORATS data or derived results, analyses, etc. are included in this codebase.

<!-- Note that some options data used was obtained by NASDAQ under a free academic use license; this license explicitly does not confer redistribution rights. All information is presented under the "as part of the classroom related activities" display rights conferred by this license. All data and related weights, charts, etc. obtained via NASDAQ has not been included in this code. 

For reproducability, data used is currently available via https://data.nasdaq.com/ using the `nasdaq-data-link` package, under "ORATS Smoothed Options Market Quotes" -->

To run the live model, run `src/live.py`.

Tests are included with many files if you run them individually, e.g. search for `if __name__ == '__main__': # TEST`.

# Credits

The LLM numerical serialization concept and portions of `src/data/llm_serializer.py` were adapted from the "Large Language Models Are Zero-Shot Time Series Forecasters" paper (https://arxiv.org/abs/2310.07820); the approach taken by `src/models/gpt.py` builds upon the capabilities developed in this paper. This code falls under the following license:

        MIT License

        Copyright (c) 2023 Nate Gruver, Marc Finzi, Shikai Qiu

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

All other code falls under my (Nathan Clark) copyright unless otherwise mentioned.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This software is for educational and academic purposes only. You should neither construe any of the material contained herein as financial nor investment advice, nor use anything contained herein as the basis for any investment decisions made by or on behalf of you.
