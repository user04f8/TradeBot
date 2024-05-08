from datetime import datetime

from src.data.news_interface import NewsInterfaceHistorical
from src.data.utils import News
from src.snram.snram import SNRAM


nih = NewsInterfaceHistorical(datetime(2024, 1, 1))

news_gen = nih.news_generator()

snram = SNRAM()

def skip_news(n):
    for i in range(n):
        next(news_gen)

def get_news_and_tickers():
    relevant_tickers = set()
    while relevant_tickers == set():
        article: News = next(news_gen)
        relevant_tickers = snram.snramify(article)

    return article, relevant_tickers
    

if __name__ == '__main__':
    skip_news(50)

    article, relevant_tickers = get_news_and_tickers()

    assert len(relevant_tickers) <= 3

    article_summaries = [snram.summarize(article, relevant_ticker) for relevant_ticker in relevant_tickers]

    print(article)

    print(relevant_tickers)

    print(article_summaries)