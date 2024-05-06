from datetime import datetime, timedelta
from enum import Enum
import json
from typing import List

from .gnews_live_api import get_gnews, get_live_url
from .utils import News



class NewsInterface:
    def __init__(self):
        pass

    def get_news(self, ticker: str) -> List[News]:
        return [News(headline=news_obj['title'], date=datetime.fromisoformat(news_obj['publishedAt'][:-1]), text=news_obj['content']) for news_obj in get_gnews(get_live_url(ticker))]
    
class NewsInterfaceHistorical(NewsInterface):
    DEFAULT_NEWS_FNAME = r'dataset/news/gnews/realnews.jsonl'
    TEST_NEWS_FNAME = r'dataset/news/gnews/realnews_tiny.jsonl'

    def __init__(self, cutoff_date: datetime, news_fname=DEFAULT_NEWS_FNAME):
        self.cutoff_date = cutoff_date  # to enforce no train/test leakage

        with open(news_fname) as f:
            self.raw_news = [json.loads(x) for x in f.readlines()]

    def get_news_by_idx(self, idx: int | slice) -> List[News]:
        if isinstance(idx, int):
            news_obj = self.raw_news[idx]
            return News(headline=news_obj['title'], date=datetime.strptime(news_obj['publish_date'], '%m-%d-%Y'), text=news_obj['text'])
        else:
            news = [News(headline=news_obj['title'], date=datetime.strptime(news_obj['publish_date'], '%m-%d-%Y'), text=news_obj['text']) for news_obj in self.raw_news[idx]]
        assert all(x.date <= self.cutoff_date for x in news)
        return news

if __name__ == '__main__':
    ni = NewsInterfaceHistorical(datetime.now(), NewsInterfaceHistorical.TEST_NEWS_FNAME)

    samples = ni.get_news_by_idx(slice(0,3))

    print(samples)