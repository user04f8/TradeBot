from datetime import datetime, timedelta
from enum import Enum
import json
from typing import List

import tarfile

from src.data.gnews_live_api import get_gnews, get_live_url
from src.data.utils import News
from constants import REALNEWS_TAR_PATH


class NewsInterface:
    def __init__(self):
        pass

    def get_news(self, ticker: str, n: int = 10) -> List[News]:
        return [News(headline=news_obj['title'], date=datetime.fromisoformat(news_obj['publishedAt'][:-1]), text=news_obj['content']) for news_obj in get_gnews(get_live_url(ticker, n=n))]
    
class NewsInterfaceHistorical(NewsInterface):
    TEST_NEWS_FNAME = r'dataset/news/gnews/realnews_tiny.jsonl'

    def __init__(self, cutoff_date: datetime, news_fname=REALNEWS_TAR_PATH, is_tar=True):
        self.cutoff_date = cutoff_date  # to enforce no train/test leakage

        if is_tar:
            tar = tarfile.open(news_fname)
            self.news_archive_file = tar.extractfile(tar.getmembers()[2])

            self.raw_news = None # This would be way too big given uncompressed dataset is 100s of gigabytes

        else:
            with open(news_fname) as f:
                self.raw_news = [json.loads(x) for x in f.readlines()]

    def news_generator(self):
        while True:
            news_obj = json.loads(self.news_archive_file.readline())  # pragmatically unlimited due to scale of dataset

            news_date = datetime.strptime(news_obj['publish_date'], '%m-%d-%Y')

            if news_date > self.cutoff_date:
                continue  # skip news that's from the future from the model's perspective

            yield News(headline=news_obj['title'], date=news_date, text=news_obj['text'])

    def get_news_by_idx(self, idx: int | slice) -> List[News]:
        if isinstance(idx, int):
            news_obj = self.raw_news[idx]
            return News(headline=news_obj['title'], date=datetime.strptime(news_obj['publish_date'], '%m-%d-%Y'), text=news_obj['text'])
        else:
            news = [News(headline=news_obj['title'], date=datetime.strptime(news_obj['publish_date'], '%m-%d-%Y'), text=news_obj['text']) for news_obj in self.raw_news[idx]]
        assert all(x.date <= self.cutoff_date for x in news)
        return news

if __name__ == '__main__':
    # UNCOMMENT for testing if you don't want to download the entire dataset
    # ni = NewsInterfaceHistorical(datetime.now(), NewsInterfaceHistorical.TEST_NEWS_FNAME, is_tar=False)

    # samples = ni.get_news_by_idx(slice(0,3))

    # print(samples)

    ni = NewsInterfaceHistorical(datetime.now())

    samples_gen = ni.news_generator()

    from time import sleep

    while True:
        print(next(samples_gen))
        sleep(1)