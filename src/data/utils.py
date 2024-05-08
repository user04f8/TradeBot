from dataclasses import dataclass
from datetime import datetime
# from enum import Enum

# class NewsSource(Enum):
#     MICROSOFT = 0
#     GNEWS = 1

@dataclass
class News:
    headline: str
    text: str
    date: datetime = None
    # source: NewsSource
