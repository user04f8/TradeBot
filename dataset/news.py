from dataclasses import dataclass
import datetime
from enum import Enum

class NewsSource(Enum):
    MICROSOFT = 0
    SEEKING_ALPHA = 1
    
    
    MIND = 10
    

@dataclass
class News:
    headline: str
    date: datetime
    text: str
    source: NewsSource
