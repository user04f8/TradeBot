from src.utils import get_name_ticker_dict, INDUSTRY_TICKER_DICT
from dataset.news_interface import News

from openai import OpenAI

from secret_constants import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)

class SNRAM:
    def __init__(self):
        name_to_ticker: dict = get_name_ticker_dict()
        industry_to_ticker: dict = INDUSTRY_TICKER_DICT

    def snramify(self, news_article: News):
        """
        SNRAMify obtains stock tickers relevant to a provided news article 
        """

        sys_message = "You are a helpful assistant"
        preprompt = "Please return the most "
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": preprompt+input_str+settings.time_sep}
                ],
            max_tokens=10,
            temperature=temp,
            n=1
        )






# Hey you got towards the end of the codebase! that's great :)
# Here's a fun fact:
# Originally this file was called StoNER = Stock/News Evaluation of Relevancy,
#   but I think this refactor to SNRAM is for the best

