from typing import List

from src.data.news_interface import News
from src.snram.snram import SNRAM, client

class SNRAMLive(SNRAM):
    

    def summarize_many(self, news_articles: List[News], relevant_ticker: str, max_tokens: int = 200) -> str:
        self.summarize_preprompt = lambda relevant_ticker: f'Please concisely summarize the following news articles as they are relevant to {self.ticker_to_name[relevant_ticker]}. Provide key details and a general sentiment from a range of negative to positive. '

        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                    {"role": "system", "content": self.sys_message_summarize},
                    {"role": "user", "content": "\n\n".join([self.summarize_preprompt(relevant_ticker)] + [f'{news_article.headline}\n{news_article.text}' for news_article in news_articles])}
                ],
            max_tokens=max_tokens,
            temperature=0.5,
            n=1
        )
        return response.choices[0].message.content