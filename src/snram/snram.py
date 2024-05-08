from src.utils import get_name_ticker_dict, get_industry_ticker_dict, get_ticker_name_dict
from src.data.news_interface import News

from openai import OpenAI

from secret_constants import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)

class SNRAM:
    def __init__(self):
        self.name_to_ticker: dict = get_name_ticker_dict(1e11)
        self.ticker_to_name: dict = get_ticker_name_dict()
        self.industry_to_ticker: dict = get_industry_ticker_dict()
        self.sys_message = "You are a helpful assistant that provides the user the stocks most relevant to a given news article"
        self.sys_message_industry = "You are a helpful assistant that provides the user the industry most relevant to a given news article"
        self.stock_preprompt = "Please return only the stock most relevant to the news article, or None if none are of particular relevance. Select from the following list of stocks, providing an output exactly as it appears below: \n" \
                    + " ".join(stock_name for stock_name in self.name_to_ticker.keys()) \
                    + "\n"
        self.industry_preprompt = "Please return only the industry most relevant to the news article, or None if none are of particular relevance. Select from the following list of industries, providing an output exactly as it appears below: \n" \
                    + " ".join(stock_name for stock_name in self.industry_to_ticker.keys()) \
                    + "\n"
        
        self.sys_message_summarize = 'You are a helpful assistant that provides the user summaries of news articles as is relevant to a provided company.'
        self.summarize_preprompt = lambda relevant_ticker: f'Please concisely summarize the following news article as is relevant to {self.ticker_to_name[relevant_ticker]}. Provide key details and a general sentiment from a range of negative to positive'
    
    def snramify(self, news_article: News):
        """
        SNRAMify obtains stock tickers relevant to a provided news article 
        """

        # we could also do basic keyword searches here, e.g.:
        # headline_words = news_article.headline.split()
        # ...

        relevant_tickers = set()
        
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                    {"role": "system", "content": self.sys_message},
                    {"role": "user", "content": "\n".join([self.stock_preprompt, news_article.headline, news_article.text])}
                ],
            max_tokens=10,
            temperature=0.7,
            n=2
        )

        for choice in response.choices:
            model_output = choice.message.content
            ticker = self.name_to_ticker.get(model_output)
            if ticker is not None:
                relevant_tickers.add(ticker)
            elif model_output == 'None':
                pass
            else:
                print(f'Warning invalid model output {model_output}')

        if len(relevant_tickers) == 0:

            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                        {"role": "system", "content": self.sys_message_industry},
                        {"role": "user", "content": "\n".join([self.industry_preprompt, news_article.headline, news_article.text])}
                    ],
                max_tokens=10,
                temperature=0.5,
                n=1
            )

            for choice in response.choices:
                model_output = choice.message.content
                tickers = self.industry_to_ticker.get(model_output)
                if tickers is not None:
                    relevant_tickers.union(tickers)
                elif model_output == 'None':
                    pass
                else:
                    print(f'Warning invalid model output {model_output}')

        return relevant_tickers
    
    def summarize(self, news_article: News, relevant_ticker: str, max_tokens: int = 100) -> str:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                    {"role": "system", "content": self.sys_message_summarize},
                    {"role": "user", "content": "\n".join([self.summarize_preprompt(relevant_ticker), news_article.headline, news_article.text])}
                ],
            max_tokens=max_tokens,
            temperature=0.5,
            n=1
        )
        return response.choices[0].message.content


# Hey you got towards the end of the codebase! that's great :)
# Here's a fun fact:
# Originally this file was called StoNER = Stock/News Evaluation of Relevancy,
#   but I think this refactor to SNRAM is for the best

if __name__ == '__main__':
    # SANITY TESTS

    snram = SNRAM()
    output = snram.snramify(
        News(headline="Google announces new AI program named Bard, in wake of viral ChatGPT",
             text="""If you ever needed artificial intelligence's help to plan a friend's baby shower, your time has come, as Google officially unveiled an AI program it's calling Bard, seemingly its answer to the viral ChatGPT.

Google CEO Sundar Pichai announced Bard on the company's blog Monday, calling it "an important next step" in AI for the search engine giant.

"Bard seeks to combine the breadth of the world's knowledge with the power, intelligence and creativity of our large language models," Pichai said. "It draws on information from the web to provide fresh, high-quality responses."

The company's new AI chatbot is based on its Language Model for Dialogue Applications (LaMDA) and is only available to a select group of testers.

Based on a video shared by Pichai on Twitter, users could use Bard to compare two Oscar-nominated movies, come up with ideas for lunch based on the ingredients in a person's refrigerator or know about the latest discoveries from the James Webb Telescope.

Google said Bard would be widely available to the public in the next few weeks.

"It's a really exciting time to be working on these technologies as we translate deep research and breakthroughs into products that truly help people," Pichai said.

Bard is seemingly Google's answer to ChatGPT, an AI-driven program that exploded in popularity in the last few months after users shared posts of the tool composing Shakespearean poetry, writing music lyrics and identifying bugs in computer code.

Created by artificial intelligence firm OpenAI, ChatGPT, which stands for Chat Generative Pre-Trained Transformer, is a chatbot -- a computer program that converses with human users. The program uses an algorithm that selects words based on lessons learned from scanning billions of pieces of text across the internet.

Microsoft, which invested in OpenAI in 2019 and 2021, announced last month that it's extending its partnership with the firm and investing billions of dollars into the company.

According to Forbes, Microsoft is investing up to $10 billion into ChatGPT and may incorporate it into its Bing search engine.""")
    )
    print(output)
