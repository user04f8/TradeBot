import json
import urllib.request


from secret_constants import GNEWS_API_KEY

def get_live_url(search_term, n):
    # see https://gnews.io/docs/v4#search-endpoint-query-parameters for documentation
    return f"https://gnews.io/api/v4/search?q={search_term}&lang=en&country=us&max={n}&apikey={GNEWS_API_KEY}"

datetime_format = '%Y-%m-%dT%H:%M:%SZ'

def get_url(search_term, start_date, end_date, n):
    # technically this could be used for historical data but it could get expensive
    return f"https://gnews.io/api/v4/search?q={search_term}&from={start_date}&to={end_date}&lang=en&country=us&max={n}&apikey={GNEWS_API_KEY}"

def get_gnews(url, debug=False):
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        articles = data["articles"]

        if debug:
            for article in articles:
                # articles[i].title
                print(f"Title: {article['title']}")
                # articles[i].description
                print(f"Description: {article['description']}")
                # You can replace {property} below with any of the article properties returned by the API.
                # articles[i].{property}
                # print(f"{articles[i]['{property}']}")

                print(article)
        
        return articles