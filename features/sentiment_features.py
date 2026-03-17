from models.sentiment_model import get_sentiment_score

def aggregate_sentiment(news_list):
    if not news_list:
        return 0
    scores = [get_sentiment_score(n) for n in news_list]
    return sum(scores) / len(scores)

def add_sentiment_feature(data, news_list):
    sentiment = aggregate_sentiment(news_list)
    data["sentiment"] = sentiment
    return data