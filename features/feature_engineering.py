import pandas as pd

def create_features(data):

    data["returns"] = data["Close"].pct_change()

    data["volatility"] = data["returns"].rolling(20).std()

    data["ma50"] = data["Close"].rolling(50).mean()

    data["ma200"] = data["Close"].rolling(200).mean()

    data = data.dropna()

    return data