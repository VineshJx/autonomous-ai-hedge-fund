import pandas as pd

def compute_rsi(df, window=14, price_col="Close"):
    delta = df[price_col].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(df, price_col="Close"):
    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = df[price_col].ewm(span=26, adjust=False).mean()

    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal


def add_technical_indicators(df):
    df = df.copy()

    df["rsi"] = compute_rsi(df)
    df["macd"], df["macd_signal"] = compute_macd(df)

    return df