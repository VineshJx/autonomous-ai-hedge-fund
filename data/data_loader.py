import yfinance as yf

def load_data(ticker="AAPL", start="2020-01-01", end=None, interval="1d"):
    """
    Load market data dynamically for any asset
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    data = data.dropna()
    
    return data