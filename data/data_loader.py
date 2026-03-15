import yfinance as yf

def load_data():
    data = yf.download("AAPL", start="2020-01-01")
    print(data.head())
    return data