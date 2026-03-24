from data.data_loader import load_data
from data.data_splitter import train_test_split

from features.feature_engineering import create_features
from features.technical_indicators import add_technical_indicators
from features.sentiment_features import add_sentiment_feature

from models.regime_model import detect_regime

from agents.rl_trader import train_agent
from backtest.backtester import Backtester


def run_pipeline(ticker="AAPL"):

    print(f"\n🚀 Running pipeline for: {ticker}")

    # Load data
    data = load_data(ticker)

    # Features
    data = create_features(data)
    data = add_technical_indicators(data)

    # Regime
    data = detect_regime(data)

    # Sentiment
    news = [
        f"{ticker} performing well",
        f"Positive outlook for {ticker}"
    ]
    data = add_sentiment_feature(data, news)

    # Clean
    data = data.dropna().reset_index(drop=True)

    # Split
    train_data, test_data = train_test_split(data)

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    # Train
    model = train_agent(train_data)

    # Backtest
    backtester = Backtester(test_data, model)
    results = backtester.run()

    print("\n📊 RESULTS:")
    print("Final Value:", results["final_value"])
    print("Return:", results["total_return"])
    print("Sharpe:", results["sharpe_ratio"])
    print("Drawdown:", results["max_drawdown"])

    return model, results


if __name__ == "__main__":
    run_pipeline("AAPL")