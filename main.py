from data.data_loader import load_data
from features.feature_engineering import create_features
from models.regime_model import detect_regime
from features.sentiment_features import add_sentiment_feature

data = load_data()
data = create_features(data)
data = detect_regime(data)

news = [
    "Tech stocks rally after strong earnings",
    "Investors confident about growth"
]

data = add_sentiment_feature(data, news)

print("Current Market Regime:", data["regime"].iloc[-1])
print("Sentiment Score:", data["sentiment"].iloc[-1])