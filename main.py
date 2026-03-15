from data.data_loader import load_data
from features.feature_engineering import create_features
from models.regime_model import detect_regime
from visualize import plot_regime


# Load data
data = load_data()

# Create features
data = create_features(data)

# Detect market regime
data = detect_regime(data)

# Print results
print(data[["Close", "regime"]].tail())

# Plot regimes
plot_regime(data)