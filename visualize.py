import matplotlib.pyplot as plt

def plot_regime(data):

    plt.figure(figsize=(10,5))

    plt.scatter(data.index, data["Close"], c=data["regime"], cmap="viridis")

    plt.title("Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.show()