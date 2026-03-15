from sklearn.cluster import KMeans

def detect_regime(data):

    features = data[["returns","volatility"]]

    model = KMeans(n_clusters=3)

    data["regime"] = model.fit_predict(features)

    return data