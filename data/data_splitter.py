def train_test_split(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    return train_data, test_data