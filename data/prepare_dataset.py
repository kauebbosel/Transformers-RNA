import os
import random

def load_reviews(path, label, max_samples):
    files = os.listdir(path)
    random.shuffle(files)
    
    texts = []
    
    for file in files[:max_samples]:
        with open(os.path.join(path, file), encoding="utf-8") as f:
            texts.append((f.read(), label))
    
    return texts


def build_small_dataset(base_path, train_size=2000, test_size=500, seed=42):
    random.seed(seed)

    train_pos = load_reviews(os.path.join(base_path, "train/pos"), 1, train_size)
    train_neg = load_reviews(os.path.join(base_path, "train/neg"), 0, train_size)

    test_pos = load_reviews(os.path.join(base_path, "test/pos"), 1, test_size)
    test_neg = load_reviews(os.path.join(base_path, "test/neg"), 0, test_size)

    train_data = train_pos + train_neg
    test_data = test_pos + test_neg

    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data