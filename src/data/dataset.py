import os
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset


def load_reviews(path, label, max_samples):
    files = os.listdir(path)
    random.shuffle(files)

    texts = []

    for file in files[:max_samples]:
        with open(os.path.join(path, file), encoding="utf-8") as f:
            texts.append((f.read(), label))

    return texts


def preprocess(text, max_len):
    tokens = text.lower().split()
    tokens = tokens[:max_len]
    return tokens


def build_vocab(texts, min_freq=2):
    counter = Counter()

    for tokens in texts:
        counter.update(tokens)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))

    return ids


class IMDBDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        tokens = preprocess(text, self.max_len)
        ids = encode(tokens, self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


def get_dataloaders(
    base_path="data/aclImdb",
    train_samples=2000,
    test_samples=500,
    max_len=200,
    batch_size=32,
    seed=42,
):
    random.seed(seed)

    train_pos = load_reviews(os.path.join(base_path, "train/pos"), 1, train_samples)
    train_neg = load_reviews(os.path.join(base_path, "train/neg"), 0, train_samples)

    test_pos = load_reviews(os.path.join(base_path, "test/pos"), 1, test_samples)
    test_neg = load_reviews(os.path.join(base_path, "test/neg"), 0, test_samples)

    train_data = train_pos + train_neg
    test_data = test_pos + test_neg

    random.shuffle(train_data)
    random.shuffle(test_data)

    tokenized_texts = [preprocess(text, max_len) for text, _ in train_data]
    vocab = build_vocab(tokenized_texts)

    train_dataset = IMDBDataset(train_data, vocab, max_len)
    test_dataset = IMDBDataset(test_data, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, len(vocab)

