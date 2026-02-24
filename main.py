import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import get_dataloaders
from data.dataset import IMDBDataset
from torch.utils.data import DataLoader

def main():
    train_loader, test_loader, vocab_size = get_dataloaders()
    train_dataset = train_loader.dataset

    all_inputs = []
    all_labels = []

    for i in range(len(train_dataset)):
        input_ids, label = train_dataset[i]
        all_inputs.append(input_ids)
        all_labels.append(label)

    all_inputs = torch.stack(all_inputs)
    all_labels = torch.stack(all_labels)

    torch.save({
        "input_ids": all_inputs,
        "labels": all_labels
    }, "data/processed_train.pt")

if __name__ == "__main__":
    main()