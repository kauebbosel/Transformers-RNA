from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Halts training when val_acc has not improved for `patience` consecutive
    epochs. Improvement must be at least `min_delta` to count.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, val_acc: float) -> bool:
        """Call after each epoch. Returns True when training should stop."""
        if self.patience == 0:
            return False
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.counter = 0
        else:
            self.counter += 1
            print(
                f"  ⚠ No improvement for {self.counter}/{self.patience} epoch(s)"
                f"  (best={self.best:.4f})"
            )
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy) for the epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for step, (input_ids, labels) in enumerate(loader, 1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        if step % 100 == 0 or step == len(loader):
            elapsed = time.time() - start
            print(
                f"  [Epoch {epoch}] step {step:>4}/{len(loader)}"
                f"  loss={total_loss/total:.4f}"
                f"  acc={correct/total:.4f}"
                f"  ({elapsed:.1f}s)"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy) over the full loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total

