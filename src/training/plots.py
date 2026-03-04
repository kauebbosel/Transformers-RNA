from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_curves(
    history: List[Dict],
    save_path: str,
    stopped_epoch: Optional[int] = None,
) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train loss", marker="o")
    ax1.plot(epochs, val_loss, label="Val loss", marker="o")
    if stopped_epoch:
        ax1.axvline(
            stopped_epoch,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Early stop",
        )
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train acc", marker="o")
    ax2.plot(epochs, val_acc, label="Val acc", marker="o")
    if stopped_epoch:
        ax2.axvline(
            stopped_epoch,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Early stop",
        )
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Training curves saved → {save_path}")

