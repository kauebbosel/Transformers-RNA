from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import get_dataloaders
from src.models.transformer_sentiment import TransformerSentiment
from src.training.loop import EarlyStopping, evaluate, train_one_epoch
from src.training.plots import plot_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransformerSentiment on IMDB")

    p.add_argument("--base-path", type=str, default="data/aclImdb")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=500)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1024)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-file", type=str, default="training_log.csv")
    p.add_argument("--plot-file", type=str, default="training_curves.png")
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def get_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str, model, optimizer, scheduler, epoch, best_val_acc, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        },
        path,
    )
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])
    print(
        f"  ✓ Resumed from {path} (epoch {ckpt['epoch']}, "
        f"best_val_acc={ckpt['best_val_acc']:.4f})"
    )
    return ckpt["epoch"], ckpt["best_val_acc"]


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"\n{'=' * 55}")
    print("  TransformerSentiment — IMDB Training")
    print(f"{'=' * 55}")
    print(f"  Device  : {device}")
    print(f"  Data    : {args.base_path}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  d_model : {args.d_model}")
    print(f"  Patience: {args.patience if args.patience > 0 else 'disabled'}")
    print(f"{'=' * 55}\n")

    print("Loading dataloaders...")
    train_loader, test_loader, vocab_size = get_dataloaders(
        base_path=args.base_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    print(f"  vocab_size      = {vocab_size:,}")
    print(
        f"  train samples   = {args.train_samples * 2} "
        f"({args.train_samples} pos + {args.train_samples} neg)"
    )
    print(
        f"  test  samples   = {args.test_samples * 2} "
        f"({args.test_samples} pos + {args.test_samples} neg)"
    )
    print(f"  train steps     = {len(train_loader)}")
    print(f"  test  steps     = {len(test_loader)}\n")

    model = TransformerSentiment(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.num_classes,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 1
    best_val_acc = 0.0

    if args.checkpoint and Path(args.checkpoint).exists():
        start_epoch, best_val_acc = load_checkpoint(
            args.checkpoint, model, optimizer, scheduler
        )
        start_epoch += 1

    log_path = args.log_file
    log_file_exists = Path(log_path).exists() and start_epoch > 1
    log_fh = open(log_path, "a", newline="")
    writer = csv.DictWriter(
        log_fh,
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
    )
    if not log_file_exists:
        writer.writeheader()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(patience=args.patience)
    history = []
    stopped_epoch = None

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = scheduler.get_last_lr()[0] if epoch > 1 else args.lr
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={current_lr:.2e})")
        print("-" * 45)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip, epoch
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(
            f"\n  → train  loss={train_loss:.4f}  acc={train_acc:.4f}"
            f"\n  → val    loss={val_loss:.4f}  acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                str(ckpt_dir / "best_model.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                args,
            )
            print(f"  ★ New best val acc: {best_val_acc:.4f}")

        save_checkpoint(
            str(ckpt_dir / "last_model.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc,
            args,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(round(train_loss, 6)),
            "train_acc": float(round(train_acc, 6)),
            "val_loss": float(round(val_loss, 6)),
            "val_acc": float(round(val_acc, 6)),
            "lr": float(round(current_lr, 8)),
        }
        writer.writerow(row)
        log_fh.flush()
        history.append(row)

        if early_stopper.step(val_acc):
            stopped_epoch = epoch
            print(f"\n  ⛔ Early stopping triggered at epoch {epoch}.")
            break

    log_fh.close()

    print(f"\n{'=' * 55}")
    print("  Training complete.")
    if stopped_epoch:
        print(f"  Stopped early      : epoch {stopped_epoch}/{args.epochs}")
    print(f"  Best val accuracy  : {best_val_acc:.4f}")
    print(f"  Log                : {log_path}")
    print(f"  Best checkpoint    : {ckpt_dir}/best_model.pt")
    print(f"{'=' * 55}\n")

    if history:
        plot_curves(history, args.plot_file, stopped_epoch=stopped_epoch)


if __name__ == "__main__":
    main()

