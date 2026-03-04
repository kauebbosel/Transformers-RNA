from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import get_dataloaders
from src.models.transformer_sentiment import TransformerSentiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a TransformerSentiment checkpoint")

    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to .pt checkpoint (default: checkpoints/best_model.pt)",
    )
    p.add_argument("--base-path", type=str, default="data/aclImdb")
    p.add_argument(
        "--test-samples",
        type=int,
        default=500,
        help="Positive OR negative samples (total = 2x)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Optional path to save per-sample CSV predictions",
    )
    p.add_argument(
        "--confusion-plot",
        type=str,
        default="confusion_matrix.png",
        help="Where to save the confusion matrix heatmap",
    )

    return p.parse_args()


def get_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    all_labels: list[int],
    all_preds: list[int],
    num_classes: int = 2,
) -> dict:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    accuracy = sum(cm[i][i] for i in range(num_classes)) / len(all_labels)

    per_class = {}
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes)) - tp
        fn = sum(cm[c][r] for r in range(num_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

    macro_precision = sum(v["precision"] for v in per_class.values()) / num_classes
    macro_recall = sum(v["recall"] for v in per_class.values()) / num_classes
    macro_f1 = sum(v["f1"] for v in per_class.values()) / num_classes

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    all_labels, all_preds, all_confs = [], [], []

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_labels, all_preds, all_confs


def plot_confusion_matrix(
    cm: list[list[int]],
    save_path: str,
    class_names=("Negative", "Positive"),
):
    import numpy as np

    cm_arr = [[cm[r][c] for c in range(len(cm[0]))] for r in range(len(cm))]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    total = sum(sum(row) for row in cm_arr)
    for r in range(len(cm_arr)):
        for c in range(len(cm_arr[0])):
            val = cm_arr[r][c]
            pct = 100 * val / total
            color = "white" if val > total * 0.3 else "black"
            ax.text(
                c,
                r,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Confusion matrix saved → {save_path}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first or pass --checkpoint <path>."
        )

    print(f"\n{'=' * 55}")
    print("  TransformerSentiment — Evaluation")
    print(f"{'=' * 55}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}\n")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})

    d_model = train_args.get("d_model", 256)
    nhead = train_args.get("nhead", 8)
    num_layers = train_args.get("num_layers", 4)
    dim_feedforward = train_args.get("dim_feedforward", 1024)
    num_classes = train_args.get("num_classes", 2)
    max_len = train_args.get("max_len", 200)
    dropout = train_args.get("dropout", 0.1)
    base_path = train_args.get("base_path", args.base_path)

    print("  Recovered from checkpoint:")
    print(f"    d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    print(f"    dim_feedforward={dim_feedforward}, max_len={max_len}\n")

    train_seed = train_args.get("seed", 42)
    train_samples = train_args.get("train_samples", 2000)

    print("Rebuilding vocabulary (must match training run)...")
    _, test_loader, vocab_size = get_dataloaders(
        base_path=base_path,
        batch_size=args.batch_size,
        max_len=max_len,
        train_samples=train_samples,
        test_samples=args.test_samples,
        seed=train_seed,
    )
    print(f"  vocab_size  = {vocab_size:,}")
    print(f"  test samples= {args.test_samples * 2}\n")

    model = TransformerSentiment(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        max_len=max_len,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    trained_epoch = ckpt.get("epoch", "?")
    saved_val_acc = ckpt.get("best_val_acc", "?")
    print(
        f"  Weights loaded  (saved at epoch {trained_epoch},"
        f" best_val_acc={saved_val_acc:.4f})\n"
    )

    print("Running inference...")
    all_labels, all_preds, all_confs = run_inference(model, test_loader, device)

    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)
    cm = metrics["confusion_matrix"]

    class_names = ["Negative", "Positive"]

    print(f"\n{'=' * 55}")
    print(f"  Evaluation Results  ({len(all_labels)} samples)")
    print(f"{'=' * 55}")
    print(
        f"  Accuracy          : {metrics['accuracy']:.4f}  "
        f"({int(metrics['accuracy']*len(all_labels))}/{len(all_labels)})"
    )
    print(f"  Macro Precision   : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall      : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1          : {metrics['macro_f1']:.4f}")

    print("\n  Per-class metrics:")
    print(f"  {'Class':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-' * 44}")
    for c, name in enumerate(class_names):
        pc = metrics["per_class"][c]
        print(
            f"  {name:<12}  {pc['precision']:>10.4f}  "
            f"{pc['recall']:>8.4f}  {pc['f1']:>8.4f}"
        )

    print("\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = f"  {'':>12}" + "".join(f"  {n:>10}" for n in class_names)
    print(header)
    for r, name in enumerate(class_names):
        row_str = f"  {name:>12}" + "".join(
            f"  {cm[r][c]:>10}" for c in range(num_classes)
        )
        print(row_str)
    print(f"{'=' * 55}\n")

    plot_confusion_matrix(cm, args.confusion_plot, class_names)

    if args.save_predictions:
        with open(args.save_predictions, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "true_label",
                    "true_name",
                    "pred_label",
                    "pred_name",
                    "confidence",
                    "correct",
                ],
            )
            w.writeheader()
            for i, (true, pred, conf) in enumerate(
                zip(all_labels, all_preds, all_confs)
            ):
                w.writerow(
                    {
                        "index": i,
                        "true_label": true,
                        "true_name": class_names[true],
                        "pred_label": pred,
                        "pred_name": class_names[pred],
                        "confidence": round(conf, 4),
                        "correct": int(true == pred),
                    }
                )
        print(f"  ✓ Predictions saved → {args.save_predictions}")


if __name__ == "__main__":
    main()

