"""
train.py - Training script for skeleton-based action recognition.

Run from project root:
    python training/body_language/train.py
    python training/body_language/train.py --epochs 5 --batch_size 32
"""

import argparse
import os
import sys
import time

# Resolve project root (two levels up from training/body_language/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_datasets, build_dataloaders
from model import build_model


# Config

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "body_language_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg, args):
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg


# Plotting

def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {save_path}")


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_acc"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Val Accuracy")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to {save_path}")


# Train / Evaluate

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# Main

def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN on NTU RGB+D")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    num_classes = cfg["data"]["num_classes"]
    class_names = cfg["data"]["class_names"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    warmup_epochs = cfg["train"]["warmup_epochs"]
    label_smoothing = cfg["train"].get("label_smoothing", 0.0)
    patience = cfg["train"].get("early_stopping_patience", 0)  # 0 = disabled

    # Resolve paths
    out_dir = os.path.join(PROJECT_ROOT, cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    cfg["data"]["data_dir"] = os.path.join(PROJECT_ROOT, cfg["data"]["data_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Skeleton Action Recognition - Training")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch        : {cfg['train']['batch_size']}")
    print(f"  LR           : {lr}")
    print(f"  Max frames   : {cfg['data']['max_frames']}")
    print(f"  Num classes  : {num_classes}")
    print(f"  Label smooth : {label_smoothing}")
    print(f"  Early stop   : {'patience=' + str(patience) if patience > 0 else 'disabled'}")
    print(f"  Output       : {out_dir}/")
    print(f"{'='*60}\n")

    # Data
    print("Loading datasets...")
    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_dataset, val_dataset)
    print(f"  Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples\n")

    # Model
    print("Building ST-GCN model...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}\n")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=cfg["train"]["weight_decay"]
    )

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    print(f"{'_'*60}")
    print(f"  Starting training...")
    print(f"{'_'*60}\n")

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs}  (lr={current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_preds = preds
            best_labels = labels
            epochs_no_improve = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }
            save_path = os.path.join(out_dir, cfg["output"]["best_model_name"])
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1

        marker = " * best" if is_best else ""
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%{marker}")

        # Early stopping
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n  Early stopping: no improvement for {patience} epochs.")
            break
        print()

    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Best val accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {os.path.join(out_dir, cfg['output']['best_model_name'])}")
    print(f"{'='*60}\n")

    # Final report
    print("Classification Report (best epoch):")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    cm = confusion_matrix(best_labels, best_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"))

    print("\nDone")


if __name__ == "__main__":
    main()
