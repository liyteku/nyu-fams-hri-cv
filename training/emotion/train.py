"""
train.py - Main training script for facial expression recognition.

Run from project root:
    python training/emotion/train.py
    python training/emotion/train.py --epochs 2 --batch_size 16
"""

import argparse
import os
import sys
import time

# Resolve project root (two levels up from training/emotion/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg") # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import build_datasets, build_dataloaders, compute_class_weights
from model import build_model



# Helpers

def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "train_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments when provided."""
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg


def plot_confusion_matrix(cm, class_names, save_path):
    """Save a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """Save loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["test_loss"], label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["test_acc"], label="Test Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {save_path}")


#  Train / Evaluate one epoch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Train FER on RAF-DB")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    num_classes = cfg["data"]["num_classes"]
    class_names = cfg["data"]["class_names"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    warmup_epochs = cfg["train"]["warmup_epochs"]

    # Resolve paths relative to project root
    out_dir = os.path.join(PROJECT_ROOT, cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    cfg["data"]["root_dir"] = os.path.join(PROJECT_ROOT, cfg["data"]["root_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Facial Expression Recognition — Training")
    print(f"{'='*60}")
    print(f"  Device   : {device}")
    print(f"  Model    : {cfg['model']['name']}")
    print(f"  Epochs   : {epochs}")
    print(f"  Batch    : {cfg['train']['batch_size']}")
    print(f"  LR       : {lr}")
    print(f"  Output   : {out_dir}/")
    print(f"{'='*60}\n")

    # Data
    print("Loading datasets...")
    train_dataset, test_dataset = build_datasets(cfg)
    train_loader, test_loader = build_dataloaders(cfg, train_dataset, test_dataset)
    print(f"  Train: {len(train_dataset)} images  |  Test: {len(test_dataset)} images")

    # Class weights
    if cfg["train"]["use_weighted_loss"]:
        weights = compute_class_weights(train_dataset, num_classes).to(device)
        print(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
    else:
        weights = None

    # Model
    print(f"\nBuilding model: {cfg['model']['name']}...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    start_time = time.time()

    print(f"\n{'─'*60}")
    print(f"  Starting training...")
    print(f"{'─'*60}\n")

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs}  (lr={current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, preds, labels = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_preds = preds
            best_labels = labels
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }
            save_path = os.path.join(out_dir, cfg["output"]["best_model_name"])
            torch.save(checkpoint, save_path)

        marker = " * best" if is_best else ""
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%{marker}\n")

    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Best test accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {os.path.join(out_dir, cfg['output']['best_model_name'])}")
    print(f"{'='*60}\n")

    # Final evaluation
    print("Classification Report (best epoch):")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    cm = confusion_matrix(best_labels, best_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"))

    print("\nDone")


if __name__ == "__main__":
    main()
