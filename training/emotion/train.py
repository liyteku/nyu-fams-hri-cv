"""
train.py - Main training script for facial expression recognition.

Run from project root:
    python training/emotion/train.py
    python training/emotion/train.py --epochs 2 --batch_size 16
"""

import argparse
import copy
import csv
import os
import sys
import time

# Resolve project root (two levels up from training/emotion/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
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
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg


# Mixup / CutMix

def mixup_data(x, y, alpha=0.2):
    """Apply Mixup: blend two samples and their labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix: paste a patch from one sample onto another."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# EMA

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# Two-stage fine-tuning

def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def build_optimizer(model, cfg, is_backbone_frozen=False):
    """Build optimizer with optional separate LR for backbone."""
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    backbone_lr_scale = cfg["train"].get("backbone_lr_scale", 0.1)

    if is_backbone_frozen:
        # Only classifier params
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        # Separate param groups: backbone (lower LR) + head (full LR)
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": head_params, "lr": lr},
        ], weight_decay=wd)


# Plotting

def plot_confusion_matrix(cm, class_names, save_path):
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


# Train / Evaluate

def train_one_epoch(model, loader, criterion, optimizer, device, cfg, ema=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    mixup_alpha = cfg["train"].get("mixup_alpha", 0.0)
    cutmix_alpha = cfg["train"].get("cutmix_alpha", 0.0)
    use_mix = (mixup_alpha > 0 or cutmix_alpha > 0)

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Apply Mixup or CutMix (50/50 chance)
        if use_mix and np.random.rand() < 0.5:
            if np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, cutmix_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

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


@torch.no_grad()
def evaluate_tta(model, loader, criterion, device):
    """Evaluate with Test-Time Augmentation (horizontal flip + average)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Original prediction
        out1 = model(images)
        # Flipped prediction
        out2 = model(torch.flip(images, dims=[3]))
        # Average softmax probabilities
        avg_out = (torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2.0

        loss = criterion(out1, labels)  # loss on original only
        running_loss += loss.item() * images.size(0)

        _, predicted = avg_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# Metrics logging

def init_csv_log(path, class_names):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "test_loss", "test_acc", "lr"]
        header += [f"recall_{c}" for c in class_names]
        writer.writerow(header)


def append_csv_log(path, epoch, train_loss, test_loss, test_acc, lr, per_class_recall):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [epoch, f"{train_loss:.4f}", f"{test_loss:.4f}", f"{test_acc:.2f}", f"{lr:.6f}"]
        row += [f"{r:.4f}" for r in per_class_recall]
        writer.writerow(row)


def compute_per_class_recall(preds, labels, num_classes):
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    recalls = []
    for i in range(num_classes):
        total = cm[i].sum()
        recalls.append(cm[i, i] / total if total > 0 else 0.0)
    return recalls


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
    warmup_epochs = cfg["train"]["warmup_epochs"]
    freeze_epochs = cfg["train"].get("freeze_backbone_epochs", 0)
    ema_decay = cfg["train"].get("ema_decay", 0.0)
    label_smoothing = cfg["train"].get("label_smoothing", 0.0)
    use_tta = cfg["train"].get("tta", False)
    use_oversample = cfg["train"].get("oversample", False)

    # Resolve paths relative to project root
    out_dir = os.path.join(PROJECT_ROOT, cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    cfg["data"]["root_dir"] = os.path.join(PROJECT_ROOT, cfg["data"]["root_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Facial Expression Recognition - Training")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  Model         : {cfg['model']['name']}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch         : {cfg['train']['batch_size']}")
    print(f"  LR            : {lr}")
    print(f"  Label smooth  : {label_smoothing}")
    print(f"  Freeze epochs : {freeze_epochs}")
    print(f"  EMA decay     : {ema_decay}")
    print(f"  Mixup/CutMix  : {cfg['train'].get('mixup_alpha', 0)}/{cfg['train'].get('cutmix_alpha', 0)}")
    print(f"  Oversample    : {use_oversample}")
    print(f"  TTA           : {use_tta}")
    print(f"  Output        : {out_dir}/")
    print(f"{'='*60}\n")

    # Data
    print("Loading datasets...")
    train_dataset, test_dataset = build_datasets(cfg)
    train_loader, test_loader = build_dataloaders(cfg, train_dataset, test_dataset)
    print(f"  Train: {len(train_dataset)} images  |  Test: {len(test_dataset)} images")

    # Class weights
    weight_mode = cfg["train"].get("weight_mode", "inverse")
    if cfg["train"]["use_weighted_loss"]:
        weights = compute_class_weights(train_dataset, num_classes, mode=weight_mode).to(device)
        print(f"  Class weights ({weight_mode}): {[f'{w:.2f}' for w in weights.tolist()]}")
    else:
        weights = None

    # Model
    print(f"\nBuilding model: {cfg['model']['name']}...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Stage 1: freeze backbone
    if freeze_epochs > 0:
        freeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Stage 1: backbone frozen, trainable params: {trainable:,}")

    # EMA
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema:
        print(f"  EMA enabled (decay={ema_decay})")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer = build_optimizer(model, cfg, is_backbone_frozen=(freeze_epochs > 0))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Metrics log
    csv_path = os.path.join(out_dir, "training_log.csv")
    init_csv_log(csv_path, class_names)

    # Training loop
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    start_time = time.time()

    print(f"\n{'_'*60}")
    print(f"  Starting training...")
    print(f"{'_'*60}\n")

    for epoch in range(1, epochs + 1):

        # Stage 2: unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            print(f"\n  >> Unfreezing backbone (epoch {epoch})")
            unfreeze_all(model)
            optimizer = build_optimizer(model, cfg, is_backbone_frozen=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # Warmup scheduler to current epoch (use last_epoch to avoid warning)
            scheduler.last_epoch = epoch - 1
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  >> Trainable params: {trainable:,}\n")

        current_lr = optimizer.param_groups[0]["lr"]
        stage = "S1-frozen" if epoch <= freeze_epochs else "S2-full"
        print(f"Epoch {epoch}/{epochs}  [{stage}]  (lr={current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg, ema=ema
        )

        # Evaluate (with TTA if enabled)
        eval_fn = evaluate_tta if use_tta else evaluate
        test_loss, test_acc, preds, labels = eval_fn(
            model, test_loader, criterion, device
        )
        scheduler.step()

        # Per-class recall
        per_class_recall = compute_per_class_recall(preds, labels, num_classes)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        append_csv_log(csv_path, epoch, train_loss, test_loss, test_acc, current_lr, per_class_recall)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_preds = preds
            best_labels = labels
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (ema.state_dict() if ema else model.state_dict()),
                "best_acc": best_acc,
                "config": cfg,
            }
            save_path = os.path.join(out_dir, cfg["output"]["best_model_name"])
            torch.save(checkpoint, save_path)

        marker = " * best" if is_best else ""
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%{marker}")
        # Print worst-performing class
        worst_idx = int(np.argmin(per_class_recall))
        print(f"  Worst class: {class_names[worst_idx]} ({per_class_recall[worst_idx]*100:.1f}% recall)\n")

    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Best test accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {os.path.join(out_dir, cfg['output']['best_model_name'])}")
    print(f"  Metrics log: {csv_path}")
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
