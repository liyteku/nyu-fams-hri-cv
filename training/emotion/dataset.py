"""
dataset.py - RAF-DB data loading and augmentation.

Uses torchvision.datasets.ImageFolder which reads class sub-folders
(1/, 2/, ..., 7/) and maps them to indices 0-6 automatically.
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def build_train_transform(cfg: dict) -> transforms.Compose:
    """Build augmentation + normalisation pipeline for training."""
    t = cfg["transform"]
    cj = t["color_jitter"]
    af = t.get("random_affine", {})
    gb = t.get("gaussian_blur", {})

    transform_list = [
        transforms.Resize((t["img_size"], t["img_size"])),
        transforms.RandomHorizontalFlip(p=t["random_horizontal_flip"]),
        transforms.RandomRotation(degrees=t["random_rotation"]),
    ]

    # RandomAffine (translation + scale)
    if af:
        transform_list.append(transforms.RandomAffine(
            degrees=0,
            translate=tuple(af.get("translate", [0.1, 0.1])),
            scale=tuple(af.get("scale", [0.9, 1.1])),
        ))

    # ColorJitter
    transform_list.append(transforms.ColorJitter(
        brightness=cj["brightness"],
        contrast=cj["contrast"],
        saturation=cj["saturation"],
        hue=cj["hue"],
    ))

    # GaussianBlur
    if gb:
        transform_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=gb.get("kernel_size", 5))],
            p=gb.get("probability", 0.2),
        ))

    # RandomGrayscale
    grayscale_p = t.get("random_grayscale", 0.0)
    if grayscale_p > 0:
        transform_list.append(transforms.RandomGrayscale(p=grayscale_p))

    # ToTensor + Normalize + RandomErasing
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=t["mean"], std=t["std"]),
        transforms.RandomErasing(p=t["random_erasing"]),
    ])

    return transforms.Compose(transform_list)


def build_test_transform(cfg: dict) -> transforms.Compose:
    """Build resize + normalisation pipeline for evaluation."""
    t = cfg["transform"]
    return transforms.Compose([
        transforms.Resize((t["img_size"], t["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=t["mean"], std=t["std"]),
    ])


def compute_class_weights(dataset, num_classes: int, mode: str = "inverse") -> torch.Tensor:
    """Compute class weights for the loss function.

    Args:
        mode: "inverse" for standard inverse-frequency, "sqrt" for sqrt of inverse (gentler).
    """
    counter = Counter(dataset.targets)
    total = len(dataset.targets)
    weights = []
    for cls_idx in range(num_classes):
        count = counter.get(cls_idx, 1)
        w = total / (num_classes * count)
        if mode == "sqrt":
            w = w ** 0.5
        weights.append(w)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor


def build_datasets(cfg: dict):
    """Build train and test ImageFolder datasets."""
    data_cfg = cfg["data"]
    root = data_cfg["root_dir"]
    train_path = os.path.join(root, data_cfg["train_dir"])
    test_path = os.path.join(root, data_cfg["test_dir"])

    train_transform = build_train_transform(cfg)
    test_transform = build_test_transform(cfg)

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    return train_dataset, test_dataset


def build_dataloaders(cfg: dict, train_dataset, test_dataset):
    """Build DataLoaders for train and test sets."""
    t = cfg["train"]
    use_oversample = t.get("oversample", False)

    if use_oversample:
        # Assign each sample a weight = 1 / (count of its class)
        counter = Counter(train_dataset.targets)
        sample_weights = [1.0 / counter[label] for label in train_dataset.targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=t["batch_size"],
            sampler=sampler,        # sampler replaces shuffle
            num_workers=t["num_workers"],
            pin_memory=t["pin_memory"],
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=t["batch_size"],
            shuffle=True,
            num_workers=t["num_workers"],
            pin_memory=t["pin_memory"],
            drop_last=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
        pin_memory=t["pin_memory"],
    )
    return train_loader, test_loader
