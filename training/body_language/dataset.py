"""
dataset.py - NTU RGB+D skeleton data loading with bone/motion features.

Expected data layout (from preprocess_ntu.py):
  data/NTU-RGBD/
    xsub_train_data.npy    # (N, 3, T, 17) — already COCO-mapped, first person
    xsub_train_label.npy   # (N,)  0-indexed class IDs
    xsub_val_data.npy
    xsub_val_label.npy

Features computed per sample:
  - Joint coords (3ch): x, y, z
  - Bone vectors (3ch): child - parent for each joint
  - Motion (3ch): frame[t] - frame[t-1]
  → Total: 9 channels input to model
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# COCO 17-joint parent mapping (each joint's parent in the skeleton tree)
# Root joints (no parent) point to themselves
COCO_PARENT = [
    0,   # 0  nose        → self (root)
    0,   # 1  left_eye    → nose
    0,   # 2  right_eye   → nose
    1,   # 3  left_ear    → left_eye
    2,   # 4  right_ear   → right_eye
    0,   # 5  left_shoulder  → nose (via neck)
    0,   # 6  right_shoulder → nose (via neck)
    5,   # 7  left_elbow     → left_shoulder
    6,   # 8  right_elbow    → right_shoulder
    7,   # 9  left_wrist     → left_elbow
    8,   # 10 right_wrist    → right_elbow
    5,   # 11 left_hip       → left_shoulder
    6,   # 12 right_hip      → right_shoulder
    11,  # 13 left_knee      → left_hip
    12,  # 14 right_knee     → right_hip
    13,  # 15 left_ankle     → left_knee
    14,  # 16 right_ankle    → right_knee
]


class NTUSkeletonDataset(Dataset):
    """NTU RGB+D skeleton dataset (preprocessed, COCO-mapped)."""

    def __init__(self, data_path, label_path, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.max_frames = cfg["data"]["max_frames"]
        self.use_bone_motion = cfg["model"].get("in_channels", 3) == 9

        # Load preprocessed data — already filtered, mapped, cropped
        print(f"  Loading {data_path}...")
        self.data = np.load(data_path)           # (N, 3, T, 17)
        self.labels = np.load(label_path).flatten()  # (N,)

        # Augmentation config
        aug = cfg.get("augmentation", {})
        self.random_rotate = aug.get("random_rotate", 0.0)
        self.random_scale = aug.get("random_scale", [1.0, 1.0])
        self.random_shift = aug.get("random_shift", 0.0)
        self.temporal_crop = aug.get("temporal_crop", False)

        print(f"  Loaded {len(self)} samples, shape={self.data.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx].copy().astype(np.float32)  # (3, T, V)
        label = self.labels[idx]

        # Temporal processing
        data = self._temporal_sample(data)

        # Augmentation (train only)
        if self.is_train:
            data = self._augment(data)

        # Compute bone + motion features
        if self.use_bone_motion:
            data = self._compute_features(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _compute_features(self, data):
        """Compute bone vectors and motion features. data: (3, T, V) → (9, T, V)"""
        C, T, V = data.shape

        # Bone: child - parent
        bone = np.zeros_like(data)
        for v in range(V):
            parent = COCO_PARENT[v]
            bone[:, :, v] = data[:, :, v] - data[:, :, parent]

        # Motion: temporal difference
        motion = np.zeros_like(data)
        motion[:, 1:, :] = data[:, 1:, :] - data[:, :-1, :]

        return np.concatenate([data, bone, motion], axis=0)  # (9, T, V)

    def _temporal_sample(self, data):
        """Crop or pad to max_frames. data: (C, T, V)"""
        C, T, V = data.shape
        max_t = self.max_frames

        # Find actual sequence length (non-zero frames)
        frame_energy = np.sum(np.abs(data), axis=(0, 2))  # (T,)
        valid_frames = np.where(frame_energy > 1e-6)[0]

        if len(valid_frames) == 0:
            return np.zeros((C, max_t, V), dtype=np.float32)

        actual_len = valid_frames[-1] + 1
        data = data[:, :actual_len, :]

        if actual_len >= max_t:
            if self.is_train and self.temporal_crop:
                start = np.random.randint(0, actual_len - max_t + 1)
            else:
                start = 0
            data = data[:, start:start + max_t, :]
        else:
            # Pad with zeros
            pad = np.zeros((C, max_t - actual_len, V), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)

        return data

    def _augment(self, data):
        """Apply spatial augmentations. data: (C, T, V)"""
        # Random rotation (around z-axis, i.e. in x-y plane)
        if self.random_rotate > 0:
            angle = np.random.uniform(-self.random_rotate, self.random_rotate)
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            x = data[0].copy()
            y = data[1].copy()
            data[0] = cos_a * x - sin_a * y
            data[1] = sin_a * x + cos_a * y

        # Random scale
        lo, hi = self.random_scale
        if lo != hi:
            scale = np.random.uniform(lo, hi)
            data[:2] *= scale

        # Random shift
        if self.random_shift > 0:
            shift = np.random.uniform(-self.random_shift, self.random_shift, size=(2, 1, 1))
            data[:2] += shift

        return data


def build_datasets(cfg):
    """Build train and val datasets from preprocessed NTU .npy files."""
    data_dir = cfg["data"]["data_dir"]

    train_data_path = os.path.join(data_dir, "xsub_train_data.npy")
    train_label_path = os.path.join(data_dir, "xsub_train_label.npy")
    val_data_path = os.path.join(data_dir, "xsub_val_data.npy")
    val_label_path = os.path.join(data_dir, "xsub_val_label.npy")

    train_dataset = NTUSkeletonDataset(train_data_path, train_label_path, cfg, is_train=True)
    val_dataset = NTUSkeletonDataset(val_data_path, val_label_path, cfg, is_train=False)

    return train_dataset, val_dataset


def build_dataloaders(cfg, train_dataset, val_dataset):
    """Build DataLoaders."""
    t = cfg["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=t["num_workers"],
        pin_memory=t["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
        pin_memory=t["pin_memory"],
    )
    return train_loader, val_loader
