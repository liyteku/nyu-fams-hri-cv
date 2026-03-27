"""
model.py - Enhanced ST-GCN for skeleton-based action recognition.

Features:
  - 9-channel input (joint + bone + motion features)
  - Adaptive adjacency (learnable joint connections beyond skeleton topology)
  - Spatio-temporal attention before final pooling

Input : (B, C, T, V)  where C=9, T=frames, V=17 joints
Output: (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# COCO 17-joint skeleton graph

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Natural bone connections (undirected edges)
COCO_EDGES = [
    (0, 1), (0, 2),      # nose - eyes
    (1, 3), (2, 4),      # eyes - ears
    (0, 5), (0, 6),      # nose - shoulders (via neck approx)
    (5, 7), (7, 9),      # left arm: shoulder-elbow-wrist
    (6, 8), (8, 10),     # right arm: shoulder-elbow-wrist
    (5, 11), (6, 12),    # shoulders - hips
    (11, 13), (13, 15),  # left leg: hip-knee-ankle
    (12, 14), (14, 16),  # right leg: hip-knee-ankle
    (5, 6),              # shoulder-shoulder
    (11, 12),            # hip-hip
]

NUM_JOINTS = 17


def build_adjacency_matrix():
    """Build normalized adjacency matrix for COCO 17-joint skeleton."""
    num_v = NUM_JOINTS
    A_self = np.eye(num_v, dtype=np.float32)

    A_full = np.zeros((num_v, num_v), dtype=np.float32)
    for (i, j) in COCO_EDGES:
        A_full[i, j] = 1.0
        A_full[j, i] = 1.0

    # BFS distance from root (nose)
    center = 0
    dist = np.full(num_v, np.inf)
    dist[center] = 0
    queue = [center]
    while queue:
        node = queue.pop(0)
        for (i, j) in COCO_EDGES:
            neighbor = j if i == node else (i if j == node else None)
            if neighbor is not None and dist[neighbor] == np.inf:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)

    A_centripetal = np.zeros((num_v, num_v), dtype=np.float32)
    A_centrifugal = np.zeros((num_v, num_v), dtype=np.float32)

    for (i, j) in COCO_EDGES:
        if dist[i] < dist[j]:
            A_centripetal[j, i] = 1.0
            A_centrifugal[i, j] = 1.0
        elif dist[j] < dist[i]:
            A_centripetal[i, j] = 1.0
            A_centrifugal[j, i] = 1.0
        else:
            A_centripetal[i, j] = 1.0
            A_centripetal[j, i] = 1.0

    A = np.stack([A_self, A_centripetal, A_centrifugal], axis=0)

    # Normalize: D^{-1/2} A D^{-1/2}
    for k in range(3):
        D = np.sum(A[k], axis=1)
        D_inv_sqrt = np.zeros_like(D)
        nonzero = D > 0
        D_inv_sqrt[nonzero] = D[nonzero] ** (-0.5)
        A[k] = A[k] * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]

    return A  # (3, 17, 17)


class GraphConvolution(nn.Module):
    """Spatial graph convolution with learnable edge importance + adaptive adjacency."""

    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        num_subsets = A.shape[0]  # 3

        self.num_subsets = num_subsets
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(num_subsets)
        ])

        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones_like(self.A[k]))
            for k in range(num_subsets)
        ])

        # Adaptive adjacency: fully learnable joint connections (initialized to 0)
        num_v = A.shape[1]
        self.adaptive_adj = nn.Parameter(torch.zeros(num_v, num_v))

        self.bn = nn.BatchNorm2d(out_channels)

        if residual and in_channels == out_channels:
            self.residual = nn.Identity()
        elif residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = None

    def forward(self, x):
        """x: (B, C, T, V)"""
        res = self.residual(x) if self.residual is not None else 0

        out = 0
        for k in range(self.num_subsets):
            # Fixed topology + learned importance + adaptive connections
            A_k = self.A[k] * self.edge_importance[k] + self.adaptive_adj
            z = torch.einsum("bctv,vw->bctw", x, A_k)
            out = out + self.convs[k](z)

        out = self.bn(out)
        out = F.relu(out + res)
        return out


class STGCNBlock(nn.Module):
    """One ST-GCN block = spatial graph conv + temporal conv."""

    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A)

        # Temporal convolution (single kernel=9)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res
        return F.relu(x)


class SpatioTemporalAttention(nn.Module):
    """Lightweight spatio-temporal attention."""

    def __init__(self, channels, num_joints=17, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)

        # Spatial attention: per-joint importance
        self.spatial_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, num_joints)),
            nn.Flatten(1),
            nn.Linear(channels * num_joints, mid),
            nn.ReLU(),
            nn.Linear(mid, num_joints),
            nn.Sigmoid(),
        )

        # Temporal attention: per-frame importance
        self.temporal_fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.temporal_fc2 = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, x):
        B, C, T, V = x.shape

        # Spatial attention
        s_attn = self.spatial_attn(x).view(B, 1, 1, V)
        x = x * s_attn

        # Temporal attention
        t_feat = x.mean(dim=3, keepdim=True)
        t_attn = torch.sigmoid(self.temporal_fc2(F.relu(self.temporal_fc1(t_feat))))
        x = x * t_attn

        return x


class STGCN(nn.Module):
    """Enhanced ST-GCN with adaptive adjacency and attention.

    Architecture: input → data_bn → 6 ST-GCN blocks → attention → pool → FC
    """

    def __init__(self, num_classes, in_channels=9, dropout=0.2):
        super().__init__()
        A = build_adjacency_matrix()  # (3, 17, 17)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 128, A, stride=2, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout),
            STGCNBlock(256, 256, A, stride=1, dropout=dropout),
        ])

        self.attention = SpatioTemporalAttention(256, NUM_JOINTS)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape

        # Batch norm over spatial dims
        x = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            x = block(x)

        # Attention
        x = self.attention(x)

        # Global average pooling over time and joints
        x = x.mean(dim=[2, 3])  # (B, 256)
        x = self.fc(x)
        return x


def build_model(cfg):
    """Build ST-GCN from config."""
    num_classes = cfg["data"]["num_classes"]
    in_channels = cfg["model"].get("in_channels", 9)
    dropout = cfg["model"].get("dropout", 0.2)
    return STGCN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
