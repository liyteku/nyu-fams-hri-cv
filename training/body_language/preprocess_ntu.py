#!/usr/bin/env python3
"""
preprocess_ntu.py - Convert raw NTU RGB+D .skeleton files to preprocessed .npy arrays.

Only processes the 8 selected action classes relevant to the AI Tutor scenario.
Does NTU 25→COCO 17 joint mapping, uses only first person, pads/crops to max_frames.
Labels are stored as 0-indexed class IDs (0..num_classes-1).

Output:
  xsub_train_data.npy   (N, 3, T, 17)
  xsub_train_label.npy  (N,)
  xsub_val_data.npy
  xsub_val_label.npy

Usage:
  python training/body_language/preprocess_ntu.py
"""

import os
import sys
import subprocess
import yaml
import numpy as np
from tqdm import tqdm

# ── Constants ───────────────────────────────────────────────────────────────
NUM_JOINTS_NTU = 25
CHANNELS = 3  # x, y, z

# NTU cross-subject training subjects
TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}

# NTU 25-joint → COCO 17-joint mapping
NTU_TO_COCO_17 = [3, 3, 3, 3, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def find_skeleton_files(data_dir, action_ids):
    """Use `find` command to locate skeleton files (much faster than Python glob on WSL).

    Args:
        data_dir: Directory containing .skeleton files
        action_ids: List of 1-based NTU action IDs to include

    Returns:
        Sorted list of absolute filepaths
    """
    # Build find expression for selected actions
    name_exprs = []
    for aid in action_ids:
        name_exprs.extend(['-name', f'*A{aid:03d}.skeleton'])
        if len(name_exprs) > 2:
            name_exprs.insert(-2, '-o')

    cmd = ['find', data_dir, '-maxdepth', '1', '('] + name_exprs + [')']
    result = subprocess.run(cmd, capture_output=True, text=True)
    files = sorted([f.strip() for f in result.stdout.strip().split('\n') if f.strip()])
    return files


def parse_skeleton_file(filepath, max_frames):
    """Parse a .skeleton file, extract first person's 3D joint coordinates.

    Returns:
        (C, max_frames, 17) float32 array with COCO-mapped joints, or None
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None

    if len(lines) < 2:
        return None

    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1

    if num_frames == 0:
        return None

    raw = np.zeros((CHANNELS, max_frames, NUM_JOINTS_NTU), dtype=np.float32)

    for t in range(num_frames):
        if idx >= len(lines):
            break

        num_bodies = int(lines[idx].strip())
        idx += 1

        for m in range(num_bodies):
            if idx >= len(lines):
                break
            idx += 1  # body info line

            if idx >= len(lines):
                break
            num_joints = int(lines[idx].strip())
            idx += 1

            for v in range(num_joints):
                if idx >= len(lines):
                    break
                parts = lines[idx].strip().split()
                idx += 1

                # Only keep first person (m=0), within max_frames
                if m == 0 and v < NUM_JOINTS_NTU and t < max_frames:
                    raw[0, t, v] = float(parts[0])
                    raw[1, t, v] = float(parts[1])
                    raw[2, t, v] = float(parts[2])

    # Map NTU 25 → COCO 17
    return raw[:, :, NTU_TO_COCO_17]  # (C, max_frames, 17)


def parse_filename(filepath):
    """Extract subject ID and action ID from NTU filename.
    Format: SsssCcccPpppRrrrAaaa.skeleton
    """
    name = os.path.basename(filepath).split('.')[0]
    subject = int(name[name.index('P') + 1: name.index('P') + 4])
    action = int(name[name.index('A') + 1:])
    return subject, action


def main():
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'body_language_config.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    max_frames = cfg['data']['max_frames']
    selected_actions = cfg['data']['selected_actions']
    class_names = cfg['data']['class_names']
    action_ids = sorted(set(int(k) for k in selected_actions.keys()))

    # Build action_id → class_index mapping
    # Supports merged classes (e.g. reading+writing → "studying" both get same index)
    action_to_class = {}
    for aid_str, class_name in selected_actions.items():
        action_to_class[int(aid_str)] = class_names.index(class_name)

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'NTU-RGBD')

    print(f"{'=' * 60}")
    print(f"  NTU RGB+D Skeleton Preprocessing")
    print(f"{'=' * 60}")
    print(f"  Source     : {data_dir}")
    print(f"  Max frames : {max_frames}")
    print(f"  Actions    : {action_ids}")
    print(f"  Classes    : {class_names}")
    print(f"  Remap      : {action_to_class}")
    print()

    # Find skeleton files (fast, using system `find`)
    print("  Finding skeleton files...")
    skeleton_files = find_skeleton_files(data_dir, action_ids)
    print(f"  Found {len(skeleton_files)} files for selected actions")

    if not skeleton_files:
        print("  [ERROR] No matching skeleton files found!")
        sys.exit(1)

    # Split by cross-subject protocol
    train_files = []
    val_files = []
    for f in skeleton_files:
        subj, _ = parse_filename(f)
        if subj in TRAIN_SUBJECTS:
            train_files.append(f)
        else:
            val_files.append(f)

    print(f"  Train: {len(train_files)} files")
    print(f"  Val  : {len(val_files)} files")
    print()

    # Process each split
    for split_name, split_files in [('train', train_files), ('val', val_files)]:
        print(f"Processing {split_name} ({len(split_files)} files)...")

        all_data = []
        all_labels = []
        skipped = 0

        for filepath in tqdm(split_files, desc=f"  {split_name}", ncols=80):
            data = parse_skeleton_file(filepath, max_frames)
            if data is None:
                skipped += 1
                continue

            _, aid = parse_filename(filepath)
            all_data.append(data)
            all_labels.append(action_to_class[aid])

        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        print(f"  {split_name}: {all_data.shape[0]} samples, shape={all_data.shape}")
        if skipped:
            print(f"  Skipped: {skipped}")

        # Save
        data_out = os.path.join(data_dir, f'xsub_{split_name}_data.npy')
        label_out = os.path.join(data_dir, f'xsub_{split_name}_label.npy')
        np.save(data_out, all_data)
        np.save(label_out, all_labels)

        size_mb = os.path.getsize(data_out) / 1e6
        print(f"  Saved: {os.path.basename(data_out)} ({size_mb:.1f} MB)")
        print()

    print(f"{'=' * 60}")
    print(f"  Preprocessing complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
