"""
infer.py - Inference interface for skeleton action recognition.

Usage:
    # From JSON file (YOLO-pose output):
    python training/body_language/infer.py --json data/sample_pose_output.json

    # As a Python module (for integration):
    from infer import ActionPredictor
    predictor = ActionPredictor("outputs/body_language/best_action_model.pth")
    predictor.push_frame(keypoints)  # (17, 3) numpy array
    action = predictor.predict()     # returns class name or None
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model import STGCN


class ActionPredictor:
    """Real-time action predictor using a sliding window of skeleton frames."""

    def __init__(self, model_path, device=None, window_size=64):
        """
        Args:
            model_path: path to saved .pth checkpoint
            device: torch device (auto-detect if None)
            window_size: number of frames to buffer before prediction
        """
        self.window_size = window_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        self.class_names = cfg["data"]["class_names"]
        num_classes = cfg["data"]["num_classes"]
        in_channels = cfg["model"].get("in_channels", 3)

        # Build and load model
        self.model = STGCN(num_classes=num_classes, in_channels=in_channels)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Frame buffer: list of (17, 3) arrays
        self.buffer = []

        print(f"ActionPredictor loaded (acc={checkpoint['best_acc']:.2f}%, "
              f"classes={self.class_names})")

    def push_frame(self, keypoints):
        """Add a frame of keypoints to the buffer.

        Args:
            keypoints: (17, 3) numpy array — [x, y, confidence] per joint
        """
        assert keypoints.shape == (17, 3), f"Expected (17, 3), got {keypoints.shape}"
        self.buffer.append(keypoints.copy())

        # Keep only the latest window_size frames
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def is_ready(self):
        """Check if enough frames have been buffered for prediction."""
        return len(self.buffer) >= self.window_size

    @torch.no_grad()
    def predict(self):
        """Run prediction on the buffered frames.

        Returns:
            dict with 'action' (str), 'confidence' (float), 'all_probs' (dict)
            or None if buffer is not full yet
        """
        if not self.is_ready():
            return None

        # Stack buffer: (window_size, 17, 3) → (3, window_size, 17)
        frames = np.stack(self.buffer, axis=0)            # (T, V, C)
        frames = frames.transpose(2, 0, 1)                # (C, T, V)
        tensor = torch.tensor(frames, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).to(self.device)      # (1, C, T, V)

        # Forward pass
        logits = self.model(tensor)                       # (1, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "action": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {name: float(p) for name, p in zip(self.class_names, probs)},
        }

    def reset(self):
        """Clear the frame buffer."""
        self.buffer = []


def predict_from_json(json_path, model_path, window_size=64):
    """Run prediction on a YOLO-pose JSON file.

    Args:
        json_path: path to JSON with YOLO-pose skeleton data
        model_path: path to trained model checkpoint
        window_size: sliding window size
    """
    predictor = ActionPredictor(model_path, window_size=window_size)

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    print(f"\nProcessing {len(frames)} frames from {json_path}...")

    for frame_data in frames:
        frame_id = frame_data["frame_id"]
        persons = frame_data.get("persons", [])

        if not persons:
            continue

        # Use first person
        person = persons[0]
        kps = np.array(person["keypoints"], dtype=np.float32)  # (17, 3)
        predictor.push_frame(kps)

        if predictor.is_ready():
            result = predictor.predict()
            print(f"  Frame {frame_id}: {result['action']} "
                  f"(conf={result['confidence']:.3f})")

    # If not enough frames for a full window, predict with what we have
    if not predictor.is_ready():
        print(f"\n  Only {len(predictor.buffer)}/{window_size} frames buffered. "
              f"Need more frames for prediction.")


def main():
    parser = argparse.ArgumentParser(description="Skeleton action inference")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to YOLO-pose JSON output")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--window", type=int, default=64,
                        help="Sliding window size (frames)")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "outputs", "body_language",
                                  "best_action_model.pth")

    predict_from_json(args.json, args.model, args.window)


if __name__ == "__main__":
    main()
