"""
Body pose detection module using YOLO11-pose via the Ultralytics API.

This mirrors the design of the modules in src/face:
- Input:  a single BGR frame (NumPy array, HxWx3, uint8)
- Output: a list of dictionaries, one per detected person, containing
  bounding boxes and COCO-style keypoints in pixel coordinates.
"""

from typing import Dict, List, Optional

import os
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover - import-time guard
    YOLO = None
    _import_error = e
else:
    _import_error = None


class BodyPoseDetector:
    """
    Wrapper around a YOLO11-pose model.

    Example:
        detector = BodyPoseDetector("yolo11n-pose.pt")
        results = detector.detect_pose(frame)
    """

    def __init__(
        self,
        model_path: str = os.path.join("outputs", "yolo11n-pose.pt"),
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: Path or name of a YOLO11-pose weights file (pt/onnx/engine).
            device: Device string understood by Ultralytics, e.g. "cpu", "cuda:0", "mps".
                    If None, Ultralytics will select a device automatically.
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics is not available. Install it via 'pip install ultralytics'."
            ) from _import_error

        self.model = YOLO(model_path)
        self.device = device

    def detect_pose(
        self,
        frame: np.ndarray,
        conf: float = 0.5,
        max_detections: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run pose detection on a single BGR frame.

        Args:
            frame: BGR image as a NumPy array (H, W, 3), uint8.
            conf: Minimum confidence threshold for detections.
            max_detections: Optional maximum number of people to return
                            (highest scoring first).

        Returns:
            List of dicts, one per detected person:
                {
                    "box": [x1, y1, x2, y2],
                    "score": float,
                    "keypoints": [
                        {"x": float, "y": float, "conf": float},
                        ...
                    ],
                }
        """
        if frame is None or frame.size == 0:
            return []

        # Ultralytics YOLO can take a single NumPy image as source.
        results = self.model.predict(
            source=frame,
            conf=conf,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        res = results[0]
        if res.boxes is None or res.keypoints is None:
            return []

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = res.boxes.conf.cpu().numpy()  # (N,)

        # keypoints.xyn gives normalised coordinates in [0, 1] relative to width/height
        # Shape: (N, K, 2)
        if hasattr(res.keypoints, "xyn"):
            kpts = res.keypoints.xyn.cpu().numpy()
            kpts_conf = None
        else:
            # Fallback: full tensor (N, K, 3) with (x, y, conf) in pixels
            data = res.keypoints.data.cpu().numpy()  # (N, K, 3)
            kpts = data[..., :2]
            kpts_conf = data[..., 2]

        h, w = frame.shape[:2]

        indices = np.argsort(-scores)
        if max_detections is not None:
            indices = indices[:max_detections]

        outputs: List[Dict] = []

        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
            score = float(scores[idx])

            kp_xy = kpts[idx]  # (K, 2)
            keypoints: List[Dict] = []
            for j, (kx, ky) in enumerate(kp_xy):
                if kpts_conf is not None:
                    conf_j = float(kpts_conf[idx][j])
                    x_px = float(kx)
                    y_px = float(ky)
                else:
                    conf_j = 1.0
                    x_px = float(kx * w)
                    y_px = float(ky * h)

                keypoints.append({"x": x_px, "y": y_px, "conf": conf_j})

            outputs.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "keypoints": keypoints,
                }
            )

        return outputs


