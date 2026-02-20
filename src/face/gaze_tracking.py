"""
Gaze Tracking Module

Uses MediaPipe Tasks FaceLandmarker API (mediapipe >= 0.10).
Detects eye gaze direction via iris landmarks and determines
whether the student is looking at the screen or is distracted.

Requires face_landmarker.task in this directory or pass model_path.
Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional
from collections import deque

# Default: src/face/face_landmarker.task
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task",
)

# Iris landmark indices (same numbering as old FaceMesh with refine_landmarks=True)
_RIGHT_IRIS_CENTER = 468   # indices 468-472 = right iris ring + center
_LEFT_IRIS_CENTER  = 473   # indices 473-477 = left  iris ring + center

# Eye corner landmarks
_RIGHT_EYE_INNER = 133
_RIGHT_EYE_OUTER = 33
_LEFT_EYE_INNER  = 362
_LEFT_EYE_OUTER  = 263

# Vertical landmarks
_RIGHT_EYE_TOP    = 159
_RIGHT_EYE_BOTTOM = 145
_LEFT_EYE_TOP     = 386
_LEFT_EYE_BOTTOM  = 374


class GazeTracker:
    """Tracks eye gaze direction using iris landmarks (MediaPipe Tasks API)."""

    def __init__(self, smoothing_window: int = 5, model_path: Optional[str] = None):
        self.smoothing_window = smoothing_window
        self.gaze_history: deque = deque(maxlen=smoothing_window)

        self.LOOKING_CENTER_THRESHOLD = 0.15
        self.LOOKING_LEFT_THRESHOLD   = 0.25
        self.LOOKING_RIGHT_THRESHOLD  = 0.25
        self.LOOKING_UP_THRESHOLD     = 0.20
        self.LOOKING_DOWN_THRESHOLD   = 0.25

        self.mediapipe_available = False
        self._landmarker = None

        self._load_landmarker(model_path or _DEFAULT_MODEL_PATH)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_landmarker(self, model_path: str) -> None:
        try:
            import mediapipe as mp

            if not os.path.exists(model_path):
                print(f"GazeTracker: model not found at {model_path}")
                print("Download face_landmarker.task from MediaPipe and place in src/face/")
                return

            BaseOptions          = mp.tasks.BaseOptions
            FaceLandmarker       = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            RunningMode          = mp.tasks.vision.RunningMode

            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            self._mp = mp
            self.mediapipe_available = True
            print("GazeTracker: FaceLandmarker (Tasks API) initialized")
        except Exception as e:
            print(f"GazeTracker: failed to load ({e}). Gaze tracking disabled.")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_gaze(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect gaze direction from a BGR frame."""
        if not self.mediapipe_available or self._landmarker is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )

        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        # Convert normalised landmarks to pixel coords
        raw = result.face_landmarks[0]
        lm = np.array([[p.x * w, p.y * h] for p in raw])

        # ── Iris centres ──────────────────────────────────────────────────
        if len(lm) <= _LEFT_IRIS_CENTER:
            return None  # model didn't return iris landmarks

        left_iris  = lm[_LEFT_IRIS_CENTER]
        right_iris = lm[_RIGHT_IRIS_CENTER]

        # ── Horizontal gaze ───────────────────────────────────────────────
        re_inner = lm[_RIGHT_EYE_INNER]
        re_outer = lm[_RIGHT_EYE_OUTER]
        re_width = np.linalg.norm(re_outer - re_inner)
        right_h_ratio = (
            np.linalg.norm(right_iris - re_inner) / re_width if re_width > 0 else 0.5
        )

        le_inner = lm[_LEFT_EYE_INNER]
        le_outer = lm[_LEFT_EYE_OUTER]
        le_width = np.linalg.norm(le_outer - le_inner)
        left_h_ratio = (
            np.linalg.norm(left_iris - le_inner) / le_width if le_width > 0 else 0.5
        )

        horizontal_ratio = (right_h_ratio + left_h_ratio) / 2.0

        if horizontal_ratio < (0.5 - self.LOOKING_RIGHT_THRESHOLD):
            horizontal_direction = "right"
        elif horizontal_ratio > (0.5 + self.LOOKING_LEFT_THRESHOLD):
            horizontal_direction = "left"
        else:
            horizontal_direction = "center"

        # ── Vertical gaze ─────────────────────────────────────────────────
        re_top    = lm[_RIGHT_EYE_TOP]
        re_bottom = lm[_RIGHT_EYE_BOTTOM]
        re_height = np.linalg.norm(re_top - re_bottom)
        right_v_ratio = (
            np.linalg.norm(right_iris - re_bottom) / re_height if re_height > 0 else 0.5
        )

        le_top    = lm[_LEFT_EYE_TOP]
        le_bottom = lm[_LEFT_EYE_BOTTOM]
        le_height = np.linalg.norm(le_top - le_bottom)
        left_v_ratio = (
            np.linalg.norm(left_iris - le_bottom) / le_height if le_height > 0 else 0.5
        )

        vertical_ratio = (right_v_ratio + left_v_ratio) / 2.0

        if vertical_ratio < (0.5 - self.LOOKING_DOWN_THRESHOLD):
            vertical_direction = "down"
        elif vertical_ratio > (0.5 + self.LOOKING_UP_THRESHOLD):
            vertical_direction = "up"
        else:
            vertical_direction = "center"

        # ── Attention score ───────────────────────────────────────────────
        h_dev = abs(horizontal_ratio - 0.5)
        v_dev = abs(vertical_ratio - 0.5)
        attention_score = max(0.0, 100.0 - (h_dev + v_dev) * 200.0)

        is_looking_at_screen = (
            horizontal_direction == "center" and vertical_direction == "center"
        )

        self.gaze_history.append(
            {
                "horizontal": horizontal_direction,
                "vertical": vertical_direction,
                "looking_at_screen": is_looking_at_screen,
            }
        )

        return {
            "horizontal_direction":      horizontal_direction,
            "vertical_direction":        vertical_direction,
            "gaze_direction":            f"{vertical_direction}_{horizontal_direction}",
            "is_looking_at_screen":      is_looking_at_screen,
            "smoothed_looking_at_screen": self._get_smoothed_attention(),
            "attention_score":           attention_score,
            "horizontal_ratio":          horizontal_ratio,
            "vertical_ratio":            vertical_ratio,
            "iris_positions": {
                "left":  left_iris.tolist(),
                "right": right_iris.tolist(),
            },
        }

    def get_attention_recommendation(self, gaze_data: Dict) -> str:
        if not gaze_data:
            return ""

        is_looking   = gaze_data.get("smoothed_looking_at_screen", False)
        score        = gaze_data.get("attention_score", 0)
        h_dir        = gaze_data.get("horizontal_direction", "center")
        v_dir        = gaze_data.get("vertical_direction", "center")

        if is_looking and score > 70:
            return "Student is attentive and focused on the content."
        elif score < 30:
            return "Student appears distracted - not looking at the screen."
        elif h_dir in ("left", "right"):
            return f"Student is looking {h_dir} - may be distracted."
        elif v_dir == "down":
            return "Student looking down - may be taking notes or using phone."
        elif v_dir == "up":
            return "Student looking up - may be thinking or distracted."
        return ""

    def draw_gaze_overlay(
        self, frame: np.ndarray, gaze_data: Optional[Dict]
    ) -> np.ndarray:
        if not gaze_data:
            return frame

        h_dir   = gaze_data.get("horizontal_direction", "unknown")
        v_dir   = gaze_data.get("vertical_direction", "unknown")
        looking = gaze_data.get("smoothed_looking_at_screen", False)
        score   = gaze_data.get("attention_score", 0)

        if looking:
            color, status = (0, 255, 0), "FOCUSED"
        elif score > 50:
            color, status = (0, 255, 255), "PARTIAL ATTENTION"
        else:
            color, status = (0, 0, 255), "DISTRACTED"

        y = 220
        cv2.putText(frame, f"Gaze: {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
        cv2.putText(frame, f"Direction: {v_dir.upper()}-{h_dir.upper()}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Attention: {score:.1f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y += 30
        bar_w = int(score * 2)
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 15), color, -1)
        cv2.rectangle(frame, (10, y), (210, y + 15), (100, 100, 100), 2)

        for key in ("left", "right"):
            pos = gaze_data.get("iris_positions", {}).get(key)
            if pos:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, (0, 255, 255), -1)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_smoothed_attention(self) -> bool:
        if not self.gaze_history:
            return False
        count = sum(1 for g in self.gaze_history if g["looking_at_screen"])
        return count > len(self.gaze_history) / 2

    def __del__(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass


# ── Standalone test ───────────────────────────────────────────────────────────

def test_gaze_tracking():
    tracker = GazeTracker()
    if not tracker.mediapipe_available:
        print("Gaze tracking unavailable.")
        return

    cap = cv2.VideoCapture(0)
    print("Testing gaze tracking. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gaze_data = tracker.detect_gaze(frame)
        frame = tracker.draw_gaze_overlay(frame, gaze_data)

        if gaze_data:
            rec = tracker.get_attention_recommendation(gaze_data)
            if rec:
                print(f"Recommendation: {rec}")

        cv2.imshow("Gaze Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_gaze_tracking()
