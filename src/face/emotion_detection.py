"""
Emotion Detection Module

Primary backend: custom MobileNetV3 model (best_model.pth, trained on RAF-DB).
Looks for outputs/best_model.pth or best_model.pth in project root.
"""

import cv2
import numpy as np
import os
import time
from typing import Dict, Optional
from collections import deque


def _default_emotion_model_path() -> str:
    """Default: project_root/outputs/best_model.pth or project_root/best_model.pth"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in (os.path.join("outputs", "best_model.pth"), "best_model.pth"):
        path = os.path.join(project_root, name)
        if os.path.exists(path):
            return path
    return os.path.join(project_root, "outputs", "best_model.pth")


class EmotionDetector:
    """
    Detects facial emotions.

    Primary: custom MobileNetV3 loaded from best_model.pth (trained via training/emotion).
    """

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Map RAF-DB class names (lowercase) → canonical names used by this module
    _MODEL_LABEL_MAP = {
        "surprise":  "surprise",
        "fear":      "fear",
        "disgust":   "disgust",
        "happiness": "happy",
        "sadness":   "sad",
        "anger":     "angry",
        "neutral":   "neutral",
    }

    def __init__(self, smoothing_window: int = 5, model_path: Optional[str] = None):
        """
        Args:
            smoothing_window: frames used for majority-vote smoothing
            model_path: path to best_model.pth; defaults to outputs/best_model.pth or best_model.pth
        """
        self.smoothing_window = smoothing_window
        self.emotion_history: deque = deque(maxlen=smoothing_window)

        self.last_detection_time = 0.0
        self.detection_interval = 0.5  # seconds between inference calls
        self.last_emotion: Optional[Dict] = None

        # ── Primary: custom model ─────────────────────────────────────────
        self.custom_model = None
        self.custom_model_available = False
        self._class_names: list = []
        self._img_size = 224
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._torch = None
        self._device = None

        self._load_custom_model(model_path or _default_emotion_model_path())

        # Face detector used when running custom model (lightweight)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_custom_model(self, model_path: str) -> None:
        try:
            import torch
            import timm

            if not os.path.exists(model_path):
                print(f"Custom emotion model not found: {model_path}")
                return

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            cfg = checkpoint.get("config", {})
            model_cfg = cfg.get("model", {})
            data_cfg  = cfg.get("data",  {})
            tf_cfg    = cfg.get("transform", {})

            model_name  = model_cfg.get("name", "mobilenetv3_large_100.ra_in1k")
            num_classes = data_cfg.get("num_classes", 7)
            self._class_names = [c.lower() for c in data_cfg.get("class_names", [])]

            self._img_size = tf_cfg.get("img_size", 224)
            self._mean = np.array(tf_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            self._std  = np.array(tf_cfg.get("std",  [0.229, 0.224, 0.225]), dtype=np.float32)

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(self._device)

            self._torch = torch
            self.custom_model = model
            self.custom_model_available = True

            best_acc = checkpoint.get("best_acc", 0)
            print(
                f"Custom emotion model loaded: {model_name} "
                f"(best_acc={best_acc:.1f}%, device={self._device})"
            )
        except Exception as e:
            print(f"Could not load custom emotion model: {e}")

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if custom model is ready."""
        return self.custom_model_available

    def detect_emotion(
        self, frame: np.ndarray, enforce_detection: bool = False
    ) -> Optional[Dict]:
        """
        Detect emotion in frame.
        Returns cached result within detection_interval to save compute.
        """
        if not self.is_available:
            return None

        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.last_emotion

        result = None
        if self.custom_model_available:
            try:
                result = self._detect_with_custom_model(frame, enforce_detection)
            except Exception as e:
                if enforce_detection:
                    raise
                print(f"Custom model inference error: {e}")

        self.last_detection_time = current_time
        if result is not None:
            self.last_emotion = result
        return self.last_emotion

    def get_teaching_recommendation(self, emotion_data: Dict) -> str:
        """Return a teaching suggestion string based on detected emotion."""
        if not emotion_data:
            return ""

        learning_state = emotion_data.get("learning_state", "neutral")
        confidence     = emotion_data.get("confidence", 0)

        if confidence < 40:
            return ""

        recommendations = {
            "positive":   "Student appears engaged and understanding. Continue current approach.",
            "confused":   "Student seems confused or surprised. Consider explaining the concept differently.",
            "frustrated": "Student appears frustrated. Take a break or try a simpler explanation.",
            "bored":      "Student seems disengaged. Try to make content more interactive or take a break.",
            "neutral":    "",
        }
        return recommendations.get(learning_state, "")

    def draw_emotion_overlay(
        self, frame: np.ndarray, emotion_data: Optional[Dict]
    ) -> np.ndarray:
        """Draw emotion info and probability bars onto frame."""
        if not emotion_data:
            return frame

        dominant_emotion = emotion_data.get("dominant_emotion", "unknown")
        smoothed_emotion = emotion_data.get("smoothed_emotion", dominant_emotion)
        confidence       = emotion_data.get("confidence", 0)
        learning_state   = emotion_data.get("learning_state", "unknown")
        backend          = emotion_data.get("backend", "")

        emotion_colors = {
            "happy":    (0, 255, 0),
            "sad":      (255, 0, 0),
            "angry":    (0, 0, 255),
            "surprise": (255, 255, 0),
            "fear":     (128, 0, 128),
            "disgust":  (0, 128, 128),
            "neutral":  (128, 128, 128),
        }
        color = emotion_colors.get(smoothed_emotion, (255, 255, 255))

        backend_tag = f" [{backend}]" if backend else ""
        y_offset = 120
        cv2.putText(
            frame,
            f"Emotion: {smoothed_emotion.upper()} ({confidence:.1f}%){backend_tag}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        y_offset += 25
        cv2.putText(
            frame,
            f"Learning State: {learning_state}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        emotion_probs = emotion_data.get("emotion_probabilities", {})
        if emotion_probs:
            sorted_emotions = sorted(
                emotion_probs.items(), key=lambda x: x[1], reverse=True
            )[:3]
            y_offset += 30
            for emotion, prob in sorted_emotions:
                bar_width = int(prob * 2)
                cv2.rectangle(
                    frame,
                    (10, y_offset),
                    (10 + bar_width, y_offset + 15),
                    emotion_colors.get(emotion, (255, 255, 255)),
                    -1,
                )
                cv2.putText(
                    frame,
                    f"{emotion}: {prob:.1f}%",
                    (220, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                )
                y_offset += 20

        return frame

    # ── Private: custom model inference ───────────────────────────────────

    def _detect_with_custom_model(
        self, frame: np.ndarray, enforce_detection: bool = False
    ) -> Optional[Dict]:
        """Run MobileNetV3 inference on the largest detected face in frame."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )

        if len(faces) == 0:
            if enforce_detection:
                raise ValueError("Face could not be detected")
            return None

        # Pick largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Crop with 10% padding
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        face_tensor = self._preprocess_face(face_crop)

        with self._torch.no_grad():
            logits = self.custom_model(face_tensor)
            probs  = self._torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Build probability dict using canonical label names
        emotion_probs: Dict[str, float] = {}
        for i, cls_name in enumerate(self._class_names):
            canonical = self._MODEL_LABEL_MAP.get(cls_name, cls_name)
            emotion_probs[canonical] = float(probs[i]) * 100.0

        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        self.emotion_history.append(dominant_emotion)
        smoothed_emotion = self._get_smoothed_emotion()

        return {
            "dominant_emotion":     dominant_emotion,
            "smoothed_emotion":     smoothed_emotion,
            "emotion_probabilities": emotion_probs,
            "confidence":           emotion_probs[dominant_emotion],
            "face_location":        (x, y, w, h),
            "learning_state":       self._get_learning_state(smoothed_emotion),
            "backend":              "custom_model",
        }

    def _preprocess_face(self, face_bgr: np.ndarray):
        """Resize, normalise, and convert face crop to a model-ready tensor."""
        face_resized = cv2.resize(face_bgr, (self._img_size, self._img_size))
        face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm    = face_rgb.astype(np.float32) / 255.0
        face_norm    = (face_norm - self._mean) / self._std
        # HWC → CHW, add batch dimension
        tensor = self._torch.from_numpy(
            face_norm.transpose(2, 0, 1)
        ).unsqueeze(0)
        return tensor.to(self._device)

    # ── Private: helpers ───────────────────────────────────────────────────

    def _get_smoothed_emotion(self) -> str:
        if not self.emotion_history:
            return "neutral"
        emotions = list(self.emotion_history)
        return max(set(emotions), key=emotions.count)

    def _get_learning_state(self, emotion: str) -> str:
        if emotion == "happy":
            return "positive"
        elif emotion in ("surprise", "fear"):
            return "confused"
        elif emotion in ("angry", "disgust"):
            return "frustrated"
        elif emotion == "sad":
            return "bored"
        else:
            return "neutral"


# ── Standalone test ────────────────────────────────────────────────────────

def test_emotion_detection():
    """Test emotion detection with webcam."""
    detector = EmotionDetector()

    if not detector.is_available:
        print("No emotion detection backend available.")
        return

    cap = cv2.VideoCapture(0)
    print("Testing emotion detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion_data = detector.detect_emotion(frame, enforce_detection=False)
        frame = detector.draw_emotion_overlay(frame, emotion_data)

        if emotion_data:
            rec = detector.get_teaching_recommendation(emotion_data)
            if rec:
                print(f"Recommendation: {rec}")

        cv2.imshow("Emotion Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_emotion_detection()
