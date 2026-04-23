"""
Emotion Detection Module

Primary backend: custom MobileNetV3 model (best_model.pth, trained on RAF-DB).
Looks for outputs/best_model.pth or best_model.pth in project root.
"""

import cv2
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from ..embodied_policy.policy_engine import PolicyEngine


def _put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple,
    font: int,
    scale: float,
    color: tuple,
    thickness: int,
) -> None:
    """``putText`` with a thin black outline for contrast on bright backgrounds."""
    x, y = int(org[0]), int(org[1])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.putText(
            img,
            text,
            (x + dx, y + dy),
            font,
            scale,
            (0, 0, 0),
            thickness + 1,
            lineType=cv2.LINE_AA,
        )
    cv2.putText(img, text, org, font, scale, color, thickness, lineType=cv2.LINE_AA)


def _top_emotions_by_prob(emotion_probs: Dict[str, float], k: int = 2) -> List[Dict[str, float]]:
    """Return top-k emotions by softmax probability, e.g. [{"emotion": "happy", "confidence": 45.2}, ...]."""
    sorted_items = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"emotion": name, "confidence": float(p)} for name, p in sorted_items]


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

    def __init__(
        self,
        vote_window: int = 3,
        inference_every_n_frames: int = 3,
        min_inference_interval_sec: float = 0.0,
        model_path: Optional[str] = None,
        *,
        smoothing_window: Optional[int] = None,
        anger_suppress: bool = True,
        anger_min_confidence: float = 45.0,
        anger_min_lead_pct: float = 6.0,
        prefer_neutral_when_ambiguous: bool = True,
        ambiguous_margin_pct: float = 11.0,
        angry_dampen_factor: float = 1.0,
    ):
        """
        Args:
            vote_window: number of recent inference results to majority-vote for
                `smoothed_emotion` / `learning_state` (each sample is one forward pass).
            inference_every_n_frames: run the model every N-th call to `detect_emotion`
                (main-loop iterations ≈ camera frames). 1 = every frame.
            min_inference_interval_sec: optional extra throttle between inferences (0 = off).
            model_path: path to best_model.pth; defaults to outputs/best_model.pth or best_model.pth
            smoothing_window: deprecated alias for `vote_window`.
            anger_suppress: if True, only accept `angry` when probability and margin vs
                runner-up are high enough; otherwise use the second-best class (reduces
                false angry from neutral/concentrated faces).
            anger_min_confidence: min softmax probability (0-100) for angry to be kept
                when ``anger_suppress`` is on (lower = easier to accept angry).
            anger_min_lead_pct: angry must lead the second class by at least this many
                percentage points.
            prefer_neutral_when_ambiguous: if True and (top1 prob - top2 prob) is below
                `ambiguous_margin_pct`, use `neutral` as the frame label for voting
                (reduces flip-flop and increases neutral when the model is unsure).
                Does not apply when the dominant label is already ``angry`` (so angry
                is not collapsed to neutral by a narrow margin).
            ambiguous_margin_pct: max gap between 1st and 2nd class (percentage points)
                to treat the frame as ambiguous and prefer neutral.
            angry_dampen_factor: multiply raw `angry` probability by this (then renormalize
                all classes to 100%). Default 1.0 disables dampening. Values below 1.0 lower
                angry prevalence in the UI. Used for `top_emotions`, bars, and voting.
        """
        if smoothing_window is not None:
            vote_window = smoothing_window
        self.vote_window = vote_window
        self._anger_suppress = anger_suppress
        self._anger_min_confidence = float(anger_min_confidence)
        self._anger_min_lead_pct = float(anger_min_lead_pct)
        self._prefer_neutral_ambiguous = prefer_neutral_when_ambiguous
        self._ambiguous_margin_pct = float(ambiguous_margin_pct)
        self._angry_dampen_factor = float(angry_dampen_factor)
        self.emotion_history: deque = deque(maxlen=vote_window)

        self.inference_every_n_frames = max(1, int(inference_every_n_frames))
        self.min_inference_interval_sec = max(0.0, float(min_inference_interval_sec))
        self._frame_counter = 0
        self.last_detection_time = 0.0
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

        The model runs every `inference_every_n_frames` calls (plus optional
        `min_inference_interval_sec`). Each successful inference appends its
        dominant label to a buffer of size `vote_window`; `smoothed_emotion` is
        the majority vote over that buffer. Between inference steps the last
        result is returned unchanged.
        """
        if not self.is_available:
            return None

        self._frame_counter += 1
        if (self._frame_counter % self.inference_every_n_frames) != 0:
            return self.last_emotion

        current_time = time.time()
        if (
            self.min_inference_interval_sec > 0
            and current_time - self.last_detection_time < self.min_inference_interval_sec
        ):
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

    def get_teaching_recommendation(
        self,
        emotion_data: Dict,
        gaze_data: Optional[Dict] = None,
        policy_engine: Optional["PolicyEngine"] = None,
        min_confidence: float = 40.0,
    ) -> str:
        """
        Tutor instruction from :class:`~src.embodied_policy.policy_engine.PolicyEngine`
        (fused emotion + optional gaze). Empty string if ``emotion_data`` is
        missing or confidence is below ``min_confidence``.
        """
        if not emotion_data:
            return ""

        from ..embodied_policy.policy_engine import instruction_from_perception

        return instruction_from_perception(
            emotion_data,
            gaze_data,
            policy_engine=policy_engine,
            min_emotion_confidence=min_confidence,
        )

    def draw_emotion_overlay(
        self, frame: np.ndarray, emotion_data: Optional[Dict]
    ) -> Tuple[np.ndarray, int]:
        """
        Draw emotion info and compact probability bars (left column + bar strip).

        Returns ``(frame, y_next)`` where ``y_next`` is the baseline row below this
        block; pass it to :meth:`~src.face.gaze_tracking.GazeTracker.draw_gaze_overlay`
        as ``start_y`` so gaze lines do not overlap emotion text or bars.
        """
        x0 = 10
        y = 28
        line_h = 26
        if not emotion_data:
            return frame, y

        dominant_emotion = emotion_data.get("dominant_emotion", "unknown")
        smoothed_emotion = emotion_data.get("smoothed_emotion", dominant_emotion)
        confidence       = emotion_data.get("confidence", 0)
        learning_state   = emotion_data.get("learning_state", "unknown")
        backend          = emotion_data.get("backend", "")
        top_emotions     = emotion_data.get("top_emotions") or []

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
        if top_emotions:
            color = emotion_colors.get(top_emotions[0].get("emotion", ""), color)

        backend_tag = f" [{backend}]" if backend else ""
        if len(top_emotions) >= 2:
            e0 = top_emotions[0]
            e1 = top_emotions[1]
            line = (
                f"Emotion: {e0['emotion'].upper()} ({e0['confidence']:.1f}%) | "
                f"{e1['emotion'].upper()} ({e1['confidence']:.1f}%){backend_tag}"
            )
        elif len(top_emotions) == 1:
            e0 = top_emotions[0]
            line = f"Emotion: {e0['emotion'].upper()} ({e0['confidence']:.1f}%){backend_tag}"
        else:
            line = f"Emotion: {smoothed_emotion.upper()} ({confidence:.1f}%){backend_tag}"
        _put_text_outline(frame, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += line_h

        _put_text_outline(
            frame,
            f"Learning State: {learning_state}",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
        y += line_h

        vs = emotion_data.get("vote_support")
        vw = emotion_data.get("vote_window")
        vf = emotion_data.get("votes_filled")
        if vs and vw is not None:
            a, b = vs[0], vs[1]
            _put_text_outline(
                frame,
                f"Vote: {a}/{b} (window={vw}, filled={vf})",
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (245, 245, 255),
                1,
            )
            y += line_h

        emotion_probs = emotion_data.get("emotion_probabilities", {})
        if emotion_probs:
            sorted_emotions = sorted(
                emotion_probs.items(), key=lambda x: x[1], reverse=True
            )[:2]
            bar_max_w = 100
            bar_h = 16
            row_gap = 6
            y += 4
            for emotion, prob in sorted_emotions:
                ec = emotion_colors.get(emotion, (255, 255, 255))
                bar_w = min(int(prob * 2), bar_max_w)
                top = y
                cv2.rectangle(frame, (x0, top), (x0 + bar_w, top + bar_h), ec, -1)
                cv2.rectangle(frame, (x0, top), (x0 + bar_max_w, top + bar_h), (70, 70, 70), 1)
                label = f"{emotion}: {prob:.1f}%"
                lx = x0 + bar_max_w + 10
                _put_text_outline(
                    frame,
                    label,
                    (lx, top + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                )
                y = top + bar_h + row_gap

        return frame, y + 12

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

        # Raw softmax (model), then unified distribution for UI + voting
        emotion_probs_raw: Dict[str, float] = {}
        for i, cls_name in enumerate(self._class_names):
            canonical = self._MODEL_LABEL_MAP.get(cls_name, cls_name)
            emotion_probs_raw[canonical] = float(probs[i]) * 100.0

        emotion_probs = self._dampen_angry_and_renormalize(emotion_probs_raw)

        top_emotions = _top_emotions_by_prob(emotion_probs, 2)

        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        dominant_emotion = self._maybe_demote_angry(emotion_probs, dominant_emotion)
        dominant_emotion = self._maybe_prefer_neutral_ambiguous(
            emotion_probs, dominant_emotion
        )
        self.emotion_history.append(dominant_emotion)
        smoothed_emotion = self._get_smoothed_emotion()
        vote_support = self._vote_support(smoothed_emotion)

        return {
            "dominant_emotion":     dominant_emotion,
            "smoothed_emotion":     smoothed_emotion,
            "top_emotions":         top_emotions,
            "emotion_probabilities": emotion_probs,
            "emotion_probabilities_raw": emotion_probs_raw,
            "confidence":           emotion_probs.get(dominant_emotion, 0.0),
            "face_location":        (x, y, w, h),
            "learning_state":       self._get_learning_state(smoothed_emotion),
            "backend":              "custom_model",
            "vote_window":          self.vote_window,
            "votes_filled":         len(self.emotion_history),
            "vote_support":         vote_support,
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

    def _dampen_angry_and_renormalize(
        self, emotion_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Single distribution for top_emotions, probability bars, and post-rules.
        Lowers angry share, then renormalizes to sum 100%.
        """
        out = {k: float(v) for k, v in emotion_probs.items()}
        f = self._angry_dampen_factor
        if "angry" in out and f < 1.0:
            out["angry"] *= f
        total = sum(out.values())
        if total <= 0:
            return dict(emotion_probs)
        return {k: v * 100.0 / total for k, v in out.items()}

    def _maybe_demote_angry(
        self, emotion_probs: Dict[str, float], argmax_emotion: str
    ) -> str:
        """If argmax is angry but not confident enough vs runner-up, use 2nd best."""
        if not self._anger_suppress or argmax_emotion != "angry":
            return argmax_emotion
        sorted_items = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) < 2:
            return argmax_emotion
        top_p = sorted_items[0][1]
        second_name, second_p = sorted_items[1]
        lead = top_p - second_p
        if (
            top_p >= self._anger_min_confidence
            and lead >= self._anger_min_lead_pct
        ):
            return "angry"
        return second_name

    def _maybe_prefer_neutral_ambiguous(
        self, emotion_probs: Dict[str, float], emotion: str
    ) -> str:
        """If top-1 and top-2 are close in probability, label this frame as neutral."""
        if not self._prefer_neutral_ambiguous:
            return emotion
        if emotion == "angry":
            return emotion
        sorted_items = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) < 2:
            return emotion
        top_p = sorted_items[0][1]
        second_p = sorted_items[1][1]
        if top_p - second_p < self._ambiguous_margin_pct:
            return "neutral"
        return emotion

    def _get_smoothed_emotion(self) -> str:
        if not self.emotion_history:
            return "neutral"
        emotions = list(self.emotion_history)
        return max(set(emotions), key=emotions.count)

    def _vote_support(self, winner: str) -> tuple:
        """(count for winner, total votes in buffer)."""
        emotions = list(self.emotion_history)
        if not emotions:
            return (0, 0)
        return (emotions.count(winner), len(emotions))

    def _get_learning_state(self, emotion: str) -> str:
        if emotion in ("surprise", "fear"):
            return "confused"
        if emotion in ("angry", "disgust", "sad"):
            return "frustrated"
        # happy, neutral, and anything else → neutral (no separate positive/bored)
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
        frame, _ = detector.draw_emotion_overlay(frame, emotion_data)

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
