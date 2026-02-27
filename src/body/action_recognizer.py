"""
Rule-based body action recogniser built on top of BodyPoseDetector outputs.

This is intentionally simple: it looks at a single person's pose (COCO 17 keypoints)
and derives a few high-level boolean flags such as whether a hand is raised.
"""

from typing import Dict, List, Optional


# COCO keypoint indices used by Ultralytics pose models (17 points)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class ActionRecognizer:
    """
    Lightweight, rule-based action recogniser.

    Input:  one or more pose dicts from BodyPoseDetector.detect_pose(...)
    Output: a summary dict with simple boolean / categorical labels, e.g.:
        {
            "has_person": bool,
            "primary_person_index": int,
            "primary_actions": {
                "hand_raised": bool,
                "standing": bool,
                "leaning_left": bool,
                "leaning_right": bool,
            },
        }
    """

    def __init__(
        self,
        min_confidence: float = 0.35,
        hand_raise_margin_ratio: float = 0.15,
    ) -> None:
        """
        Args:
            min_confidence: minimum keypoint confidence used in rules.
            hand_raise_margin_ratio: margin relative to torso height for hand raise.
        """
        self.min_confidence = min_confidence
        self.hand_raise_margin_ratio = hand_raise_margin_ratio

    # ── Public API ──────────────────────────────────────────────────────────

    def analyse(self, poses: List[Dict]) -> Dict:
        """
        Analyse a list of pose dicts and return a high-level summary.
        """
        if not poses:
            return {
                "has_person": False,
                "primary_person_index": None,
                "primary_actions": {
                    "hand_raised": False,
                    "standing": False,
                    "leaning_left": False,
                    "leaning_right": False,
                },
            }

        # Take the highest-score person as "primary"
        primary_idx = max(range(len(poses)), key=lambda i: poses[i].get("score", 0.0))
        primary_pose = poses[primary_idx]

        actions = {
            "hand_raised": self._is_hand_raised(primary_pose),
            "standing": self._is_standing(primary_pose),
            "leaning_left": self._is_leaning(primary_pose, direction="left"),
            "leaning_right": self._is_leaning(primary_pose, direction="right"),
        }

        return {
            "has_person": True,
            "primary_person_index": primary_idx,
            "primary_actions": actions,
        }

    # ── Internal rule helpers ──────────────────────────────────────────────

    def _get_kp(self, pose: Dict, index: int) -> Optional[Dict]:
        kpts = pose.get("keypoints") or []
        if 0 <= index < len(kpts):
            kp = kpts[index]
            if kp.get("conf", 1.0) >= self.min_confidence:
                return kp
        return None

    def _torso_height(self, pose: Dict) -> Optional[float]:
        lh = self._get_kp(pose, LEFT_HIP)
        rh = self._get_kp(pose, RIGHT_HIP)
        ls = self._get_kp(pose, LEFT_SHOULDER)
        rs = self._get_kp(pose, RIGHT_SHOULDER)
        if not (lh and rh and ls and rs):
            return None
        hip_y = 0.5 * (lh["y"] + rh["y"])
        shoulder_y = 0.5 * (ls["y"] + rs["y"])
        return abs(hip_y - shoulder_y)

    def _is_hand_raised(self, pose: Dict) -> bool:
        """
        A hand is considered raised if either wrist is significantly
        above the corresponding shoulder (by a margin relative to torso height).
        """
        ls = self._get_kp(pose, LEFT_SHOULDER)
        rs = self._get_kp(pose, RIGHT_SHOULDER)
        lw = self._get_kp(pose, LEFT_WRIST)
        rw = self._get_kp(pose, RIGHT_WRIST)
        nose = self._get_kp(pose, NOSE)

        if not (ls and rs and (lw or rw)):
            return False

        torso_h = self._torso_height(pose)
        if torso_h is None or torso_h <= 0:
            torso_h = abs(ls["y"] - rs["y"]) or 1.0

        margin = self.hand_raise_margin_ratio * torso_h
        threshold_y = min(ls["y"], rs["y"]) - margin

        def wrist_above(wrist):
            if not wrist:
                return False
            if wrist["y"] < threshold_y:
                return True
            if nose and wrist["y"] < nose["y"]:
                return True
            return False

        return wrist_above(lw) or wrist_above(rw)

    def _is_standing(self, pose: Dict) -> bool:
        """
        Approximate standing vs sitting using the aspect ratio of the person box
        and the relative positions of hips and ankles.
        """
        box = pose.get("box")
        if not box or len(box) != 4:
            return False

        x1, y1, x2, y2 = box
        h = max(1.0, float(y2 - y1))
        w = max(1.0, float(x2 - x1))
        aspect = h / w

        # Basic heuristic: tall and narrow ⇒ likely standing
        if aspect > 1.4:
            return True

        # Fallback: check ankle vs hip vertical distance
        lh = self._get_kp(pose, LEFT_HIP)
        rh = self._get_kp(pose, RIGHT_HIP)
        la = self._get_kp(pose, LEFT_ANKLE)
        ra = self._get_kp(pose, RIGHT_ANKLE)

        if not ((lh or rh) and (la or ra)):
            return False

        hip_y = None
        ankle_y = None

        if lh and rh:
            hip_y = 0.5 * (lh["y"] + rh["y"])
        elif lh:
            hip_y = lh["y"]
        elif rh:
            hip_y = rh["y"]

        if la and ra:
            ankle_y = 0.5 * (la["y"] + ra["y"])
        elif la:
            ankle_y = la["y"]
        elif ra:
            ankle_y = ra["y"]

        if hip_y is None or ankle_y is None:
            return False

        leg_extent = abs(ankle_y - hip_y)
        return leg_extent > 0.5 * h

    def _is_leaning(self, pose: Dict, direction: str) -> bool:
        """
        Determine if the upper body is leaning left or right based on shoulder
        symmetry and the horizontal position of the nose.
        """
        ls = self._get_kp(pose, LEFT_SHOULDER)
        rs = self._get_kp(pose, RIGHT_SHOULDER)
        nose = self._get_kp(pose, NOSE)
        if not (ls and rs and nose):
            return False

        shoulder_mid_x = 0.5 * (ls["x"] + rs["x"])
        dx = nose["x"] - shoulder_mid_x

        # Use shoulder distance as a scale for "significant" lean
        shoulder_span = abs(ls["x"] - rs["x"]) or 1.0
        if abs(dx) < 0.15 * shoulder_span:
            return False

        if direction == "left":
            return dx < 0
        if direction == "right":
            return dx > 0
        return False

