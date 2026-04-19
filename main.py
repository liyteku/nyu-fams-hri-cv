#!/usr/bin/env python3
"""
Live demo: webcam → EmotionDetector + GazeTracker → PolicyEngine instruction.

Run from project root:
    python main.py
    python main.py --camera 1

Does not load StudentFaceRecognition (see README).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.face import EmotionDetector, GazeTracker
from src.embodied_policy import PolicyEngine, instruction_from_perception


def main() -> None:
    parser = argparse.ArgumentParser(description="HRI-CV live demo (emotion + gaze + policy)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="Print policy instruction to terminal every N frames (0=disable)",
    )
    args = parser.parse_args()

    detector = EmotionDetector()
    if not detector.is_available:
        print("EmotionDetector: no model loaded (place best_model.pth under outputs/ or project root).")
        sys.exit(1)

    tracker = GazeTracker()
    policy_engine = PolicyEngine()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}")
        sys.exit(1)

    print("Running. Press 'q' to quit.")
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        emotion_data = detector.detect_emotion(frame, enforce_detection=False)
        gaze_data = tracker.detect_gaze(frame)

        frame = detector.draw_emotion_overlay(frame, emotion_data)
        frame = tracker.draw_gaze_overlay(frame, gaze_data)

        instruction = instruction_from_perception(
            emotion_data,
            gaze_data,
            policy_engine=policy_engine,
        )

        y = 30
        if instruction:
            for line in _wrap_text(instruction, max_chars=72):
                cv2.putText(
                    frame,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 255, 220),
                    1,
                    lineType=cv2.LINE_AA,
                )
                y += 18

        if args.print_every and instruction and frame_i % args.print_every == 0:
            print(instruction)

        cv2.imshow("NYU FAMS HRI-CV", frame)
        frame_i += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _wrap_text(text: str, max_chars: int = 72) -> list:
    """Split long policy text into lines that fit on screen."""
    words = text.split()
    lines: list = []
    cur: list = []
    n = 0
    for w in words:
        if n + len(w) + (1 if cur else 0) > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += len(w) + (1 if len(cur) > 1 else 0)
    if cur:
        lines.append(" ".join(cur))
    return lines[:6]


if __name__ == "__main__":
    main()
