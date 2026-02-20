"""
Face module: face recognition, emotion detection, gaze tracking.
"""

from .face_recognition_module import StudentFaceRecognition
from .emotion_detection import EmotionDetector
from .gaze_tracking import GazeTracker

__all__ = [
    "StudentFaceRecognition",
    "EmotionDetector",
    "GazeTracker",
]
