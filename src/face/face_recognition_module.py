"""
Face Recognition Module for Student Identification
Handles student enrollment and real-time face recognition.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


def _default_encodings_path() -> str:
    """Default path: project_root/data/student_encodings.pkl"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "student_encodings.pkl")


class StudentFaceRecognition:
    """Manages student face recognition and enrollment"""

    def __init__(self, encodings_path: Optional[str] = None):
        """
        Initialize face recognition system.

        Args:
            encodings_path: Path to save/load student face encodings.
                Defaults to project data/student_encodings.pkl.
        """
        self.encodings_path = encodings_path or _default_encodings_path()
        self.known_encodings = []
        self.known_names = []
        self.known_student_ids = []

        # Try to import face_recognition
        try:
            import face_recognition

            self.face_recognition = face_recognition
            self.face_recognition_available = True
        except ImportError:
            print(
                "Warning: face_recognition not available. Face recognition disabled."
            )
            self.face_recognition_available = False

        # Load existing encodings if available
        if self.face_recognition_available:
            self.load_encodings()

    def enroll_student(
        self, name: str, student_id: str, image_path: str
    ) -> Tuple[bool, str]:
        """
        Enroll a new student with their face encoding.

        Args:
            name: Student's name
            student_id: Unique student ID
            image_path: Path to student's photo

        Returns:
            (success, message)
        """
        if not self.face_recognition_available:
            return False, "Face recognition library not available"

        # Load image
        if not os.path.exists(image_path):
            return False, f"Image not found: {image_path}"

        image = self.face_recognition.load_image_file(image_path)

        # Detect face and get encoding
        face_locations = self.face_recognition.face_locations(image)
        if len(face_locations) == 0:
            return False, "No face detected in image"

        if len(face_locations) > 1:
            return (
                False,
                f"Multiple faces detected ({len(face_locations)}). Please use image with single face.",
            )

        # Get face encoding
        face_encodings = self.face_recognition.face_encodings(image, face_locations)
        if len(face_encodings) == 0:
            return False, "Failed to generate face encoding"

        encoding = face_encodings[0]

        # Check if student already enrolled
        if student_id in self.known_student_ids:
            # Update existing student
            idx = self.known_student_ids.index(student_id)
            self.known_encodings[idx] = encoding
            self.known_names[idx] = name
            message = f"Updated enrollment for {name} (ID: {student_id})"
        else:
            # Add new student
            self.known_encodings.append(encoding)
            self.known_names.append(name)
            self.known_student_ids.append(student_id)
            message = f"Enrolled {name} (ID: {student_id})"

        # Save encodings
        self.save_encodings()

        return True, message

    def enroll_student_from_frame(
        self, name: str, student_id: str, frame: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Enroll a student directly from a video frame.

        Args:
            name: Student's name
            student_id: Unique student ID
            frame: Video frame (BGR format)

        Returns:
            (success, message)
        """
        if not self.face_recognition_available:
            return False, "Face recognition library not available"

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and get encoding
        face_locations = self.face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            return False, "No face detected in frame"

        if len(face_locations) > 1:
            return False, f"Multiple faces detected. Please ensure only one person is in frame."

        # Get face encoding
        face_encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)
        if len(face_encodings) == 0:
            return False, "Failed to generate face encoding"

        encoding = face_encodings[0]

        # Check if student already enrolled
        if student_id in self.known_student_ids:
            idx = self.known_student_ids.index(student_id)
            self.known_encodings[idx] = encoding
            self.known_names[idx] = name
            message = f"Updated enrollment for {name} (ID: {student_id})"
        else:
            self.known_encodings.append(encoding)
            self.known_names.append(name)
            self.known_student_ids.append(student_id)
            message = f"Enrolled {name} (ID: {student_id})"

        self.save_encodings()
        return True, message

    def recognize_face(
        self, frame: np.ndarray, tolerance: float = 0.6
    ) -> Optional[Dict]:
        """
        Recognize face in a frame.

        Args:
            frame: Video frame (BGR format)
            tolerance: Face matching tolerance (lower = stricter)

        Returns:
            Dict with recognition results or None if no face detected
        """
        if not self.face_recognition_available or len(self.known_encodings) == 0:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = self.face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            return None

        # Get face encodings
        face_encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = self.face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=tolerance
            )
            face_distances = self.face_recognition.face_distance(
                self.known_encodings, face_encoding
            )

            name = "Unknown"
            student_id = None
            confidence = 0.0

            if True in matches:
                # Get best match
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    name = self.known_names[best_match_idx]
                    student_id = self.known_student_ids[best_match_idx]
                    # Convert distance to confidence (0-100%)
                    confidence = (1.0 - face_distances[best_match_idx]) * 100

            # Face location is (top, right, bottom, left)
            top, right, bottom, left = face_location

            results.append(
                {
                    "name": name,
                    "student_id": student_id,
                    "confidence": confidence,
                    "face_location": (left, top, right - left, bottom - top),  # x, y, w, h
                    "is_known": student_id is not None,
                }
            )

        # Return first result for now (can be extended for multi-face)
        return results[0] if results else None

    def save_encodings(self):
        """Save face encodings to disk"""
        data = {
            "encodings": self.known_encodings,
            "names": self.known_names,
            "student_ids": self.known_student_ids,
        }
        d = os.path.dirname(self.encodings_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.encodings_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.known_encodings)} student encodings to {self.encodings_path}")

    def load_encodings(self):
        """Load face encodings from disk"""
        if not os.path.exists(self.encodings_path):
            print(f"No existing encodings found at {self.encodings_path}")
            return

        try:
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                self.known_student_ids = data["student_ids"]
            print(f"Loaded {len(self.known_encodings)} student encodings from {self.encodings_path}")
        except Exception as e:
            print(f"Error loading encodings: {e}")

    def get_enrolled_students(self) -> List[Dict]:
        """Get list of all enrolled students"""
        return [
            {"name": name, "student_id": sid}
            for name, sid in zip(self.known_names, self.known_student_ids)
        ]

    def remove_student(self, student_id: str) -> Tuple[bool, str]:
        """
        Remove a student from the system.

        Args:
            student_id: Student ID to remove

        Returns:
            (success, message)
        """
        if student_id not in self.known_student_ids:
            return False, f"Student ID {student_id} not found"

        idx = self.known_student_ids.index(student_id)
        name = self.known_names[idx]

        del self.known_encodings[idx]
        del self.known_names[idx]
        del self.known_student_ids[idx]

        self.save_encodings()
        return True, f"Removed {name} (ID: {student_id})"
