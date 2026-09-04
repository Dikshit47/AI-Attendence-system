"""
Face recognition + emotion detection, built on DeepFace.

Why DeepFace instead of dlib/face_recognition:
- dlib needs a C++ build toolchain to install, which frequently breaks on
  newer Python versions and on hosted platforms like Streamlit Community Cloud.
- DeepFace is pure-pip-installable (TensorFlow backend) and bundles emotion
  detection too, so one library covers both Module 3 (encoding) and
  Module 5 (emotion) from the project plan.

IMPORTANT (per the project's ethical requirement): emotion detection here is
a prediction of *facial expression* from pixels, not a measurement of a
student's real internal emotional state. Treat/display it accordingly.
"""

import numpy as np
from deepface import DeepFace

from config import MODEL_NAME, DETECTOR_BACKEND, RECOGNITION_THRESHOLD


def get_embedding(image_path_or_array):
    """
    Returns a 1D numpy embedding for the first detected face.
    Raises ValueError if no face could be detected.
    """
    try:
        result = DeepFace.represent(
            img_path=image_path_or_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except Exception as e:
        raise ValueError(f"No face detected or a processing error occurred: {e}")

    if not result:
        raise ValueError("No face detected in image.")

    return np.array(result[0]["embedding"], dtype=np.float32)


def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return 1 - float(np.dot(a, b))


def recognize_face(embedding, known_students, threshold=RECOGNITION_THRESHOLD):
    """
    known_students: list of dicts, each with 'embedding', 'student_id', 'name'.
    Returns (best_match_dict_or_None, distance_to_best_match).

    A match is only accepted if its distance is within `threshold` -- this is
    the "do not blindly identify every face" safeguard from the spec.
    """
    best_match = None
    best_distance = float("inf")

    for student in known_students:
        dist = cosine_distance(embedding, student["embedding"])
        if dist < best_distance:
            best_distance = dist
            best_match = student

    if best_match is not None and best_distance <= threshold:
        return best_match, best_distance
    return None, best_distance


def analyze_emotion(image_path_or_array):
    """
    Returns the dominant predicted facial expression as a capitalized string
    (e.g. "Happy"), or "Unknown" if it can't be determined.
    """
    try:
        result = DeepFace.analyze(
            img_path=image_path_or_array,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if isinstance(result, list):
            result = result[0]
        return str(result.get("dominant_emotion", "Unknown")).capitalize()
    except Exception:
        return "Unknown"
