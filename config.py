import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_IMG_DIR = os.path.join(DATA_DIR, "students")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")

os.makedirs(STUDENTS_IMG_DIR, exist_ok=True)

# ---- Face recognition settings ----
# SFace is a lightweight model -> keeps memory/CPU usage low, which matters
# on free-tier hosts like Streamlit Community Cloud.
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"          # fast, no heavy extra dependencies
RECOGNITION_THRESHOLD = 0.60         # cosine distance: lower = stricter match

# ---- Emotion detection ----
EMOTION_ACTIONS = ["emotion"]

APP_TITLE = "AI-Based Student Attendance System with Emotion Detection"
