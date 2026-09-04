"""
Attendance business logic: marking present + duplicate prevention.

Rule (per project spec, section 10): if a student is already marked present
for the current day, do NOT create another record -- just report that they
are already marked.
"""

from src import database as db


def process_attendance(student, emotion, distance):
    """
    student: dict with at least 'student_id' and 'name'.
    emotion: predicted dominant emotion string.
    distance: cosine distance from the recognition match (lower = more confident).

    Returns a human-readable status message.
    """
    confidence = round((1 - distance) * 100, 2)  # rough similarity, for display only

    if db.already_marked_today(student["student_id"]):
        return f"ℹ️ {student['name']} is already marked present today."

    db.insert_attendance(student["student_id"], student["name"], emotion, confidence)
    return f"✅ Attendance marked for {student['name']} ({student['student_id']}) — Emotion: {emotion}"
