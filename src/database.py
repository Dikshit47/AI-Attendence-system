"""
Database layer (SQLite) for the AI Attendance System.

Two tables:
- students:   student_id, name, class, section, email, embedding (pickled numpy array), image_path
- attendance: id, student_id, name, date, time, status, emotion, confidence

Kept deliberately simple (raw sqlite3, no ORM) so it is easy to read, debug,
and later swap for MySQL/Postgres if you outgrow SQLite.
"""

import sqlite3
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            class TEXT,
            section TEXT,
            email TEXT,
            embedding BLOB NOT NULL,
            image_path TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            emotion TEXT,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- students

def add_student(student_id, name, class_, section, email, embedding, image_path):
    conn = get_connection()
    cur = conn.cursor()
    emb_blob = pickle.dumps(np.array(embedding, dtype=np.float32))
    cur.execute("""
        INSERT OR REPLACE INTO students
            (student_id, name, class, section, email, embedding, image_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, name, class_, section, email, emb_blob, image_path,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def student_exists(student_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_all_students():
    """Returns list of dicts, each with a decoded numpy 'embedding'."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT student_id, name, class, section, email, embedding, image_path FROM students")
    rows = cur.fetchall()
    conn.close()

    students = []
    for r in rows:
        students.append({
            "student_id": r[0],
            "name": r[1],
            "class": r[2],
            "section": r[3],
            "email": r[4],
            "embedding": pickle.loads(r[5]),
            "image_path": r[6],
        })
    return students


def get_students_df():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT student_id, name, class, section, email, created_at FROM students", conn)
    conn.close()
    return df


def delete_student(student_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()


# -------------------------------------------------------------- attendance

def already_marked_today(student_id):
    conn = get_connection()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("SELECT 1 FROM attendance WHERE student_id = ? AND date = ?", (student_id, today))
    row = cur.fetchone()
    conn.close()
    return row is not None


def insert_attendance(student_id, name, emotion, confidence):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now()
    cur.execute("""
        INSERT INTO attendance (student_id, name, date, time, status, emotion, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
          "Present", emotion, confidence))
    conn.commit()
    conn.close()


def get_attendance_df():
    conn = get_connection()
    df = pd.read_sql_query(
        """SELECT student_id, name, date, time, status, emotion, confidence
           FROM attendance ORDER BY date DESC, time DESC""",
        conn,
    )
    conn.close()
    return df
