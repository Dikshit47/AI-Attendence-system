import streamlit as st
from datetime import datetime

from src import database as db
from config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
db.init_db()

st.title("🎓 " + APP_TITLE)

st.markdown("""
Welcome! Use the sidebar to navigate:

- **📝 Register Student** — add a new student's face to the system
- **📷 Take Attendance** — capture a photo, recognize the student, mark attendance, detect emotion
- **📋 Attendance Records** — view and export attendance history
- **📊 Dashboard** — attendance and emotion analytics

---

### How recognition works
This system uses **DeepFace** to compare a captured photo against faces you've registered.
It also predicts the **apparent facial expression** of the recognized student — this is an
AI estimate, not a measurement of how the student actually feels.

### A few practical notes
- Camera capture uses your browser via `st.camera_input`, so it works on Streamlit
  Community Cloud without extra server setup — you click a photo rather than streaming
  continuous video. See the README for how to upgrade to live video later.
- Face photos and embeddings are stored locally under `data/`. On Streamlit Community
  Cloud, local storage does **not persist** across app restarts/redeploys — see the
  README's "Persistence" section before using this for a real deployment.
""")

students = db.get_all_students()
att_df = db.get_attendance_df()

col1, col2 = st.columns(2)
col1.metric("Registered Students", len(students))

today_count = 0
if not att_df.empty:
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = int((att_df["date"] == today).sum())
col2.metric("Present Today", today_count)
