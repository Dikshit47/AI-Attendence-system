import streamlit as st

from src import database as db
from src import face_engine as fe
from src import attendance as att
from config import RECOGNITION_THRESHOLD

st.set_page_config(page_title="Take Attendance", page_icon="📷")
db.init_db()
st.title("📷 Take Attendance")

students = db.get_all_students()
if not students:
    st.warning("No students registered yet. Please register students first "
               "(see the Register Student page).")
    st.stop()

st.markdown("Point the camera at a student's face and click the capture button below.")

threshold = st.slider(
    "Recognition strictness (lower = stricter match)",
    min_value=0.30, max_value=0.90, value=float(RECOGNITION_THRESHOLD), step=0.02,
    help="If you see false/incorrect matches, lower this value. If a known student keeps "
         "showing as Unknown, raise it slightly.",
)

img = st.camera_input("Capture student's face")

if img is not None:
    with st.spinner("Analyzing..."):
        tmp_path = "temp_capture.jpg"
        with open(tmp_path, "wb") as f:
            f.write(img.getvalue())

        embedding = None
        try:
            embedding = fe.get_embedding(tmp_path)
        except ValueError as e:
            st.error(f"❌ No face detected. Please retake the photo. ({e})")

        if embedding is not None:
            match, distance = fe.recognize_face(embedding, students, threshold=threshold)

            if match is None:
                st.error("❌ Unknown face — not registered, or match confidence too low. "
                          "Attendance was NOT marked.")
                st.caption(f"Closest distance found: {distance:.3f} (current threshold: {threshold:.2f})")
            else:
                emotion = fe.analyze_emotion(tmp_path)
                message = att.process_attendance(match, emotion, distance)
                st.success(message)

                c1, c2, c3 = st.columns(3)
                c1.metric("Student", match["name"])
                c2.metric("Emotion (AI estimate)", emotion)
                c3.metric("Match confidence", f"{round((1 - distance) * 100, 1)}%")
                st.caption(
                    "Emotion is an AI estimate of facial expression from the photo, "
                    "not a measurement of the student's actual feelings."
                )
