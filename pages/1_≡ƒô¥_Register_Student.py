import os

import numpy as np
import streamlit as st

from src import database as db
from src import face_engine as fe
from config import STUDENTS_IMG_DIR

st.set_page_config(page_title="Register Student", page_icon="📝")
db.init_db()
st.title("📝 Register Student")

st.markdown(
    "Fill in student details, then capture **2–3 clear, well-lit photos** of their face "
    "(facing the camera) for better recognition accuracy."
)

with st.form("register_form"):
    student_id = st.text_input("Student ID *")
    name = st.text_input("Full Name *")
    class_ = st.text_input("Class")
    section = st.text_input("Section")
    email = st.text_input("Email")
    submitted_details = st.form_submit_button("Save Details & Continue to Photo Capture")

if submitted_details:
    if not student_id or not name:
        st.error("Student ID and Name are required.")
    elif db.student_exists(student_id):
        st.error(f"Student ID '{student_id}' already exists. Use a different ID, "
                  "or delete the existing student below first.")
    else:
        st.session_state["reg_student_id"] = student_id
        st.session_state["reg_name"] = name
        st.session_state["reg_class"] = class_
        st.session_state["reg_section"] = section
        st.session_state["reg_email"] = email
        st.session_state["reg_captures"] = []
        st.success("Details saved. Now capture photos below.")

if "reg_student_id" in st.session_state:
    st.divider()
    st.subheader(f"Capturing photos for: {st.session_state['reg_name']} ({st.session_state['reg_student_id']})")

    num_captured = len(st.session_state.get("reg_captures", []))
    img = st.camera_input("Capture a photo", key=f"cam_{num_captured}")

    if img is not None:
        st.session_state.setdefault("reg_captures", [])
        st.session_state["reg_captures"].append(img.getvalue())
        st.success(f"Captured {len(st.session_state['reg_captures'])} photo(s) so far.")
        st.rerun()  # refresh so the camera widget resets for the next capture

    st.write(f"Photos captured: **{len(st.session_state.get('reg_captures', []))}** (recommended: 2–3)")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Reset Photos"):
            st.session_state["reg_captures"] = []
            st.rerun()

    with col_b:
        save_clicked = st.button(
            "💾 Save Student", type="primary",
            disabled=len(st.session_state.get("reg_captures", [])) == 0,
        )

    if save_clicked:
        s_id = st.session_state["reg_student_id"]
        s_name = st.session_state["reg_name"]
        os.makedirs(os.path.join(STUDENTS_IMG_DIR, s_id), exist_ok=True)

        embeddings = []
        saved_path = None
        skipped = 0

        with st.spinner("Processing photos..."):
            for i, img_bytes in enumerate(st.session_state["reg_captures"]):
                img_path = os.path.join(STUDENTS_IMG_DIR, s_id, f"img_{i}.jpg")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                try:
                    emb = fe.get_embedding(img_path)
                    embeddings.append(emb)
                    saved_path = img_path
                except ValueError:
                    skipped += 1

        if not embeddings:
            st.error(
                "Could not detect a face in any captured photo. Please retake with better "
                "lighting, facing the camera directly, and try again."
            )
        else:
            avg_embedding = np.mean(embeddings, axis=0)
            db.add_student(
                s_id, s_name,
                st.session_state.get("reg_class", ""),
                st.session_state.get("reg_section", ""),
                st.session_state.get("reg_email", ""),
                avg_embedding, saved_path,
            )
            st.success(f"✅ {s_name} registered successfully with {len(embeddings)} usable photo(s).")
            if skipped:
                st.warning(f"{skipped} photo(s) were skipped (no face detected).")

            for key in ["reg_student_id", "reg_name", "reg_class", "reg_section",
                        "reg_email", "reg_captures"]:
                st.session_state.pop(key, None)

st.divider()
st.subheader("Registered Students")
df = db.get_students_df()
if df.empty:
    st.info("No students registered yet.")
else:
    st.dataframe(df, use_container_width=True)
    del_id = st.selectbox("Select a Student ID to delete", [""] + df["student_id"].tolist())
    if del_id and st.button("🗑️ Delete Selected Student"):
        db.delete_student(del_id)
        st.success(f"Deleted student {del_id}.")
        st.rerun()
