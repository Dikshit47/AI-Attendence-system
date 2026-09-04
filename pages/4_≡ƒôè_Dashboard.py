from datetime import datetime

import streamlit as st

from src import database as db

st.set_page_config(page_title="Dashboard", page_icon="📊")
db.init_db()
st.title("📊 Dashboard")

students_df = db.get_students_df()
att_df = db.get_attendance_df()

col1, col2, col3 = st.columns(3)
col1.metric("Total Registered Students", len(students_df))

today = datetime.now().strftime("%Y-%m-%d")
today_present = int((att_df["date"] == today).sum()) if not att_df.empty else 0
col2.metric("Present Today", today_present)

pct = round((today_present / len(students_df) * 100), 1) if len(students_df) else 0
col3.metric("Today's Attendance %", f"{pct}%")

st.divider()

if att_df.empty:
    st.info("No attendance data yet — analytics will appear here once attendance is taken.")
    st.stop()

st.subheader("Attendance Over Time")
by_date = att_df.groupby("date").size().reset_index(name="count").sort_values("date")
st.bar_chart(by_date.set_index("date"))

st.subheader("Attendance by Student")
by_student = (
    att_df.groupby("name").size().reset_index(name="days_present")
    .sort_values("days_present", ascending=False)
)
st.dataframe(by_student, use_container_width=True)

st.subheader("Emotion Distribution (AI estimate)")
emo_counts = att_df["emotion"].value_counts().reset_index()
emo_counts.columns = ["emotion", "count"]
st.bar_chart(emo_counts.set_index("emotion"))
st.caption(
    "Reflects AI-predicted facial expressions at the moment attendance was captured — "
    "not a measurement of students' actual emotional states."
)
