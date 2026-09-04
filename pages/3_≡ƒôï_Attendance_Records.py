import streamlit as st

from src import database as db

st.set_page_config(page_title="Attendance Records", page_icon="📋")
db.init_db()
st.title("📋 Attendance Records")

df = db.get_attendance_df()

if df.empty:
    st.info("No attendance records yet.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    date_filter = st.date_input("Filter by date", value=None)
with col2:
    student_filter = st.selectbox("Filter by student", ["All"] + sorted(df["name"].unique().tolist()))

filtered = df.copy()
if date_filter:
    filtered = filtered[filtered["date"] == date_filter.strftime("%Y-%m-%d")]
if student_filter != "All":
    filtered = filtered[filtered["name"] == student_filter]

st.dataframe(filtered, use_container_width=True)
st.caption(f"Showing {len(filtered)} of {len(df)} total records.")

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download as CSV", csv, "attendance_export.csv", "text/csv")
