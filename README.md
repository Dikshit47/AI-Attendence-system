# AI-Based Student Attendance System with Emotion Detection

An AI-powered classroom attendance platform that recognizes registered students
from a photo, marks attendance automatically, predicts their apparent facial
expression, and gives teachers a dashboard of attendance/emotion trends.

Built with **Python + Streamlit + DeepFace + SQLite**, designed to run locally
and deploy directly on **Streamlit Community Cloud**.

---

## 1. Why DeepFace instead of `dlib` / `face_recognition`

The original plan considered `face_recognition` (built on `dlib`). In practice,
`dlib` needs a C++ build toolchain to compile, which is a common source of
installation failures — especially on newer Python versions (you mentioned
3.13.2) and on hosted platforms like Streamlit Community Cloud, which don't
give you control over build tools.

**DeepFace** was chosen instead because:
- It installs via plain `pip` (TensorFlow-based), no C++ compilation required.
- It bundles **both** face recognition and emotion analysis in one library,
  covering Module 3 (encoding) and Module 5 (emotion detection) together.
- The `SFace` model + `opencv` detector used here are lightweight, which
  matters on free-tier hosting with limited RAM/CPU.

If you later need dlib-level accuracy or want to compare models, DeepFace
also supports `Facenet`, `ArcFace`, `VGG-Face`, etc. — just change
`MODEL_NAME` in `config.py`. Swapping models does **not** require touching
any other file.

---

## 2. Project Structure

```text
AI_Attendance_System/
├── app.py                          # Home page
├── config.py                       # Paths, model & threshold settings
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                 # Theme
├── src/
│   ├── database.py                 # SQLite: students + attendance tables
│   ├── face_engine.py              # DeepFace wrapper: embeddings, matching, emotion
│   └── attendance.py               # Duplicate-prevention + marking logic
├── pages/                          # Streamlit native multipage nav
│   ├── 1_📝_Register_Student.py
│   ├── 2_📷_Take_Attendance.py
│   ├── 3_📋_Attendance_Records.py
│   └── 4_📊_Dashboard.py
└── data/
    ├── students/                   # Saved face photos (gitignored except .gitkeep)
    └── attendance.db               # Created automatically at first run
```

This maps directly onto the modular plan from your project spec:

| Spec Module              | File(s)                          |
|---------------------------|-----------------------------------|
| Student Registration      | `pages/1_📝_Register_Student.py` |
| Face Dataset               | `data/students/`                 |
| Face Encoding              | `src/face_engine.py`             |
| Live Attendance            | `pages/2_📷_Take_Attendance.py`  |
| Emotion Detection          | `src/face_engine.py` (`analyze_emotion`) |
| Attendance Storage         | `src/database.py`, `data/attendance.db` |
| Dashboard                  | `pages/4_📊_Dashboard.py`        |

---

## 3. Local Setup

**Recommended Python version: 3.10 or 3.11** (best current compatibility with
TensorFlow/DeepFace; avoids the same kind of version friction you hit with
`dlib`).

```bash
# 1. Clone your repo
git clone https://github.com/<your-username>/<your-repo>.git
cd AI_Attendance_System

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The first time DeepFace runs, it downloads the `SFace` model weights
(a few MB) automatically and caches them — this needs internet access once.

---

## 4. How It Works

### Registration (`pages/1_📝_Register_Student.py`)
1. Enter student ID, name, class, section, email.
2. Capture 2–3 photos via the browser camera.
3. Each photo is converted to a face embedding (`face_engine.get_embedding`).
4. Embeddings are averaged and stored in SQLite (`students` table), along
   with one saved reference photo.

### Attendance (`pages/2_📷_Take_Attendance.py`)
1. Capture a photo.
2. Compute its embedding.
3. Compare (cosine distance) against every registered student's embedding.
4. If the closest match is within the recognition threshold → mark present
   and run emotion analysis. Otherwise → "Unknown", nothing is marked.
5. **Duplicate prevention**: if that student already has an attendance row
   for today's date, no new row is inserted — you just get an
   "already marked" message (per spec section 10).

### Emotion Detection
Run only **after** a face is detected/recognized (per spec section 12), using
`DeepFace.analyze(..., actions=["emotion"])`. The dominant predicted
expression (Happy/Sad/Angry/Fear/Surprise/Neutral/Disgust) is stored
alongside the attendance row.

> **Ethical note (kept from your spec, section 24/11):** this is a facial
> *expression* classification from pixels, not a measurement of a student's
> actual psychological state. The UI captions this explicitly, and your
> project report/viva answers should say the same.

---

## 5. Deploying to Streamlit Community Cloud

1. Push this project to a **public or private GitHub repo**. Make sure
   `data/students/*` and `data/attendance.db` are excluded (already handled
   by `.gitignore`) — don't commit real student photos to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch, and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud will install everything from `requirements.txt`
   automatically.

### ⚠️ Persistence warning — read before relying on this in production
Streamlit Community Cloud's filesystem is **ephemeral**: whenever the app
restarts (redeploys, sleeps after inactivity, etc.), anything written to
`data/` — registered students and attendance history — is **lost**.

This is fine for a **college demo** (register + take attendance in the same
session). For a real deployment, do one of:
- Swap `src/database.py` to use a hosted database (e.g. Supabase/Postgres,
  PlanetScale/MySQL, or Streamlit's native
  [connections](https://docs.streamlit.io/develop/api-reference/connections)
  feature) instead of local SQLite.
- Store face photos in cloud object storage (S3, GCS, etc.) instead of the
  local `data/students/` folder.

The rest of the codebase (`face_engine.py`, `attendance.py`, the pages)
doesn't need to change for this — only `database.py`'s connection logic.

---

## 6. Tuning Recognition Accuracy

- The **Recognition strictness** slider on the Take Attendance page controls
  the cosine-distance threshold (`RECOGNITION_THRESHOLD` in `config.py` is
  the default, `0.60`).
- **Lower** it if the system is matching the wrong student (false positives).
- **Raise** it slightly if a genuinely registered student keeps showing as
  "Unknown".
- Registering **2–3 varied, well-lit photos** per student (as the app
  prompts you to do) improves accuracy more than threshold tuning does.
- Per spec section 19, you can further reduce false positives later by
  requiring the same identity across 2–3 consecutive captures before
  marking attendance — this is a good "Phase 8" enhancement.

---

## 7. Roadmap (maps to your Phase 1–8 plan)

| Phase | Status | Notes |
|---|---|---|
| 1. Basic face recognition | ✅ Done | `face_engine.py` |
| 2. Attendance marking | ✅ Done | `attendance.py` |
| 3. Duplicate prevention | ✅ Done | one record per student per day |
| 4. Emotion detection | ✅ Done | `analyze_emotion` |
| 5. Data storage | ✅ Done | SQLite; swap for MySQL/Postgres in production |
| 6. Dashboard | ✅ Done | `pages/4_📊_Dashboard.py` |
| 7. CCTV/IP camera | 🔜 Future | see below |
| 8. Advanced (liveness, multi-frame confirmation, notifications) | 🔜 Future | |

### Upgrading to live/CCTV video later
The current app uses `st.camera_input` (snapshot-per-click), which is the
most reliable option for cloud deployment. True continuous video (webcam or
RTSP/CCTV stream) needs
[`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) and,
for Streamlit Cloud specifically, a TURN server (e.g. via Twilio) because
the free hosting environment doesn't allow direct peer connections. This is
a meaningful additional piece of infrastructure — worth doing as a follow-up
phase once the snapshot-based flow is working well, not before.

---

## 8. Known Limitations (be upfront about these in your report/viva)

- Recognition accuracy depends on lighting, angle, and camera quality.
- Emotion prediction is approximate and can be affected by lighting, camera
  angle, and the model's own limitations — it is not a certain reading of
  how someone feels.
- No liveness/anti-spoofing yet — a photo of a registered student could
  currently be accepted. Flagged in your spec as a Phase 8 item.
- Local SQLite + local file storage is not persistent on free cloud hosting
  (see section 5).

---

## 9. License / Academic Use

Built for coursework/demonstration purposes. Review your institution's
data-privacy policy before capturing real students' biometric data, and
delete `data/students/` and `data/attendance.db` when you're done
demonstrating if you used real faces.
