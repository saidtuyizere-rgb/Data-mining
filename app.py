"""
Student Performance Predictor - Streamlit App
Run with: streamlit run app.py

Requirements:
    pip install streamlit scikit-learn joblib numpy
"""

import streamlit as st
import joblib
import numpy as np
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #0f1420;
    color: #ccd6f6;
}

.main { background-color: #0f1420; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 960px; }

h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0f1420 0%, #1a2240 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #1e2d50;
}
.app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    color: #e8d5b7;
    margin: 0 0 0.4rem;
}
.app-header p {
    color: #7a8eaa;
    font-size: 1.05rem;
    margin: 0;
}

/* Section labels */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #e8d5b7;
    background: #1a2240;
    border-left: 4px solid #3d6ef5;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    margin: 1.5rem 0 1rem;
}

/* Input labels */
label { color: #aabbcc !important; font-size: 0.9rem !important; }

/* Sliders */
.stSlider > div > div { background: #3d6ef5 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1e2a40 !important;
    border: 1px solid #2c3d5a !important;
    color: #ccd6f6 !important;
    border-radius: 8px !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #3d6ef5, #5580ff);
    color: white;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 3rem;
    cursor: pointer;
    width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5580ff, #3d6ef5);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(61,110,245,0.4);
}

/* Result cards */
.result-high {
    background: linear-gradient(135deg, #0d3326, #1a5c3a);
    border: 2px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-middle {
    background: linear-gradient(135deg, #3b2a08, #6b4d10);
    border: 2px solid #f39c12;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #3b0d0d, #6b1a1a);
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-emoji { font-size: 3rem; }
.result-range { font-size: 1rem; color: #aabbcc; margin-top: 0.4rem; }

/* Divider */
hr { border-color: #1e2d50; margin: 1.5rem 0; }

/* Input container */
.input-card {
    background: #141929;
    border: 1px solid #1e2d50;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists("student_model.pkl"):
        return None
    return joblib.load("student_model.pkl")

model = load_model()


# ─── CLASS DISPLAY INFO ────────────────────────────────────────────────────────

CLASS_INFO = {
    "H": {"label": "HIGH PERFORMANCE",   "emoji": "🏆", "css": "result-high",
          "range": "Score range: 90 – 100", "color": "#2ecc71"},
    "M": {"label": "MIDDLE PERFORMANCE", "emoji": "📘", "css": "result-middle",
          "range": "Score range: 70 – 89", "color": "#f39c12"},
    "L": {"label": "LOW PERFORMANCE",    "emoji": "⚠️",  "css": "result-low",
          "range": "Score range: 0 – 69",  "color": "#e74c3c"},
}


# ─── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Enter student data below to predict academic performance class</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ `student_model.pkl` not found. Place it in the same folder as `app.py` and restart.")
    st.stop()


# ─── FORM ──────────────────────────────────────────────────────────────────────

with st.form("prediction_form"):

    # ── Section 1: Activity Data
    st.markdown('<div class="section-title">📊 Student Activity Data</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        raised_hands = st.slider("✋ Raised Hands", 0, 100, 50,
                                  help="How many times the student raised their hand in class")
        announcements = st.slider("📢 Announcements Viewed", 0, 100, 50,
                                   help="Number of announcements the student viewed")

    with col2:
        visited_resources = st.slider("📚 Visited Resources", 0, 100, 50,
                                       help="Number of course resources the student visited")
        discussion = st.slider("💬 Discussion Participation", 0, 100, 50,
                                help="Number of times the student participated in discussions")

    st.markdown("---")

    # ── Section 2: Student Profile
    st.markdown('<div class="section-title">👤 Student Profile</div>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)

    with col3:
        gender = st.selectbox("Gender", ["M", "F"])
        stage = st.selectbox("School Stage", ["lowerlevel", "MiddleSchool", "HighSchool"])
        grade = st.selectbox("Grade", ["G-02","G-04","G-05","G-06","G-07",
                                        "G-08","G-09","G-10","G-11","G-12"])
        section = st.selectbox("Section", ["A", "B", "C"])

    with col4:
        topic = st.selectbox("Subject / Topic", ["IT","Math","Arabic","Science","English",
                                                   "Quran","Spanish","French","History",
                                                   "Biology","Chemistry","Geology"])
        semester = st.selectbox("Semester", ["First", "Second"])
        relation = st.selectbox("Parent Responsible", ["Father", "Mum"])
        absence = st.selectbox("Absence Days", ["Under-7", "Above-7"])

    with col5:
        nationality = st.selectbox("Nationality", [
            "Kuwait","Jordan","Palestine","Iraq","Lebanon","Tunis",
            "Saudi","Egypt","Syria","USA","Iran","Libya","Morocco","Venezuela","KW"
        ])
        place_of_birth = st.selectbox("Place of Birth", [
            "Kuwait","Jordan","Palestine","Iraq","Lebanon","Tunis",
            "Saudi","Egypt","Syria","USA","Iran","Libya","Morocco","Venezuela","KW"
        ])
        parent_survey = st.selectbox("Parent Answered Survey", ["Yes", "No"])
        parent_satisfaction = st.selectbox("Parent School Satisfaction", ["Good", "Bad"])

    st.markdown("---")

    # ── Submit
    submitted = st.form_submit_button("🔍  PREDICT PERFORMANCE")


# ─── PREDICTION ────────────────────────────────────────────────────────────────

if submitted:
    # Build feature row — order must match training columns
    input_data = [
        raised_hands,
        visited_resources,
        announcements,
        discussion,
        gender,
        nationality,
        place_of_birth,
        stage,
        grade,
        section,
        topic,
        semester,
        relation,
        parent_survey,
        parent_satisfaction,
        absence,
    ]

    try:
        prediction = model.predict([input_data])[0]
        info = CLASS_INFO.get(prediction, {
            "label": str(prediction), "emoji": "❓",
            "css": "result-middle", "range": "", "color": "#aaaaaa"
        })

        st.markdown("---")
        st.markdown(f"""
        <div class="{info['css']}">
            <div class="result-emoji">{info['emoji']}</div>
            <div class="result-title" style="color:{info['color']}">
                {info['label']}
            </div>
            <div class="result-range">{info['range']}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure the feature names and order in `app.py` match exactly what the model was trained on.")