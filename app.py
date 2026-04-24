import streamlit as st
import pandas as pd
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Set page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #4CAF50; color: white; }
    .prediction-box { padding: 20px; border-radius: 15px; background-color: #ffffff; text-align: center; border: 2px solid #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_student = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kd54vcmx.json")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Header Section
st.title("🎓 Student Outcome Prediction App")
st_lottie(lottie_student, height=200, key="coding")

st.markdown("### Enter Student Details to Predict Outcome")

# Input Layout using Columns
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100, value=18)
    study_hours = st.slider("Study Hours Per Week", 0, 50, 15)

with col2:
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
    parent_edu = st.selectbox("Parent Education Level", options=["High School", "Bachelor's", "Master's", "PhD"])
    internet = st.radio("Has Internet Access?", options=["Yes", "No"])

with col3:
    extracurricular = st.radio("Extracurricular Activities?", options=["Yes", "No"])
    prev_score = st.number_input("Previous Score", 0, 100, 75)
    final_score_input = st.number_input("Current Final Score", 0, 100, 80)

# Preprocessing Inputs
# Mapping categorical to numeric (adjust based on your specific model training)
gender_val = 1 if gender == "Male" else 0
internet_val = 1 if internet == "Yes" else 0
extra_val = 1 if extracurricular == "Yes" else 0
edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
edu_val = edu_map[parent_edu]

# Create Input Array
features = np.array([[gender_val, age, study_hours, attendance, edu_val, 
                      internet_val, extra_val, prev_score, final_score_input]])

# Prediction
if st.button("Predict Outcome"):
    prediction = model.predict(features)
    
    st.markdown("---")
    st.balloons()
    
    with st.container():
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        result_text = "SUCCESS" if prediction[0] == 1 else "FURTHER REVIEW NEEDED"
        st.header(f"✨ {result_text} ✨")
        st.markdown('</div>', unsafe_allow_html=True)
