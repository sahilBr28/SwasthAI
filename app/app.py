# ============================================
# SWASTHAI - Phase 6: Web Application
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(r"D:\Python\PythonProject\SwasthAI")
from src.config import MODEL_DIR

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="SwasthAI — Diabetes Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

# ============================================
# Load Model and Scaler
# ============================================
@st.cache_resource
def load_model_and_scaler():
    model_path  = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_and_scaler()

# ============================================
# App Header
# ============================================
st.title("🏥 SwasthAI")
st.subheader("Diabetes Risk Prediction System")
st.markdown("""
Enter the patient's medical details below.
The AI will analyze the information and predict diabetes risk.
""")
st.divider()

# ============================================
# Input Form
# ============================================
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="How many times the patient has been pregnant"
    )

    glucose = st.slider(
        "Glucose Level (mg/dL)",
        min_value=44, max_value=200, value=110,
        help="Blood glucose concentration after oral glucose test"
    )

    blood_pressure = st.slider(
        "Blood Pressure (mm Hg)",
        min_value=24, max_value=122, value=72,
        help="Diastolic blood pressure"
    )

    skin_thickness = st.slider(
        "Skin Thickness (mm)",
        min_value=7, max_value=99, value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.slider(
        "Insulin Level (IU/mL)",
        min_value=14, max_value=846, value=80,
        help="2-hour serum insulin level"
    )

    bmi = st.slider(
        "BMI",
        min_value=18.0, max_value=67.0,
        value=25.0, step=0.1,
        help="Body Mass Index = weight(kg) / height(m)²"
    )

    dpf = st.slider(
        "Diabetes Pedigree Function",
        min_value=0.07, max_value=2.50,
        value=0.47, step=0.01,
        help="Family history score — higher means stronger family history of diabetes"
    )

    age = st.slider(
        "Age (years)",
        min_value=21, max_value=81, value=30,
        help="Patient's age"
    )

st.divider()

# ============================================
# Show a quick summary of entered values
# ============================================
with st.expander("📊 View entered values summary"):
    summary_data = {
        "Feature": [
            "Pregnancies", "Glucose", "Blood Pressure",
            "Skin Thickness", "Insulin", "BMI",
            "Diabetes Pedigree Function", "Age"
        ],
        "Value": [
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ],
        "Normal Range": [
            "0–5", "70–110 mg/dL", "60–80 mm Hg",
            "10–30 mm", "16–166 IU/mL", "18.5–24.9",
            "0.1–0.5", "Any"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ============================================
# Predict Button
# ============================================
predict_btn = st.button(
    "🔍 Analyze & Predict",
    type="primary",
    use_container_width=True
)

if predict_btn:
    # Prepare input
    patient_data = [[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]]

    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]

    patient_df = pd.DataFrame(patient_data, columns=columns)
    patient_scaled = scaler.transform(patient_df)
    patient_scaled = pd.DataFrame(patient_scaled, columns=columns)

    prediction  = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]

    no_diabetes_pct  = round(probability[0] * 100, 1)
    has_diabetes_pct = round(probability[1] * 100, 1)

    st.divider()
    st.subheader("🧠 AI Prediction Result")

    # ============================================
    # Show result
    # ============================================
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("No Diabetes Probability",  f"{no_diabetes_pct}%")
    with col_b:
        st.metric("Has Diabetes Probability", f"{has_diabetes_pct}%")

    # Progress bar showing risk visually
    st.markdown("**Risk Level Indicator:**")
    st.progress(int(has_diabetes_pct))

    # Main result
    if prediction == 1:
        if has_diabetes_pct >= 80:
            st.error("""
            ### 🔴 VERY HIGH RISK OF DIABETES

            The AI has detected strong indicators of diabetes.
            **Immediate medical consultation is strongly recommended.**
            """)
        elif has_diabetes_pct >= 60:
            st.error("""
            ### 🔴 HIGH RISK OF DIABETES

            Several diabetes indicators are elevated.
            **Please consult a doctor as soon as possible.**
            """)
        else:
            st.warning("""
            ### 🟡 MODERATE RISK OF DIABETES

            Some diabetes indicators are present.
            **Schedule a medical checkup soon.**
            """)
    else:
        if no_diabetes_pct >= 80:
            st.success("""
            ### 🟢 VERY LOW RISK OF DIABETES

            Your indicators look healthy!
            **Maintain your current lifestyle with regular checkups.**
            """)
        else:
            st.success("""
            ### 🟢 LOW RISK OF DIABETES

            Most indicators are within normal range.
            **Stay active and get checkups every 6 months.**
            """)

    # Key factors
    st.divider()
    st.subheader("🔍 Key Risk Factors in Your Input")

    flags = []
    if glucose > 140:
        flags.append(f"⚠️ Glucose is HIGH ({glucose} mg/dL) — normal is below 110")
    if bmi > 30:
        flags.append(f"⚠️ BMI is HIGH ({bmi}) — normal is below 25")
    if age > 45:
        flags.append(f"⚠️ Age ({age}) is a risk factor — risk increases after 45")
    if dpf > 0.8:
        flags.append(f"⚠️ Strong family history of diabetes (DPF: {dpf})")
    if blood_pressure > 90:
        flags.append(f"⚠️ Blood pressure is HIGH ({blood_pressure} mm Hg)")

    if flags:
        for flag in flags:
            st.markdown(flag)
    else:
        st.success("✅ No major individual risk factors detected.")

    # Disclaimer
    st.divider()
    st.caption("""
    ⚠️ DISCLAIMER: This prediction is generated by an AI model trained on
    historical data. It is NOT a medical diagnosis. Always consult a
    qualified healthcare professional for proper medical advice.
    """)