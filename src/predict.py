# ============================================
# SWASTHAI - Phase 5: Predict for New Patient
# ============================================

import sys
import os
import pickle
import numpy as np
import pandas as pd

sys.path.append(r"D:\Python\PythonProject\SwasthAI")
from src.config import MODEL_DIR

# ============================================
# STEP 1: Load saved model and scaler
# ============================================
def load_model_and_scaler():
    model_path  = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print("Model and scaler loaded successfully!")
    return model, scaler

# ============================================
# STEP 2: Predict for a single patient
# ============================================
def predict_patient(patient_data, model, scaler):
    # Column order must match training data exactly
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]

    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data], columns=columns)

    # Scale exactly like training data
    patient_scaled = scaler.transform(patient_df)
    patient_scaled = pd.DataFrame(patient_scaled, columns=patient_df.columns)

    # Predict
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]

    return prediction, probability

# ============================================
# STEP 3: Show result in a readable way
# ============================================
def show_result(patient_data, prediction, probability):
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]

    print("\n" + "="*50)
    print("SWASTHAI - Patient Analysis Report")
    print("="*50)

    print("\nPatient Details:")
    for col, val in zip(columns, patient_data):
        print(f"  {col:30s}: {val}")

    print("\n" + "-"*50)
    print("AI Prediction:")
    print("-"*50)

    no_diabetes_chance  = probability[0] * 100
    has_diabetes_chance = probability[1] * 100

    print(f"  No Diabetes  probability : {no_diabetes_chance:.1f}%")
    print(f"  Has Diabetes probability : {has_diabetes_chance:.1f}%")

    print("\n  RESULT:", end=" ")
    if prediction == 1:
        print("HIGH RISK OF DIABETES")
        print("\n  Recommendation:")
        print("  Please consult a doctor immediately.")
        print("  Early treatment can prevent complications.")
        if has_diabetes_chance >= 80:
            print("  Risk level: VERY HIGH")
        elif has_diabetes_chance >= 60:
            print("  Risk level: HIGH")
        else:
            print("  Risk level: MODERATE")
    else:
        print("LOW RISK OF DIABETES")
        print("\n  Recommendation:")
        print("  Maintain healthy diet and exercise.")
        print("  Get regular checkups every 6 months.")
        if no_diabetes_chance >= 80:
            print("  Risk level: VERY LOW")
        else:
            print("  Risk level: LOW")

    print("="*50)
    print("DISCLAIMER: This is an AI prediction only.")
    print("Always consult a qualified doctor.")
    print("="*50)

# ============================================
# STEP 4: Test with sample patients
# ============================================
def run_predictions():
    model, scaler = load_model_and_scaler()

    print("\n" + "="*50)
    print("Testing with 3 sample patients")
    print("="*50)

    # Patient 1 — High risk profile
    # (older, high glucose, high BMI)
    patient1 = [6, 148, 72, 35, 125, 33.6, 0.627, 50]

    # Patient 2 — Low risk profile
    # (young, normal glucose, normal BMI)
    patient2 = [1, 85, 66, 29, 125, 26.6, 0.351, 31]

    # Patient 3 — Borderline case
    # (middle aged, slightly high glucose)
    patient3 = [3, 120, 70, 25, 100, 28.5, 0.450, 38]

    for i, patient in enumerate([patient1, patient2, patient3], 1):
        print(f"\n--- Patient {i} ---")
        prediction, probability = predict_patient(patient, model, scaler)
        show_result(patient, prediction, probability)

if __name__ == "__main__":
    run_predictions()