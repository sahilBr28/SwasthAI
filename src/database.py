# ============================================
# SWASTHAI - Database Layer
# ============================================

import sqlite3
import os
import sys
from datetime import datetime

sys.path.append(r"D:\Python\PythonProject\SwasthAI")
from src.config import BASE_DIR

DB_PATH = os.path.join(BASE_DIR, "data", "swasthai.db")

# ============================================
# Create tables if they don't exist
# ============================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp                 TEXT NOT NULL,
            patient_name              TEXT DEFAULT 'Anonymous',
            age                       INTEGER,
            pregnancies               INTEGER,
            glucose                   REAL,
            blood_pressure            REAL,
            skin_thickness            REAL,
            insulin                   REAL,
            bmi                       REAL,
            diabetes_pedigree         REAL,
            prediction                INTEGER,
            no_diabetes_probability   REAL,
            diabetes_probability      REAL,
            risk_level                TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id  INTEGER,
            actual_outcome INTEGER,
            notes          TEXT,
            timestamp      TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized at:", DB_PATH)

# ============================================
# Save a new prediction
# ============================================
def save_prediction(patient_name, age, pregnancies, glucose,
                    blood_pressure, skin_thickness, insulin,
                    bmi, diabetes_pedigree, prediction,
                    no_prob, yes_prob, risk_level):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            timestamp, patient_name, age, pregnancies, glucose,
            blood_pressure, skin_thickness, insulin, bmi,
            diabetes_pedigree, prediction,
            no_diabetes_probability, diabetes_probability, risk_level
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient_name, age, pregnancies, glucose,
        blood_pressure, skin_thickness, insulin,
        bmi, diabetes_pedigree, prediction,
        round(no_prob, 3), round(yes_prob, 3), risk_level
    ))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

# ============================================
# Get all predictions
# ============================================
def get_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, patient_name, age, glucose,
               bmi, prediction, diabetes_probability, risk_level
        FROM predictions
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

# ============================================
# Get single prediction by ID
# ============================================
def get_prediction_by_id(pred_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id=?", (pred_id,))
    row = cursor.fetchone()
    conn.close()
    return row

# ============================================
# Get analytics summary
# ============================================
def get_analytics():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1")
    high_risk = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(glucose) FROM predictions")
    avg_glucose = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(bmi) FROM predictions")
    avg_bmi = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(age) FROM predictions")
    avg_age = cursor.fetchone()[0]

    cursor.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as count
        FROM predictions
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """)
    daily = cursor.fetchall()

    cursor.execute("""
        SELECT risk_level, COUNT(*) as count
        FROM predictions
        GROUP BY risk_level
    """)
    risk_dist = cursor.fetchall()

    cursor.execute("""
        SELECT AVG(glucose), AVG(bmi), AVG(age)
        FROM predictions WHERE prediction=1
    """)
    diabetic_avg = cursor.fetchone()

    cursor.execute("""
        SELECT AVG(glucose), AVG(bmi), AVG(age)
        FROM predictions WHERE prediction=0
    """)
    healthy_avg = cursor.fetchone()

    conn.close()

    return {
        "total":        total,
        "high_risk":    high_risk,
        "low_risk":     total - high_risk,
        "avg_glucose":  round(avg_glucose or 0, 1),
        "avg_bmi":      round(avg_bmi or 0, 1),
        "avg_age":      round(avg_age or 0, 1),
        "daily":        daily,
        "risk_dist":    risk_dist,
        "diabetic_avg": diabetic_avg,
        "healthy_avg":  healthy_avg,
    }

# ============================================
# Delete a prediction
# ============================================
def delete_prediction(pred_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    conn.commit()
    conn.close()

# ============================================
# Save doctor feedback on a prediction
# ============================================
def save_feedback(prediction_id, actual_outcome, notes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (prediction_id, actual_outcome, notes, timestamp)
        VALUES (?,?,?,?)
    """, (
        prediction_id, actual_outcome, notes,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database ready!")