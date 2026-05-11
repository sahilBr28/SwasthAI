# ============================================
# SWASTHAI - Phase 2: Preprocessing
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(r"D:\Python\PythonProject\SwasthAI")
from src.config import DIABETES_DATA_PATH, PROCESSED_DATA_DIR, MODEL_DIR


def load_data():
    df = pd.read_csv(DIABETES_DATA_PATH)
    print("Data loaded:", df.shape)
    return df

def fix_zero_values(df):
    print("\n Fixing impossible zero values...")

    # These columns CANNOT be zero in real life
    # Zero here means the value was not recorded
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in columns_with_zeros:
        zero_count = (df[col] == 0).sum()
        print(f"  {col}: {zero_count} zeros found")

        # Replace 0 with NaN (Not a Number) so we can treat them as missing
        df[col] = df[col].replace(0, np.nan)

    print("Zeros replaced with NaN successfully!")
    return df

def fix_missing_values(df):
    print("\n Fixing missing values...")

    # For each column, fill missing values with the MEDIAN of that column
    # We use median (not average) because it's not affected by extreme values
    columns_to_fill = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in columns_to_fill:
        median_value = df[col].median()
        missing_count = df[col].isnull().sum()
        df[col] = df[col].fillna(median_value)
        print(f"  {col}: filled {missing_count} missing values with median ({median_value:.1f})")

    print("Missing values fixed!")
    return df

def split_features_and_target(df):
    print("\n Splitting features and target...")

    # X = inputs (what we give the AI to learn from)
    # y = output (what we want AI to predict)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print(f"  Features (X): {X.shape}")
    print(f"  Target (y):   {y.shape}")
    print(f"  Feature names: {list(X.columns)}")
    return X, y

def scale_features(X):
    print("\n Scaling features to same range...")

    # StandardScaler brings all values to same scale
    # So AI doesn't think Insulin (0-846) is more important than Age (21-81)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame so column names are preserved
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("  Before scaling - Glucose range:", f"{X['Glucose'].min():.1f} to {X['Glucose'].max():.1f}")
    print("  After scaling  - Glucose range:", f"{X_scaled['Glucose'].min():.2f} to {X_scaled['Glucose'].max():.2f}")

    return X_scaled, scaler

def save_processed_data(X_scaled, y):
    print("\n Saving processed data...")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Combine X and y back together and save
    processed_df = X_scaled.copy()
    processed_df['Outcome'] = y.values
    processed_df.to_csv(
        os.path.join(PROCESSED_DATA_DIR, "diabetes_processed.csv"),
        index=False
    )
    print("Saved to data/processed/diabetes_processed.csv")

def run_preprocessing():
    print("="*50)
    print("SWASTHAI - Starting Preprocessing")
    print("="*50)

    df = load_data()
    df = fix_zero_values(df)
    df = fix_missing_values(df)
    X, y = split_features_and_target(df)
    X_scaled, scaler = scale_features(X)
    save_processed_data(X_scaled, y)
    save_scaler(scaler)

    print("\n" + "="*50)
    print("Preprocessing Complete!")
    print("="*50)

    return X_scaled, y, scaler

# ============================================
# Save the scaler so prediction can use it
# ============================================
def save_scaler(scaler):
    import pickle
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to models/scaler.pkl")

# Run it
if __name__ == "__main__":
    X, y, scaler = run_preprocessing()