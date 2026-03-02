import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ===============================
# Step 0: Load Dataset
# ===============================
df = pd.read_csv("../data/raw/diabetes data.csv")

# ===============================
# Step 1️⃣  Replace 0 with NaN
# ===============================
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# ===============================
# Separate Features & Target
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# Step 2️⃣  80-20 Stratified Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# Step 3️⃣  Calculate Median ONLY from TRAIN
# ===============================
train_medians = X_train[cols_with_zero].median()

print("Train Medians:")
print(train_medians)

# ===============================
# Step 4️⃣  Fill Missing Values
# ===============================

# Fill train using train median
X_train[cols_with_zero] = X_train[cols_with_zero].fillna(train_medians)

# Fill test using SAME train median
X_test[cols_with_zero] = X_test[cols_with_zero].fillna(train_medians)

# ===============================
# Final Check
# ===============================
print("\nMissing values in Train:")
print(X_train.isnull().sum())

print("\nMissing values in Test:")
print(X_test.isnull().sum())