# ============================================
# SWASTHAI - Phase 3: Training the Model
# ============================================

import sys
import os
import pickle

sys.path.append(r"D:\Python\PythonProject\SwasthAI")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.config import PROCESSED_DATA_DIR, MODEL_DIR

# ============================================
# STEP 1: Load the processed data
# ============================================
def load_processed_data():
    path = os.path.join(PROCESSED_DATA_DIR, "diabetes_processed.csv")
    df = pd.read_csv(path)
    print("Processed data loaded:", df.shape)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

# ============================================
# STEP 2: Split into Train and Test sets
# ============================================
def split_data(X, y):
    print("\n Splitting data into Train and Test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% for testing
        random_state=42,     # ensures same split every time you run
        stratify=y           # keeps diabetes ratio same in both sets
    )

    print(f"  Training set: {X_train.shape[0]} patients")
    print(f"  Testing set:  {X_test.shape[0]} patients")
    print(f"  Diabetes in train: {y_train.sum()} patients")
    print(f"  Diabetes in test:  {y_test.sum()} patients")

    return X_train, X_test, y_train, y_test

# ============================================
# STEP 3: Train Model 1 - Logistic Regression
# ============================================
def train_logistic_regression(X_train, y_train):
    print("\n Training Model 1: Logistic Regression...")

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    print("  Logistic Regression trained!")
    return model

# ============================================
# STEP 4: Train Model 2 - Random Forest
# ============================================
def train_random_forest(X_train, y_train):
    print("\n Training Model 2: Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,    # 100 decision trees
        random_state=42
    )
    model.fit(X_train, y_train)

    print("  Random Forest trained!")
    return model

# ============================================
# STEP 5: Evaluate both models
# ============================================
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n Results for {model_name}:")
    print("-" * 40)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"\n  Detailed Report:")
    print(classification_report(y_test, y_pred,
          target_names=["No Diabetes", "Has Diabetes"]))

    return accuracy, y_pred

# ============================================
# STEP 6: Plot Confusion Matrix
# ============================================
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Diabetes", "Has Diabetes"],
                yticklabels=["No Diabetes", "Has Diabetes"])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    save_path = os.path.join(
        r"D:\Python\PythonProject\SwasthAI\reports\figures",
        f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    )
    plt.savefig(save_path)
    plt.show()
    print(f"  Confusion matrix saved!")

# ============================================
# STEP 7: Save the best model
# ============================================
def save_model(model, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n Model saved to models/{filename}")

# ============================================
# MAIN: Run everything
# ============================================
def run_training():
    print("="*50)
    print("SWASTHAI - Starting Model Training")
    print("="*50)

    # Load data
    X, y = load_processed_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train both models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate both
    lr_accuracy, lr_pred = evaluate_model(
        lr_model, X_test, y_test, "Logistic Regression"
    )
    rf_accuracy, rf_pred = evaluate_model(
        rf_model, X_test, y_test, "Random Forest"
    )

    # Plot confusion matrices
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")

    # Compare and save the best model
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"  Logistic Regression: {lr_accuracy*100:.2f}%")
    print(f"  Random Forest:       {rf_accuracy*100:.2f}%")

    if rf_accuracy >= lr_accuracy:
        print("\n Winner: Random Forest!")
        save_model(rf_model, "best_model.pkl")
        best_model = rf_model
    else:
        print("\n Winner: Logistic Regression!")
        save_model(lr_model, "best_model.pkl")
        best_model = lr_model

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

    return best_model, X_test, y_test

if __name__ == "__main__":
    best_model, X_test, y_test = run_training()