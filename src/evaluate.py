# ============================================
# SWASTHAI - Phase 4: Evaluate & Improve
# ============================================

import sys
import os
import pickle

sys.path.append(r"D:\Python\PythonProject\SwasthAI")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from src.config import PROCESSED_DATA_DIR, MODEL_DIR

# ============================================
# STEP 1: Load processed data
# ============================================
def load_data():
    path = os.path.join(PROCESSED_DATA_DIR, "diabetes_processed.csv")
    df = pd.read_csv(path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    print("Data loaded:", df.shape)
    return X, y

# ============================================
# STEP 2: Cross Validation
# Idea: test on 5 different splits, average the result
# More reliable than testing on just 1 split
# ============================================
def cross_validate_model(X, y):
    print("\n Running Cross Validation (5 folds)...")
    print("This tests the model on 5 different train/test splits")

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print(f"\n  Accuracy on each fold:")
    for i, score in enumerate(scores):
        print(f"    Fold {i+1}: {score*100:.2f}%")

    print(f"\n  Average accuracy: {scores.mean()*100:.2f}%")
    print(f"  Std deviation:    {scores.std()*100:.2f}%")
    print("  (lower std = more consistent model)")

    return scores.mean()

# ============================================
# STEP 3: Hyperparameter Tuning
# Idea: try different settings and find the best one
# Like tuning the settings on a camera for best photo
# ============================================
def tune_model(X_train, y_train):
    print("\n Tuning model settings (this may take 1-2 minutes)...")

    # These are the settings we'll try
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }

    # class_weight='balanced' tells AI to pay more attention
    # to diabetes cases since they are fewer in number

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best settings found:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")

    print(f"\n  Best recall score: {grid_search.best_score_*100:.2f}%")

    return grid_search.best_estimator_

# ============================================
# STEP 4: Evaluate the improved model
# ============================================
def evaluate_improved_model(model, X_test, y_test):
    print("\n Evaluating improved model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy:  {accuracy*100:.2f}%")
    print(f"  AUC Score: {auc_score:.3f}  (closer to 1.0 = better)")
    print(f"\n  Detailed Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Diabetes", "Has Diabetes"]
    ))

    return y_pred, y_prob, accuracy, auc_score

# ============================================
# STEP 5: Plot ROC Curve
# ROC curve shows how well AI separates diabetic vs non-diabetic
# AUC = Area Under Curve, closer to 1.0 = perfect model
# ============================================
def plot_roc_curve(y_test, y_prob, auc_score):
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='gray',
             linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve — How well does the model separate cases?')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        r"D:\Python\PythonProject\SwasthAI\reports\figures",
        "roc_curve.png"
    )
    plt.savefig(save_path)
    plt.show()
    print("  ROC curve saved!")

# ============================================
# STEP 6: Feature Importance
# Which columns matter most to the AI?
# ============================================
def plot_feature_importance(model, X):
    print("\n Checking which features matter most...")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importance.plot(kind='barh', color='steelblue')
    plt.title('Feature Importance — What does the AI rely on most?')
    plt.xlabel('Importance Score')
    plt.tight_layout()

    save_path = os.path.join(
        r"D:\Python\PythonProject\SwasthAI\reports\figures",
        "feature_importance.png"
    )
    plt.savefig(save_path)
    plt.show()
    print("  Feature importance chart saved!")

    print("\n  Feature ranking:")
    for feat, score in importance.sort_values(ascending=False).items():
        print(f"    {feat:30s} {score:.4f}")

# ============================================
# STEP 7: Save the improved model
# ============================================
def save_best_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n Improved model saved to models/best_model.pkl")

# ============================================
# MAIN
# ============================================
def run_evaluation():
    print("="*50)
    print("SWASTHAI - Phase 4: Evaluate & Improve")
    print("="*50)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross validate first
    avg_cv_score = cross_validate_model(X, y)

    # Tune and get improved model
    best_model = tune_model(X_train, y_train)

    # Evaluate improved model
    y_pred, y_prob, accuracy, auc_score = evaluate_improved_model(
        best_model, X_test, y_test
    )

    # Graphs
    plot_roc_curve(y_test, y_prob, auc_score)
    plot_feature_importance(best_model, X)

    # Save
    save_best_model(best_model)

    print("\n" + "="*50)
    print("Phase 4 Complete!")
    print("="*50)

    return best_model

if __name__ == "__main__":
    best_model = run_evaluation()