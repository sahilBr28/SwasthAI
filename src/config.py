import os

# Base directory of the entire project
BASE_DIR = r"D:\Python\PythonProject\SwasthAI"

# Data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# File paths
DIABETES_DATA_PATH = os.path.join(RAW_DATA_DIR, "diabetes data.csv")

# Model path
MODEL_DIR = os.path.join(BASE_DIR, "models")