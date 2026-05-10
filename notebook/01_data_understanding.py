
# Phase 1: Understanding Our Data
import sys
sys.path.append(r"D:\Python\PythonProject\SwasthAI")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DIABETES_DATA_PATH


# STEP 1: Load the data
df = pd.read_csv(DIABETES_DATA_PATH)

print("✅ Data loaded successfully!")
print("="*50)

# STEP 2: See the first 5 rows
print("\n📋 First 5 rows of our data:")
print(df.head())

# STEP 3: How big is our dataset?
print("\n📐 Shape of dataset (rows, columns):")
print(df.shape)

# STEP 4: What are the column names?
print("\n📌 Column names:")
print(df.columns.tolist())

# STEP 5: What type of data is in each column?
print("\n🔍 Data types of each column:")
print(df.dtypes)

# STEP 6: Basic statistics (min, max, average)
print("\n📊 Basic Statistics:")
print(df.describe())

# STEP 7: Are there any missing values?
print("\n❓ Missing values in each column:")
print(df.isnull().sum())

# STEP 8: How many patients have diabetes vs not?
print("\n📊 Outcome Distribution:")
print(df['Outcome'].value_counts())
print("0 = No Diabetes, 1 = Has Diabetes")

# STEP 9: Plot - How many have diabetes vs not
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title('How many patients have Diabetes vs Not')
plt.xlabel('0 = No Diabetes | 1 = Has Diabetes')
plt.ylabel('Number of Patients')
plt.show()

# STEP 10: Plot - What does glucose look like for diabetic vs non-diabetic?
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Glucose', hue='Outcome', bins=30, palette='Set1')
plt.title('Glucose Level Distribution by Diabetes Outcome')
plt.xlabel('Glucose Level')
plt.ylabel('Number of Patients')
plt.show()

# STEP 11: Plot - Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Age', hue='Outcome', bins=20, palette='Set2')
plt.title('Age Distribution by Diabetes Outcome')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()

# STEP 12: See correlation between all columns
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Between All Features')
plt.show()