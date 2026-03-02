import pandas as pd

df = pd.read_csv("diabetes data.csv")

print(df.head())
print("\nShape of dataset:", df.shape)
print("\nColumn Names:", df.columns)