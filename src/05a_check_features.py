import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

features_path = os.path.join(project_root, 'data', 'features', 'numerical_features.csv')

# Load the features dataset
df = pd.read_csv(features_path)

# 1. Print the shape (rows, columns)
print("Dataset Shape:", df.shape)

# 2. Print the names of the first 10 feature columns
print("\nFirst 10 Feature Names:")
print(df.columns[8:18].tolist())

# 3. Print the actual numbers for the first 5 rows of these features
print("\nActual Feature Numbers (First 5 rows):")
print(df.iloc[0:5, 8:13]) 