import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")

input_file = os.path.join(data_dir, "raw","train-00000-of-00001.parquet")
output_file = os.path.join(data_dir, "raw", "metallic_glass_dataset.csv")
df = pd.read_parquet(input_file)
df.to_csv(output_file, index=False)
print("CSV  file created successfully.")
print(df.shape)
print(df.head())