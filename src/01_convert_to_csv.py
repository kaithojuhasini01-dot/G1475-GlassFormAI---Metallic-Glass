import pandas as pd
df = pd.read_parquet("c:\\Users\\Kaithoju Hasini\\Downloads\\train-00000-of-00001.parquet")
df.to_csv("metallic_glass_dataset.csv", index=False)
print("CSV  file created successfully.")
print(df.shape)
print(df.head())