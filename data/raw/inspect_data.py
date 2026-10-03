import pandas as pd

df = pd.read_csv("metallic_glass_dataset.csv")

# Basic information
print("----- FIRST 5 ROWS -----")
print(df.head())

print("\n----- DATASET SHAPE -----")
print(df.shape)

print("\n----- COLUMNS -----")
print(df.columns)

# Missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Duplicate records
print("\n----- DUPLICATE RECORDS -----")
print(df.duplicated().sum())

# Negative / invalid numerical values
print("\n----- NEGATIVE VALUES -----")

print("Negative Tg:", (df["Tg_[K]"] < 0).sum())
print("Negative Tx:", (df["Tx_[K]"] < 0).sum())
print("Negative Tl:", (df["Tl_[K]"] < 0).sum())
print("Negative Zmax:", (df["Zmax_[mm]"] < 0).sum())
print("Negative Dmax:", (df["Dmax_[mm]"] < 0).sum())

# Empty compositions
print("\n----- EMPTY COMPOSITIONS -----")
print(df["Composition"].isna().sum())
print((df["Composition"].astype(str).str.strip() == "").sum())

# Temperature relationship checks
print("\n----- TEMPERATURE CHECKS -----")
print("Tg >= Tx:", (df["Tg_[K]"] >= df["Tx_[K]"]).sum())
print("Tx >= Tl:", (df["Tx_[K]"] >= df["Tl_[K]"]).sum())

print("\n----- SUSPICIOUS TEMPERATURE RECORDS -----")

print("\nRecords where Tg >= Tx:")
print(df[df["Tg_[K]"] >= df["Tx_[K]"]])

print("\nRecords where Tx >= Tl:")
print(df[df["Tx_[K]"] >= df["Tl_[K]"]])

print("\n----- DETAILED CHECK OF SUSPICIOUS RECORDS -----")

suspicious = df[
    (df["Tg_[K]"] >= df["Tx_[K]"]) |
    (df["Tx_[K]"] >= df["Tl_[K]"])
]

print(suspicious.to_string())

print("\n----- RECORDS TO VERIFY -----")

print(df.loc[[370, 780], [
    "Composition",
    "Reference",
    "Tg_[K]",
    "Tx_[K]",
    "Tl_[K]",
    "Zmax_[mm]",
    "Dmax_[mm]"
]].to_string())