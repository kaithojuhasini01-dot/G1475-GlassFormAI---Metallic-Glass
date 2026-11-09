import pandas as pd
import re

# Read the cleaned dataset
input_file = r"C:\Users\Kaithoju Hasini\OneDrive\Documents\metallic_glass_project\data\cleaned\metallic_glass_cleaned.csv"
df = pd.read_csv(input_file)

# Regular expression for checking chemical formula format
pattern = r"^([A-Z][a-z]?\d*(\.\d+)?)+$"

invalid_formulas = []

for index, formula in df["Composition"].items():

    # Check for missing or empty composition
    if pd.isna(formula) or str(formula).strip() == "":
        invalid_formulas.append((index, formula, "Missing/empty"))
        continue

    formula = str(formula).strip()

    # Check formula pattern
    if not re.fullmatch(pattern, formula):
        invalid_formulas.append((index, formula, "Invalid format"))

print("----- CHEMICAL FORMULA CHECK -----")

print("Total records:", len(df))
print("Invalid formulas:", len(invalid_formulas))

if len(invalid_formulas) > 0:
    print("\n----- INVALID FORMULAS -----")

    for index, formula, reason in invalid_formulas:
        print("Row:", index)
        print("Composition:", formula)
        print("Reason:", reason)
        print()

else:
    print("\nAll chemical formulas passed the format check.")