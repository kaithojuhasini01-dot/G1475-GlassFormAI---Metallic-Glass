import pandas as pd
from pymatgen.core import Composition

# Load the cleaned dataset
df = pd.read_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv')

# Check the first 5 compositions
print("First 5 compositions from the CSV:")
for i in range(5):
    formula = df['Composition'].iloc[i]
    print(f"Row {i}: {repr(formula)}")
    try:
        comp = Composition(str(formula))
        print("  -> Valid")
    except Exception as e:
        print(f"  -> INVALID: {e}")