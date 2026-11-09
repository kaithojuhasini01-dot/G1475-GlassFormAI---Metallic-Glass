import pandas as pd
from pymatgen.core import Composition
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the cleaned dataset
csv_path = os.path.join(project_root, 'data', 'cleaned', 'metallic_glass_cleaned.csv')
df = pd.read_csv(csv_path)

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