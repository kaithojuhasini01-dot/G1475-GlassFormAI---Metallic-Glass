import pandas as pd
from pymatgen.core import Composition

# ---------------------------------------------------------
# 1. LOAD YOUR  CLEANED CSV FILE
# ---------------------------------------------------------
input_file = 'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv'  

df = pd.read_csv(input_file)

formula_column = 'Composition' 

print(f"Original dataset shape: {df.shape}")
print(f"Preview of original data:\n{df.head()}\n")

# ---------------------------------------------------------
# 2. DEFINE THE FUNCTION TO EXTRACT ELEMENTAL FRACTIONS
# ---------------------------------------------------------
def get_fractions(formula_str):
    try:
        # Convert the string to a pymatgen Composition object
        comp = Composition(str(formula_str))
        
        # Get the fractional composition (normalized so total = 1)
        fractions = comp.fractional_composition.get_el_amt_dict()
        return fractions
    except Exception as e:
        # If there is an error (e.g., bad formula), return an empty dictionary
        # and print the error so you know which row failed
        print(f"Error processing formula '{formula_str}': {e}")
        return {}

# ---------------------------------------------------------
# 3. APPLY THE FUNCTION TO ALL 998 RECORDS
# ---------------------------------------------------------
print("Processing formulas... (this may take a few seconds)")
# This creates a new column with dictionaries of elemental fractions
df['elemental_fractions'] = df[formula_column].apply(get_fractions)

# ---------------------------------------------------------
# 4. EXPAND THE DICTIONARIES INTO SEPARATE COLUMNS
# ---------------------------------------------------------
# This turns the dictionary into individual columns for each element (e.g., 'Zr', 'Cu', 'Al')
elemental_df = df['elemental_fractions'].apply(pd.Series)

# Fill missing values with 0 (if a formula doesn't have an element, it should be 0)
elemental_df = elemental_df.fillna(0)

# ---------------------------------------------------------
# 5. COMBINE AND SAVE
# ---------------------------------------------------------
# Combine the original dataframe with the new elemental columns
final_df = pd.concat([df.drop(columns=['elemental_fractions']), elemental_df], axis=1)

# Save to a new CSV file so you don't overwrite your original data
output_file = 'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\dataset_with_elemental_composition.csv'
final_df.to_csv(output_file, index=False)

print(f"\nSuccess! Processed dataset saved as '{output_file}'")
print(f"New dataset shape: {final_df.shape}")
print(f"Preview of new data:\n{final_df.head()}")