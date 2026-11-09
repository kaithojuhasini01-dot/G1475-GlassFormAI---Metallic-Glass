import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition 

def main():

  # 1. Loading the  cleaned dataset
  df = pd.read_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv')

  #2. Convert the formula strings to pymatgen Composition objects first  
  print("Converting formulas to Composition objects...")
  df['Composition'] = df['Composition'].apply(lambda x: Composition(str(x)))

  # 3. Setting up the Matminer featurizer
  # "magpie" is the standard preset that generates the elemental properties (like melting point, electronegativity, etc.)
  ep_feat = ElementProperty.from_preset(preset_name="magpie")

  # 4. Applying it to the 'Composition' column
  print("Generating features... this may take a minute.")
  df_features = ep_feat.featurize_dataframe(df, col_id="Composition", ignore_errors=False)

  # 5. Saving the new feature dataset to the 'features' folder
  df_features.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\numerical_features.csv', index=False)

  print(f"Success! New dataset shape: {df_features.shape}")
  print("Feature names:")
  print(df_features.columns.tolist())

if __name__ == "__main__":
  main()
