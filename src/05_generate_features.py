import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition 
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():

  # 1. Loading the  cleaned dataset
  input_file = os.path.join(project_root, 'data', 'cleaned', 'metallic_glass_cleaned.csv')
  output_file = os.path.join(project_root, 'data', 'features', 'numerical_features.csv')

  os.makedirs(os.path.dirname(output_file), exist_ok=True)

  print("Loading cleaned dataset...")
  df = pd.read_csv(input_file)

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
  df_features.to_csv(output_file, index=False)

  print(f"Success! Processed dataset saved as '{output_file}'")
  print(f"New dataset shape: {df_features.shape}")
  print("Feature names:")
  print(df_features.columns.tolist())

if __name__ == "__main__":
  main()
