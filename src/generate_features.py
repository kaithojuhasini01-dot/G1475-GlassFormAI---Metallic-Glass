import pandas as pd
from matminer.featurizers.composition import ElementProperty

def main():

  # 1. Loading the  cleaned dataset
  df = pd.read_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv') 

  # 2. Setting up the Matminer featurizer
  # "magpie" is the standard preset that generates the elemental properties (like melting point, electronegativity, etc.)
  ep_feat = ElementProperty.from_preset(preset_name="magpie")

  # 3. Applying it to the 'Composition' column
  print("Generating features... this may take a minute.")
  df_features = ep_feat.featurize_dataframe(df, col_id="Composition", ignore_errors=True)

  # 4. Saving the new feature dataset to the 'features' folder
  df_features.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\numerical_features.csv', index=False)

  print(f"Success! New dataset shape: {df_features.shape}")
  print("Feature names:")
  print(df_features.columns.tolist())

if __name__ == "__main__":
  main()
