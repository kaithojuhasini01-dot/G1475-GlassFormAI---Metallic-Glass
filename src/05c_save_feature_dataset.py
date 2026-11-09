import pandas as pd
import numpy as np

def main():
    print("--- SAVING THE FEATURE DATASET ---\n")
    
    # 1. Loading the generated features
    df = pd.read_csv(r'C:\Users\Kaithoju Hasini\OneDrive\Documents\metallic_glass_project\data\features\numerical_features.csv')

    # 2. Checking the number of rows and columns
    print(f"1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Displaying the feature names (First 5 and last 5)
    print(f"\n2. Feature Names (Sample):")
    print(f"   First 5: {df.columns[:5].tolist()}")
    print(f"   Last 5:  {df.columns[-5:].tolist()}")

    # 4. Checking missing (NaN) and infinite values
    missing_vals = df.isnull().sum().sum()
    # Selecting only numerical columns for infinite check
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_vals = np.isinf(df[num_cols]).sum().sum()
    print(f"\n3. Missing Values (NaN): {missing_vals}")
    print(f"   Infinite Values: {inf_vals}")

    # 5. Verifying the features are numerical
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\n4. Non-numerical columns (should only be identifiers/formula):")
    print(f"   {non_numeric_cols}")

    # 6. Preserving the original chemical formula
    # The 'Composition' column is our formula. We will keep it as an ID column.
    print(f"\n5. Formula column preserved: 'Composition'")

    # 7. Keeping the target columns separately
    # Defining which columns are our targets 
    target_cols = ['Tg_[K]', 'Tx_[K]', 'Tl_[K]', 'Dmax_[mm]']
    
    # Creating a separate DataFrame for targets
    df_targets = df[['Composition'] + target_cols].copy()
    
    # Creating a separate DataFrame for features (Formula + 132 Matminer features)
    # Droping the targets and other metadata (Reference, Zmax, Tx_Flag) from the feature set
    feature_cols = [col for col in df.columns if col not in target_cols + ['Reference', 'Zmax_[mm]', 'Tx_Flag']]
    df_features = df[feature_cols].copy()

    # 8. Saving the generated features in CSV format
    print("\n6. Saving final datasets to data/features/...")
    df.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\final_feature_dataset.csv', index=False)
    df_features.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\X_features.csv', index=False)
    df_targets.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\y_targets.csv', index=False)
    
    print("   -> Saved: final_feature_dataset.csv (Everything combined)")
    print("   -> Saved: X_features.csv (Formula + Numerical Features only)")
    print("   -> Saved: y_targets.csv (Formula + Target Columns only)")
    print("\n--- FEATURE DATASET SAVED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()