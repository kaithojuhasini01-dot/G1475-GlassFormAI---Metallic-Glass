import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("---PREPARING TARGET DATASETS ---\n")
    
    # 1. Loading the final feature dataset 
    input_file = os.path.join(project_root, 'data', 'features', 'final_feature_dataset.csv')
    features_dir = os.path.join(project_root, 'data', 'features')

    os.makedirs(features_dir, exist_ok=True)

    df = pd.read_csv(input_file)

    # ==========================================
    # A. CLASSIFICATION TASK (Dmax > 0)
    # ==========================================
    print("Preparing Classification Task...")
    
    # Createing the Dmax_label column (1 if Dmax > 0, else 0)
    df['Dmax_label'] = (
        pd.to_numeric(df['Dmax_[mm]'], errors='coerce') > 0
    ).astype('Int64')
    
    # Saving the classification dataset
    class_output = os.path.join(features_dir, 'classification_data.csv')
    df.to_csv(class_output, index=False)
    
    print(f"  -> Classification dataset saved to '{class_output}'")
    print(f"  -> Target counts:\n{df['Dmax_label'].value_counts()}\n")

    # ==========================================
    # B. REGRESSION TASKS (Tl, Tx, Tg)
    # ==========================================
    print("Preparing Regression Tasks...")
    
    # Defining the columns we want to keep as input features
    # We drop the targets and metadata like Reference, Zmax, Tx_Flag
    cols_to_drop = ['Tg_[K]', 'Tx_[K]', 'Tl_[K]', 'Dmax_[mm]', 'Reference', 'Zmax_[mm]', 'Tx_Flag','Dmax_label']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    # --- Task 1: Predicting Tl (Liquidus Temperature) ---
    # Important: Drop rows where Tl is missing (NaN)
    df_tl = df.dropna(subset=['Tl_[K]']).copy()
    df_tl = df_tl[feature_cols + ['Tl_[K]']]
    tl_output = os.path.join(features_dir, 'regression_Tl_data.csv')
    df_tl.to_csv(tl_output, index=False)
    print(f"  -> Tl Regression dataset saved. Shape: {df_tl.shape}")

    # --- Task 2: Predicting Tx (Crystallization Temperature) ---
    df_tx = df.dropna(subset=['Tx_[K]']).copy()
    df_tx = df_tx[feature_cols + ['Tx_[K]']]
    tx_output = os.path.join(features_dir, 'regression_Tx_data.csv')
    df_tx.to_csv(tx_output, index=False)
    print(f"  -> Tx Regression dataset saved. Shape: {df_tx.shape}")

    # --- Task 3: Predicting Tg (Glass Transition Temperature) ---
    df_tg = df.dropna(subset=['Tg_[K]']).copy()
    df_tg = df_tg[feature_cols + ['Tg_[K]']]
    tg_output = os.path.join(features_dir, 'regression_Tg_data.csv')
    df_tg.to_csv(tg_output, index=False)
    print(f"  -> Tg Regression dataset saved. Shape: {df_tg.shape}")

    print("\nAll target datasets are ready.")

if __name__ == "__main__":
    main()