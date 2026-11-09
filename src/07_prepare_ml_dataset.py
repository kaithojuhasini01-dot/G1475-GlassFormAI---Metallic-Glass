import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_dataset(target_name):
    print(f"\n--- PREPARING ML DATASET FOR: {target_name} ---")
    
    # 1. Loading the reduced dataset 
    input_file = f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\selected_features_{target_name}.csv'
    df = pd.read_csv(input_file)
    
    # 2. Removing invalid and missing target values (just to be safe)
    target_col = f'{target_name}_[K]'
    df = df.dropna(subset=[target_col])
    
    # 3. Separating input features (X) and target (y)
    # We drop 'Composition' because models can't read strings, and we drop the target.
    # This ensures target columns are NOT included in the input features 
    X = df.drop(columns=['Composition', target_col])
    y = df[target_col]
    
    print(f"  Input features (X) shape: {X.shape}")
    print(f"  Target variable (y) shape: {y.shape}")
    
    # 4. Spliting the data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Train set size: {X_train.shape[0]} records")
    print(f"  Test set size:  {X_test.shape[0]} records")
    
    # 5. Saving the final training and testing datasets
    # We combine X and y back together just for saving to CSV
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Creating a folder for the ML datasets if it doesn't exist
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets', exist_ok=True)
    
    train_df.to_csv(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\train_{target_name}.csv', index=False)
    test_df.to_csv(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\test_{target_name}.csv', index=False)
    
    print(f"  -> Saved: C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\train_{target_name}.csv")
    print(f"  -> Saved: C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\test_{target_name}.csv")

if __name__ == "__main__":
    targets = ['Tl', 'Tx', 'Tg']
    for target in targets:
        prepare_dataset(target)
    
    print("\nDatasets prepared.")