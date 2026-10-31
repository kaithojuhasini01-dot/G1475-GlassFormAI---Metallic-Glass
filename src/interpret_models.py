import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import os

def interpret_model(target_name):
    print(f"\n--- INTERPRETING MODEL FOR: {target_name} ---")
    
    # 1. Loading the trained model
    model_path = f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\models\\etr_{target_name}_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return
    model = joblib.load(model_path)
    
    # 2. Loading the test data
    test_df = pd.read_csv(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\test_{target_name}.csv')
    target_col = f'{target_name}_[K]'
    X_test = test_df.drop(columns=[target_col])
    
    # 3. Feature Importance Plot
    importances = model.feature_importances_
    features = X_test.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
    plt.title(f'Feature Importance for {target_name} Prediction')
    plt.tight_layout()
    plt.savefig(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\feature_importance_{target_name}.png')
    plt.close()
    print(f"  Saved Feature Importance plot.")

    # 4. PDP and ICE Plots (For the Top 2 Important Features)
    top_features = importance_df['Feature'].head(2).tolist()
    
    for feature in top_features:
        plt.figure(figsize=(8, 5))
        
        # Generating PDP and ICE
        display = PartialDependenceDisplay.from_estimator(
            model, X_test, [feature], kind='both', subsample=50, 
            random_state=42, ax=plt.gca()
        )
        plt.title(f'PDP & ICE for {feature} ({target_name})')
        plt.tight_layout()
        
        # Cleaning up feature name for file saving
        safe_feature_name = feature.replace(" ", "_").replace("/", "_")
        plt.savefig(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\pdp_ice_{target_name}_{safe_feature_name}.png')
        plt.close()
        print(f"  Saved PDP/ICE plot for {feature}")

if __name__ == "__main__":
    targets = ['Tl', 'Tx', 'Tg']
    
    for target in targets:
        interpret_model(target)
        
    print("\nInterpretation Complete! Check the 'results' folder for all images.")