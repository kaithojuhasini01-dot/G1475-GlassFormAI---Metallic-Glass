import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def perform_feature_selection(target_name):
    print(f"\n{'='*40}")
    print(f"--- FEATURE SELECTION FOR: {target_name} ---")
    print(f"{'='*40}\n")
    
    # 1. Loading the regression dataset for the specific target
    input_file = f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\regression_{target_name}_data.csv'
    df = pd.read_csv(input_file)
    
    # 2. Separating features (X) and target (y)
    target_col = f'{target_name}_[K]' # e.g., 'Tl_[K]'
    X = df.drop(columns=['Composition', target_col])
    y = df[target_col]
    
    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Initial number of features: {X.shape[1]}\n")
    
    # --- Variance Filtering ---
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(X_scaled)
    X_var = pd.DataFrame(X_var, columns=X_scaled.columns[var_thresh.get_support()])
    print(f"After Variance Filtering: {X_var.shape[1]} features")
    
    # --- Correlation / F-test Filtering ---
    k_best = SelectKBest(score_func=f_regression, k=30)
    X_corr = k_best.fit_transform(X_var, y)
    X_corr = pd.DataFrame(X_corr, columns=X_var.columns[k_best.get_support()])
    print(f"After Correlation / F-test Filtering: {X_corr.shape[1]} features")
    
    # --- Embedded Method ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_corr, y)
    importances = rf.feature_importances_
    important_features = X_corr.columns[importances > 0.01]
    X_embed = X_corr[important_features]
    print(f"After Embedded Method: {X_embed.shape[1]} features")
    
    # --- Recursive Feature Elimination (RFE) ---
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=10)
    X_rfe = rfe.fit_transform(X_embed, y)
    selected_features = X_embed.columns[rfe.get_support()]
    print(f"After Recursive Feature Elimination: {len(selected_features)} features")
    
    # --- Final Selection (Top 4) ---
    final_features = selected_features[:4] 
    print(f"\n--- FINAL SELECTED FEATURES (Top 4 for {target_name}) ---")
    for i, feat in enumerate(final_features, 1):
        print(f"{i}. {feat}")
        
    # --- Saving Reduced Dataset ---
    final_columns = ['Composition', target_col] + list(final_features)
    final_df = df[final_columns].copy()
    output_file = f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\selected_features_{target_name}.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nReduced dataset saved to '{output_file}'")
    print(f"New dataset shape: {final_df.shape}")

def main():
    targets = ['Tl', 'Tx', 'Tg']
    for target in targets:
        perform_feature_selection(target)

if __name__ == "__main__":
    main()