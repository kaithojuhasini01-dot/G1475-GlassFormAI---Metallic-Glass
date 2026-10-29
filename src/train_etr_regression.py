import pandas as pd
import joblib
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_evaluate(target_name):
    print(f"\n{'='*40}")
    print(f"TRAINING ETR MODEL FOR: {target_name}")
    print(f"{'='*40}")
    
    # 1. Loading the train and test datasets 
    train_df = pd.read_csv(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\train_{target_name}.csv')
    test_df = pd.read_csv(f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\ml_datasets\\test_{target_name}.csv')
    
    target_col = f'{target_name}_[K]'
    
    # 2. Separating X and y
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # 3. Initializing and training the ETR model
    etr_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    etr_model.fit(X_train, y_train)
    
    # 4. Making predictions
    y_train_pred = etr_model.predict(X_train)
    y_test_pred = etr_model.predict(X_test)
    
    # 5. Calculating metrics for BOTH sets
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Only calculating MAE/MSE on the test set for final reporting
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print(f"Results for {target_name}:")
    print(f"  R² Score (Train): {r2_train:.4f}")
    print(f"  R² Score (Test):  {r2_test:.4f}")
    print(f"  MAE (Test):       {mae_test:.2f} K")
    print(f"  MSE (Test):       {mse_test:.2f} K²")

    # 6. Save the predictions to CSV
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results', exist_ok=True)
    predictions_df = pd.DataFrame({
        'Actual_Temp': y_test,
        'Predicted_Temp': y_test_pred
    })
    pred_filename = f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\predictions_{target_name}.csv'
    predictions_df.to_csv(pred_filename, index=False)
    print(f"  Predictions saved as: {pred_filename}")

    # 6. Saving the trained model
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\models', exist_ok=True)
    model_filename = f'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\models\\etr_{target_name}_model.pkl'
    joblib.dump(etr_model, model_filename)
    print(f"  Model saved as: {model_filename}")
    
    return {
        'Target': target_name,
        'Model': 'ETR',
        'Train_R2': round(r2_train, 4),
        'Test_R2': round(r2_test, 4),
        'Test_MAE': round(mae_test, 2),
        'Test_MSE': round(mse_test, 2)
        
    }

if __name__ == "__main__":
    targets = ['Tl', 'Tx', 'Tg']
    all_results = []
    
    for target in targets:
        result = train_and_evaluate(target)
        all_results.append(result)

    # Saving summary results   
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results', exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\etr_regression_results.csv', index=False)
    
    
    print("\nAll ETR models trained, evaluated, and saved!")
    print("Summary results saved to 'results/etr_regression_results.csv'")