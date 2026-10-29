import pandas as pd
import joblib 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def main():
    print("--- TRAINING XGBOOST CLASSIFIER ---")
    
    # 1. Loading the dataset
    df = pd.read_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\features\\classification_data.csv')
    
    # --- THE FIX: Renaming the column ---
    df.rename(columns={'Dmax_label': 'GFA_label'}, inplace=True)
    
    # 2. Separating Features (X) and Target (y)
    # Updating the drop list to use the new name
    drop_cols = ['Composition', 'Reference', 'Tg_[K]', 'Tx_[K]', 'Tl_[K]', 'Dmax_[mm]', 'Zmax_[mm]', 'Tx_Flag', 'GFA_label']
    X = df.drop(columns=drop_cols)
    
    # Getting original labels (All 1s)
    y_original = df['GFA_label']

    # Creating a temporary target for XGBoost (Map 1 to 0 because XGBoost needs to start at 0)
    y_temp = y_original - 1

    # 3. Splitting into Train and Test (80/20)
    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)

    # Also get the original test labels to save at the end
    _,_,_,y_test_original = train_test_split(X, y_original, test_size=0.2, random_state=42)
    
    print(f"Training on {X_train.shape[0]} records, testing on {X_test.shape[0]} records.")
    
    # 4. Training XGBoost
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train_temp)

    # 5.Checking Training Performance
    y_train_pred_temp = model.predict(X_train)
    train_acc = accuracy_score(y_train_temp, y_train_pred_temp)

    # 6. Testing Performance
    y_pred_temp = model.predict(X_test)
    test_acc = accuracy_score(y_test_temp, y_pred_temp)

    print("\n--- RESULTS ---")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:  {test_acc * 100:.2f}%")

    # Saving the predictions (Actual vs Predicted) for classification
    # Adding 1 back to the predictions so they show as 1 (Glass Forming)
    predictions_df = pd.DataFrame({
        'Actual_Label': y_test_original,
        'Predicted_Label': y_pred_temp + 1
    })
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results', exist_ok=True)
    predictions_df.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\predictions_xgboost.csv', index=False)
    print("\nPredictions saved to 'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\predictions_xgboost.csv'")


    # 7. Evaluating the Model
    prec = precision_score(y_test_temp, y_pred_temp, zero_division=1)
    rec = recall_score(y_test_temp, y_pred_temp, zero_division=1)
    f1 = f1_score(y_test_temp, y_pred_temp, zero_division=1)
    
    print("\n--- RESULTS ---")
    print(f"Accuracy:  {test_acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    # 8. Saving Results
    os.makedirs('C:/Users/Kaithoju Hasini/OneDrive/Documents/metallic_glass_project/results', exist_ok=True)
    results_df = pd.DataFrame([{
        'Model': 'XGBoost (XGC)',
        'Accuracy': test_acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }])
    results_df.to_csv('C:/Users/Kaithoju Hasini/OneDrive/Documents/metallic_glass_project/results/classification_xgboost_results.csv', index=False)

    #---Saving the model ---
    os.makedirs('C:/Users/Kaithoju Hasini/OneDrive/Documents/metallic_glass_project/models', exist_ok=True)
    joblib.dump(model, 'C:/Users/Kaithoju Hasini/OneDrive/Documents/metallic_glass_project/models/xgboost_classifier.pkl')
    print("\nTrained Model saved to 'C:/Users/Kaithoju Hasini/OneDrive/Documents/metallic_glass_project/models/xgboost_classifier.pkl'")

if __name__ == "__main__":
    main()