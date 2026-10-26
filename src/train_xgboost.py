import pandas as pd
import joblib 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
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
    
    # Setting y to the new column name
    y = df['GFA_label']

    #--- THE FIX: Encoding the labels so they start from 0 ---
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 3. Splitting into Train and Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {X_train.shape[0]} records, testing on {X_test.shape[0]} records.")
    
    # 4. Training XGBoost
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # 5. Making Predictions
    y_pred = model.predict(X_test)
    
    # 6. Evaluating the Model
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=1)
    rec = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    
    print("\n--- RESULTS ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    # 7. Saving Results
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results', exist_ok=True)
    results_df = pd.DataFrame([{
        'Model': 'XGBoost (XGC)',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }])
    results_df.to_csv('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\results\\classification_xgboost_results.csv', index=False)

    #---Saving the model ---
    os.makedirs('C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\models', exist_ok=True)
    joblib.dump(model, 'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\models\\xgboost_classifier.pkl')
    print("\nTrained Model saved to 'C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\models\\xgboost_classifier.pkl'")

if __name__ == "__main__":
    main()