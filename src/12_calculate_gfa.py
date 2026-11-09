import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calculate_gamma():
    print("--- CALCULATING GFA PARAMETER (GAMMA) ---")

    results_dir = os.path.join(project_root, 'results')

    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Loading the prediction files saved earlier
    # (We merge them on their index because they are row-aligned from the test set)
    pred_tl = pd.read_csv(os.path.join(results_dir, 'predictions_Tl.csv'))
    pred_tx = pd.read_csv(os.path.join(results_dir, 'predictions_Tx.csv'))
    pred_tg = pd.read_csv(os.path.join(results_dir, 'predictions_Tg.csv'))
    
    # 2. Creating a new DataFrame for the GFA parameter
    gfa_df = pd.DataFrame()
    
    # 3. Grabing the actual/predicted temperatures
    # Note: We use the Predicted_Temp for the calculation
    gfa_df['Predicted_Tx'] = pred_tx['Predicted_Temp']
    gfa_df['Predicted_Tg'] = pred_tg['Predicted_Temp']
    gfa_df['Predicted_Tl'] = pred_tl['Predicted_Temp']

    # Dropping rows where Tl is missing (the last few rows of the dataset)
    gfa_df = gfa_df.dropna(subset=['Predicted_Tl'])

    # 4. Calculating Gamma
    # Formula: Gamma = Tx / (Tg + Tl)
    gfa_df['Gamma'] = gfa_df['Predicted_Tx'] / (gfa_df['Predicted_Tg'] + gfa_df['Predicted_Tl'])
    
    # 5. Saving the result
    output_file = os.path.join(results_dir, 'gfa_parameters.csv')
    gfa_df.to_csv(output_file, index=False)
    
    print(f"Successfully calculated Gamma for {len(gfa_df)} alloys.")
    print(f"Results saved to: {output_file}")

    print("\nFirst 5 calculations (showing the highest GFA alloys):")
    # Sorting by highest Gamma (best glass forming ability)
    top_gfa = gfa_df.sort_values(by='Gamma', ascending=False).head(5)
    print(top_gfa.round(4).to_string())

if __name__ == "__main__":
    calculate_gamma()