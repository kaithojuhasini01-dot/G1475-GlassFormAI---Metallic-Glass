# Note: This script was executed on Google Colab (Linux) to bypass a 
# known Windows multiprocessing error with pymatgen and pymoo. 
# The paths below are configured for local Windows VS Code execution.

import pandas as pd
import numpy as np
import joblib
import os
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(project_root, 'models')
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

# 1. Loading saved models 
etr_tl = joblib.load(os.path.join(models_dir, 'etr_Tl_model.pkl'))
etr_tx = joblib.load(os.path.join(models_dir, 'etr_Tx_model.pkl'))
etr_tg = joblib.load(os.path.join(models_dir, 'etr_Tg_model.pkl'))
ep_feat = ElementProperty.from_preset(preset_name="magpie")

# Defining feature lists globally so both the optimizer and saving loop can see them
features_tl = ['MagpieData mean MeltingT', 'MagpieData avg_dev Column', 'MagpieData avg_dev GSvolume_pa', 'MagpieData mode GSmagmom']
features_tx = ['MagpieData mean MeltingT', 'MagpieData avg_dev Column', 'MagpieData avg_dev GSvolume_pa', 'MagpieData mode GSmagmom']
features_tg = ['MagpieData mean MeltingT', 'MagpieData mean Column', 'MagpieData mean Electronegativity', 'MagpieData avg_dev GSvolume_pa']

class CompositionOptimization(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=1, xl=np.array([5, 5, 5]), xu=np.array([20, 20, 10]))

    def _evaluate(self, x, out, *args, **kwargs):
        cu, ni, al = x
        zr = 100 - (cu + ni + al)
        
        if zr <= 0:
            out["G"] = [1.0] 
            out["F"] = [[10000, 10000]] 
            return
            
        out["G"] = [0.0] 
        formula = f"Zr{zr:.1f}Cu{cu:.1f}Ni{ni:.1f}Al{al:.1f}"
        
        try:
            comp = Composition(formula)
            df_temp = pd.DataFrame({'Composition': [comp]})
            df_features = ep_feat.featurize_dataframe(df_temp, col_id="Composition", ignore_errors=True)
            
            # Predicting using the correct features for each model
            pred_tl = etr_tl.predict(df_features[features_tl])[0]
            pred_tx = etr_tx.predict(df_features[features_tx])[0]
            pred_tg = etr_tg.predict(df_features[features_tg])[0]
            
            obj1 = -pred_tx
            obj2 = - (1 / (pred_tl + pred_tg))
            
            out["F"] = [obj1, obj2]
            
        except Exception as e:
            out["F"] = [10000, 10000]

print("Starting NSGA-II Optimization... (This will take about 2 minutes)")
# Parameters set to 50 population, 20 generations
algorithm = NSGA2(pop_size=50) 

res = minimize(CompositionOptimization(), algorithm, ('n_gen', 20), seed=42, verbose=True)

print("\nOptimization Complete! Saving Pareto Front...")
results_df = pd.DataFrame(res.X, columns=['Cu_wt%', 'Ni_wt%', 'Al_wt%'])
results_df['Zr_wt%'] = 100 - (results_df['Cu_wt%'] + results_df['Ni_wt%'] + results_df['Al_wt%'])

preds = []
for index, row in results_df.iterrows():
    formula = f"Zr{row['Zr_wt%']:.1f}Cu{row['Cu_wt%']:.1f}Ni{row['Ni_wt%']:.1f}Al{row['Al_wt%']:.1f}"
    comp = Composition(formula)
    df_temp = pd.DataFrame({'Composition': [comp]})
    df_features = ep_feat.featurize_dataframe(df_temp, col_id="Composition", ignore_errors=True)
    
    # Using the correct features again for the final saving loop
    tl = etr_tl.predict(df_features[features_tl])[0]
    tx = etr_tx.predict(df_features[features_tx])[0]
    tg = etr_tg.predict(df_features[features_tg])[0]
    preds.append({'Formula': formula, 'Pred_Tl': tl, 'Pred_Tx': tx, 'Pred_Tg': tg})

final_results = pd.concat([results_df, pd.DataFrame(preds)], axis=1)
# Saving results to csv
output_file = os.path.join(results_dir, 'nsga2_optimized_compositions.csv')
final_results.to_csv(output_file, index=False)

print(f"Results saved to '{output_file}'")
print("\nTop 5 Optimized Alloys:")
print(final_results.head())