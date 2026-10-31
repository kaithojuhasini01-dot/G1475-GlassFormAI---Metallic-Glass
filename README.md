# GlassFormAI - Machine Learning Aided Prediction of Glass-Forming Ability of Metallic Glass

## 🚀 Project Overview

This project reproduces and extends the machine learning pipeline from the paper "Machine Learning Aided Prediction of Glass-Forming Ability of Metallic Glass" (Liu et al., 2023). We successfully built predictive models to determine whether an alloy can form a metallic glass and to predict its key thermal properties (Tl, Tx, Tg). 

Beyond just prediction, we performed comprehensive **Model Interpretation** using Feature Importance, Partial Dependence Plots (PDP), and Individual Conditional Expectation (ICE) plots to explain *how* the models make their decisions. Our final ETR regression models achieved an exceptional **R² score > 0.96**, outperforming the base paper's reported results.

## 👥 Team Members
- Hasini Kaithoju (26P85A0502)
- Venu (26P85A0507)
- Abhigna (26P85A0505)
- Krishna (26P85A0513)
- Spoorthi (26P85A0511)
- Laya (26P85A6603)

**Guide:** Mr. P. Sreenivas

## 📂 Repository Structure
- `data/` : Contains raw, cleaned, features and ml_datasets (train/test splits).
- `documentation/` : Daily logs, reports, feature selection analysis, screenshots and source evidence.
- `models/` : Saved trained models(.pkl files) for classification and regression.
- `results/` : Model evaluatioon metrics (CSV), predictions, and interpretation plots(PDP/ICE).
- `src/` : Python scripts for data processing, feature generation, selection, training, and interpretation.

## 🚀 Progress Tracker
- [x] Milestone 1: Data processing and elemental fraction generation.
- [x] Day 2:  Generated 140 numerical features using Matminer.
- [x] Day 3: Fixed the NaN Error in numerical feature generation.
- [x] Day 5: Documented feature comparison between paper and matminer.
- [x] Day 6: Saved final feature dataset and separated targets.
- [x] Day 7: Prepared target datasets.
- [x] Day 8: Prepared target datasets (Classification & Regression) and fixed data leakage.
- [x] Day 9: Completed Feature Selection(Variance,F-test,Embedded,RFE,Exhaustive Search) for Tl,Tx, and Tg.
- [x] Day 10: Prepared Machine Learning datasets (80/20 Train-Test split) for Regression models.
- [x] Day 11: Implemented and saved XGBoost Classification model.Documented dataset limitations.
- [x] Day 12: Trained and saved ETR Regression models(R^2 > 0.96), saved predictions, and all documentation logs.
- [x] Day 13: Completed Model Interpretation(Feature Importance, PDP, and ICE plots) for all targets.

## 🛠️ Technologies Used
- **Programming Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Materials Science & Feature Engineering:** Pymatgen, Matminer
- **Machine Learning & Evaluation:** Scikit-Learn, XGBoost(Classification), ETR(Regression)
- **Model Interpretation & Visualization:** Matplotlib, Seaborn
- **Model Saving & Deployment:** Joblib
- **Version Control:** Git & GitHub
